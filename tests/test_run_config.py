"""One config file drives a local run and a cluster job alike."""

import json
import sys
from unittest import mock

import pytest

from spikeshpc import slurm_env
from spikeshpc.cli import main
from spikeshpc.config import DEFAULT_PIPELINE, deep_merge


def paths(values):
    """Compare by path, not by platform separator: the config holds
    cluster (POSIX) paths that pathlib rewrites when read on Windows."""
    return [str(v).replace("\\", "/") for v in values]


@pytest.fixture
def captured(monkeypatch):
    """Intercept run_pipeline so we see exactly what the CLI resolved."""
    seen = {}
    monkeypatch.setattr("spikeshpc.cli.run_pipeline", lambda **kw: seen.update(kw))
    return seen


def write(tmp_path, cfg):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(cfg))
    return p


def test_run_block_supplies_everything(tmp_path, captured):
    cfg = write(tmp_path, {
        "run": {
            "phys_paths": ["/data/a", "/data/b"],
            "output_dir": "/out",
            "tmp_dir": "/scratch",
            "skip_statescoring": True,
            "skip_sorting": True,
        },
        "sorting": {"nblocks": 5},
    })
    main(["--pipeline_config", str(cfg)])

    assert paths(captured["phys_path"]) == ["/data/a", "/data/b"]
    assert captured["output_dir"] == "/out"
    assert captured["tmp_dir"] == "/scratch"
    assert captured["skip_statescoring"] is True
    assert captured["skip_sorting"] is True
    assert captured["skip_preprocessing"] is False
    assert captured["sorting"] == {"nblocks": 5}
    # "run" is consumed by the CLI, not passed through as a pipeline override
    assert "run" not in captured


def test_command_line_overrides_the_config(tmp_path, captured):
    cfg = write(tmp_path, {
        "run": {"phys_paths": ["/data/a"], "output_dir": "/out"},
    })
    main(["/data/override", "--output_dir", "/elsewhere",
          "--pipeline_config", str(cfg)])

    assert paths(captured["phys_path"]) == ["/data/override"]
    assert paths([captured["output_dir"]]) == ["/elsewhere"]


def test_a_skip_flag_can_be_added_but_not_removed_at_the_command_line(tmp_path, captured):
    """store_true cannot express "off", so the config's True must survive."""
    cfg = write(tmp_path, {"run": {"phys_paths": ["/d"], "skip_sorting": True}})
    main(["--pipeline_config", str(cfg)])
    assert captured["skip_sorting"] is True

    captured.clear()
    cfg2 = write(tmp_path, {"run": {"phys_paths": ["/d"]}})
    main(["--skip_sorting", "--pipeline_config", str(cfg2)])
    assert captured["skip_sorting"] is True


def test_no_recordings_anywhere_is_an_error(tmp_path, capsys):
    cfg = write(tmp_path, {"run": {"output_dir": "/out"}})
    with pytest.raises(SystemExit):
        main(["--pipeline_config", str(cfg)])
    assert "run.phys_paths" in capsys.readouterr().err


def test_recordings_still_work_with_no_config_at_all(captured):
    main(["/data/a", "/data/b", "--output_dir", "/out"])
    assert paths(captured["phys_path"]) == ["/data/a", "/data/b"]
    assert paths([captured["output_dir"]]) == ["/out"]


def test_run_is_a_recognised_top_level_key():
    assert "run" in DEFAULT_PIPELINE


# ── the slurm side reads the same file ──────────────────────────────────
def _pipeline(cfg):
    return deep_merge(DEFAULT_PIPELINE, cfg)


def test_bind_paths_cover_recordings_output_scratch_and_tracking():
    binds = slurm_env.bind_paths(_pipeline({
        "run": {
            "phys_paths": ["/data/phys/s1", "/data/phys/s2"],
            "output_dir": "/data/out",
            "tmp_dir": "/scratch/tmp",
        },
        "state_scoring": {"movement": {
            "optitrack_csv": "/behavior/5800941/{session}/{session}.csv",
        }},
    }))
    assert "/data/phys/s1" in binds
    assert "/data/phys/s2" in binds
    assert "/data/out" in binds
    assert "/scratch/tmp" in binds
    # the template root, stopping above the {session} placeholder
    assert "/behavior/5800941" in binds


def test_bind_paths_drop_children_of_other_binds():
    binds = slurm_env.bind_paths(_pipeline({
        "run": {
            "phys_paths": ["/data/s1", "/data/s2"],
            "output_dir": "/data/s1/kilosort",
            "tmp_dir": "/data",
        },
    }))
    assert binds == ["/data"], binds


def test_explicit_bind_paths_win():
    binds = slurm_env.bind_paths(_pipeline({
        "run": {"phys_paths": ["/data/s1"], "bind_paths": ["/mnt/everything"]},
    }))
    assert binds == ["/mnt/everything"]


def _eval_in_bash(text):
    """What bash actually ends up with after `eval`-ing the assignments."""
    import shutil
    import subprocess

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    script = (
        f"{text}\n"
        'printf "OUT=%s\\n" "$SPIKESHPC_OUTPUT_DIR"\n'
        'printf "TMP=%s\\n" "$SPIKESHPC_TMP_DIR"\n'
        'for p in ${SPIKESHPC_PHYS[@]+"${SPIKESHPC_PHYS[@]}"}; do printf "PHYS=%s\\n" "$p"; done\n'
        'for p in ${SPIKESHPC_BIND[@]+"${SPIKESHPC_BIND[@]}"}; do printf "BIND=%s\\n" "$p"; done\n'
    )
    r = subprocess.run([bash, "-c", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = {"PHYS": [], "BIND": []}
    for line in r.stdout.splitlines():
        key, _, value = line.partition("=")
        if key in ("PHYS", "BIND"):
            out[key].append(value)
        else:
            out[key] = value
    return out


def test_shell_assignments_survive_eval_including_spaces(tmp_path):
    """The slurm script `eval`s this, so paths with spaces must stay intact."""
    cfg = _pipeline({
        "run": {
            "phys_paths": ["/data/a b", "/data/c"],
            "output_dir": "/out dir",
            "tmp_dir": "/scratch",
        },
    })
    got = _eval_in_bash(slurm_env.shell_assignments(cfg))
    assert got["OUT"] == "/out dir"
    assert got["TMP"] == "/scratch"
    assert got["PHYS"] == ["/data/a b", "/data/c"]
    assert "/data/a b" in got["BIND"]


def test_empty_optional_paths_eval_to_empty_strings():
    cfg = _pipeline({"run": {"phys_paths": ["/data/a"]}})
    got = _eval_in_bash(slurm_env.shell_assignments(cfg))
    assert got["OUT"] == "" and got["TMP"] == ""
    assert got["PHYS"] == ["/data/a"]


def test_slurm_env_main_prints_for_eval(tmp_path, capsys):
    p = write(tmp_path, {"run": {"phys_paths": ["/data/a"], "output_dir": "/out"}})
    assert slurm_env.main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "SPIKESHPC_PHYS=" in out and "/data/a" in out
    assert "SPIKESHPC_BIND=(" in out
    assert out.count("\n") == 4


def test_slurm_env_main_rejects_bad_usage(capsys):
    assert slurm_env.main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_the_example_config_is_valid_and_complete(tmp_path, captured):
    """The shipped example must actually run, not just parse."""
    import pathlib

    example = (pathlib.Path(__file__).resolve().parent.parent
               / "slurm" / "pipeline_config.example.json")
    cfg = json.loads(example.read_text())
    assert set(cfg) <= set(DEFAULT_PIPELINE), set(cfg) - set(DEFAULT_PIPELINE)

    main(["--pipeline_config", str(example)])
    assert captured["phys_path"], "example config yields no recordings"
    assert captured["output_dir"]

    binds = slurm_env.bind_paths(deep_merge(DEFAULT_PIPELINE, cfg))
    assert binds, "example config yields no bind paths"
