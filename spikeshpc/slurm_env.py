"""Emit shell assignments from a pipeline config, for the slurm wrapper.

The job script needs a few things before it can launch the container -- which
directories to bind, what to mkdir -- and those are the same paths the
pipeline reads from the config. Deriving them here keeps one file as the
single source of truth instead of restating every path in the slurm script.

    eval "$(python -m spikeshpc.slurm_env pipeline_config.json)"
"""

import json
import shlex
import sys
from pathlib import Path

from .config import DEFAULT_PIPELINE, deep_merge


# These paths describe the cluster filesystem and are consumed by a POSIX
# shell, so they are handled as strings throughout. Passing them through
# pathlib would rewrite '/data/a' as '\data\a' whenever the config is read on
# Windows, which is exactly what happens when the same file is used locally.
def _parent(path: str) -> str:
    cut = max(path.rfind("/"), path.rfind("\\"))
    return path[:cut] if cut > 0 else path


def _template_root(template: str) -> str | None:
    """Directory part of a '{session}' path template, above the placeholder."""
    brace = template.find("{")
    if brace == -1:
        return _parent(template)
    cut = max(template.rfind("/", 0, brace), template.rfind("\\", 0, brace))
    return template[:cut] if cut > 0 else None


def _is_ancestor(parent: str, child: str) -> bool:
    parent = parent.rstrip("/\\")
    return child == parent or child.startswith(parent + "/") or child.startswith(
        parent + "\\"
    )


def bind_paths(pipeline: dict) -> list[str]:
    """Directories the container needs, derived from the configured paths.

    Explicit run.bind_paths wins; otherwise the recordings, the output and
    scratch directories, and the roots of the OptiTrack templates are used.
    """
    run = pipeline.get("run") or {}
    if run.get("bind_paths"):
        return [str(p) for p in run["bind_paths"]]

    candidates: list[str] = []
    for path in run.get("phys_paths") or []:
        path = str(path)
        # a recording may be given as a directory or as a raw binary file
        candidates.append(_parent(path) if path[-4:].lower() in
                          (".bin", ".dat", ".raw") else path)
    for key in ("output_dir", "tmp_dir"):
        if run.get(key):
            candidates.append(str(run[key]))

    movement = ((pipeline.get("state_scoring") or {}).get("movement") or {})
    for key in ("optitrack_csv", "frame_times"):
        if movement.get(key):
            root = _template_root(str(movement[key]))
            if root:
                candidates.append(root)

    # Drop paths already covered by an ancestor, so apptainer gets a short list.
    unique = sorted({c for c in candidates if c}, key=len)
    kept: list[str] = []
    for path in unique:
        if not any(_is_ancestor(k, path) for k in kept):
            kept.append(path)
    return kept


def shell_assignments(pipeline: dict) -> str:
    run = pipeline.get("run") or {}
    phys = [str(p) for p in (run.get("phys_paths") or [])]
    lines = [
        f"SPIKESHPC_OUTPUT_DIR={shlex.quote(str(run.get('output_dir') or ''))}",
        f"SPIKESHPC_TMP_DIR={shlex.quote(str(run.get('tmp_dir') or ''))}",
        "SPIKESHPC_PHYS=({})".format(" ".join(shlex.quote(p) for p in phys)),
        "SPIKESHPC_BIND=({})".format(
            " ".join(shlex.quote(p) for p in bind_paths(pipeline))
        ),
    ]
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1:
        print("usage: python -m spikeshpc.slurm_env <pipeline_config.json>",
              file=sys.stderr)
        return 2
    with open(argv[0]) as f:
        pipeline = deep_merge(DEFAULT_PIPELINE, json.load(f))
    print(shell_assignments(pipeline))
    return 0


if __name__ == "__main__":
    sys.exit(main())
