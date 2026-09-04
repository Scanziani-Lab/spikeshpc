"""Stage 4: read kilosort4's output back and build the sorting analyzer."""

from pathlib import Path

import spikeinterface.full as si

from .config import ANALYZER_NAME, SORTER_DIRNAME


def postprocess(rec, output_dir: Path, extensions: dict, bad_channel_ids=None):
    """Read kilosort4's output back into spikeinterface and save the analyzer.

    `bad_channel_ids` are dropped from the recording first so the analyzer is
    built on the same channels kilosort4 sorted -- otherwise templates and
    center-of-mass unit locations would be computed over dead channels that
    the sorter never saw.
    """
    results_dir = output_dir / SORTER_DIRNAME
    if not (results_dir / "spike_times.npy").exists():
        raise FileNotFoundError(
            f"No kilosort4 output in {results_dir} -- run the sorting stage first."
        )
    if bad_channel_ids:
        keep = [c for c in rec.channel_ids if str(c) not in set(bad_channel_ids)]
        print(f"    excluding {len(bad_channel_ids)} bad channel(s) from the analyzer")
        rec = rec.select_channels(keep)

    sorting = si.read_kilosort(folder_path=results_dir)
    print(f"    loaded {len(sorting.unit_ids)} units")

    analyzer = si.create_sorting_analyzer(
        recording=rec,
        sorting=sorting,
        folder=(output_dir / ANALYZER_NAME),
        format="zarr",
        overwrite=True,
    )
    analyzer.compute(extensions)
    return analyzer
