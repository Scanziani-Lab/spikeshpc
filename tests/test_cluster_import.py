"""The cluster image has no OpenCV or IPython, so the pipeline has to import without them.

The slurm jobs run hpc_load_sort_post.py inside containers/si_kilosort4.def.
Its `from spikeshpc.cli import main` runs spikeshpc/__init__.py, which imports
the widget and notebook-step modules along with everything else. The one that
reads video, the OptiTrack heading video widget, therefore imports cv2 only when
it opens one, and nothing imports IPython at module level: ipykernel is not in
the image, and spikeinterface[full] does not bring IPython in.
"""

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_the_cli_imports_without_opencv_or_ipython():
    # A fresh interpreter, because this one has imported spikeshpc already.
    # None in sys.modules makes `import cv2` fail even where OpenCV is
    # installed, and running from the repo root imports this checkout, the way
    # hpc_load_sort_post.py does on the cluster.
    code = (
        "import sys; sys.modules['cv2'] = None; sys.modules['IPython'] = None; "
        "from spikeshpc.cli import main"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert result.returncode == 0, result.stderr
