#!/usr/bin/env python3
"""Entry point for running the pipeline straight out of a checkout.

Equivalent to `python -m spikeshpc ...` or the `spikeshpc` console script,
but needs no install: the slurm jobs bind-mount this repo into the container
and run it in place.
"""

import sys
from pathlib import Path

# This file sits at the repo root, one level above the package, so adding its
# own directory is what makes `import spikeshpc` resolve without an install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spikeshpc.cli import main  # noqa: E402
from spikeshpc.pipeline import run_pipeline  # noqa: E402,F401  (re-export)

if __name__ == "__main__":
    main()
