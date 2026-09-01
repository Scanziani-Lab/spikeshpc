#!/usr/bin/env python3
"""Backwards-compatible entry point -- the pipeline now lives in spikepipe/.

Kept so existing slurm scripts that call
`python hpc_load_sort_post.py ...` keep working. Equivalent to
`python -m spikepipe ...`.
"""

import sys
from pathlib import Path

# Allow running this file directly from a checkout that is not pip-installed,
# which is how the slurm jobs invoke it inside the container.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from spikepipe.cli import main  # noqa: E402
from spikepipe.pipeline import run_pipeline  # noqa: E402,F401  (re-export)

if __name__ == "__main__":
    main()
