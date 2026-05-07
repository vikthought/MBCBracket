#!/usr/bin/env bash
# setup.sh — create .venv and install dependencies from requirements.txt.
#
# One-time setup: `./setup.sh`. After this `./run_all.sh` will auto-detect
# .venv/ and use it as Python.
#
# Override the bootstrap interpreter with PY_BOOT (must be Python 3.10+):
#   PY_BOOT=/opt/homebrew/bin/python3.11 ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY_BOOT="${PY_BOOT:-python3}"
if [ ! -x "$(command -v "$PY_BOOT")" ]; then
    echo "$PY_BOOT not found on PATH; set PY_BOOT to a python3.10+ binary." >&2
    exit 1
fi

if [ ! -d .venv ]; then
    echo "Creating .venv with $PY_BOOT ..."
    "$PY_BOOT" -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

echo
echo "Versions:"
.venv/bin/python - <<'PY'
import numpy, scipy, sklearn, matplotlib, pandas, tabulate
from hdbscan import HDBSCAN  # noqa: F401
print(f"  numpy        {numpy.__version__}")
print(f"  scipy        {scipy.__version__}")
print(f"  scikit-learn {sklearn.__version__}")
print(f"  matplotlib   {matplotlib.__version__}")
print(f"  pandas       {pandas.__version__}")
print(f"  tabulate     {tabulate.__version__}")
print(f"  hdbscan      ok")
PY

echo
echo "Done. You can now run ./run_all.sh"
