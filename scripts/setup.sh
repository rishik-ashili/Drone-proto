#!/usr/bin/env bash
# One-time setup on Linux / macOS:  ./scripts/setup.sh  [--train] [--px4]
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"
$PY -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
$PY -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements-dev.txt
for arg in "$@"; do
  case "$arg" in
    --train) .venv/bin/pip install -r requirements-train.txt ;;
    --px4)   .venv/bin/pip install -r requirements-px4.txt ;;
  esac
done
.venv/bin/python -c "import cv2, onnxruntime, yaml; print('OK  OpenCV', cv2.__version__, '| onnxruntime', onnxruntime.__version__)"
echo
echo "Done. Start the demo with:"
echo "  source .venv/bin/activate && python scripts/run_dashboard.py --open"
