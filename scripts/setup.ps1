# One-time setup on Windows (PowerShell):  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -r requirements-dev.txt
.\.venv\Scripts\python -c "import cv2, onnxruntime, yaml; print('OK  OpenCV', cv2.__version__)"
Write-Host ""
Write-Host "Done. Start the demo with:"
Write-Host "  .\.venv\Scripts\activate; python scripts\run_dashboard.py --open"
