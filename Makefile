# Convenience targets (Linux / macOS). Windows: see README.
PY ?= python3

setup:            ## create .venv and install the simulator
	$(PY) -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-dev.txt

demo:             ## live dashboard on http://localhost:8080
	.venv/bin/python scripts/run_dashboard.py --open

sim:              ## headless investor mission
	.venv/bin/python scripts/run_sim.py

video:            ## record demo.mp4
	.venv/bin/python scripts/run_sim.py --record demo.mp4

test:             ## unit + mission tests
	.venv/bin/python -m pytest -q

benchmark:        ## exit-criteria campaign -> docs/benchmark_results.md
	.venv/bin/python scripts/benchmark.py

train:            ## regenerate dataset + retrain the YOLO landmark model
	.venv/bin/pip install -r requirements-train.txt
	.venv/bin/python scripts/generate_dataset.py
	.venv/bin/python scripts/train_landmark_model.py

.PHONY: setup demo sim video test benchmark train
