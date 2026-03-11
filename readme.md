# CausalStream Prototype — README

> **Note:** this repository contains a research prototype implementing **our method** for robust, low-latency streaming inference on windowed network traces. The code is intended for experimentation and research; it uses synthetic data by default. Please follow your institution’s ethics and legal guidelines before running any experiments on real or sensitive traffic.

---

## Overview

This repository provides a compact, end-to-end Python prototype implementing the core components described in the paper. It focuses on streaming-friendly preprocessing, a causal front-end, a lightweight selective state-space encoder, meta-learned confounder estimation with a front-door-style adjustment, simple counterfactual regularization, and evaluation utilities.

The implementation is purposely small and readable so it can be adapted or replaced with production-grade building blocks (e.g., a high-performance Mamba-2 operator) as needed.

Key components:

* windowed, causal preprocessing and adaptive normalization
* causal convolutional front-end + streaming state operator
* MetaNet confounder estimator + front-door adjustment
* contrastive regularizer for early/partial traces
* simple Stackelberg / MINE utilities (estimator skeleton)
* training / evaluation / visualization scripts and unit tests

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml           # example config file (training hyperparameters)
├── src/
│   └── causalstream/
│       ├── __init__.py
│       ├── models.py          # model definitions
│       ├── data.py            # synthetic dataset & windowing helpers
│       ├── utils.py           # helpers, normalizer, seeding
│       ├── train.py           # training loop
│       ├── eval.py            # simple evaluation script
│       └── viz.py             # small plotting helpers
├── tests/
│   └── test_core.py           # pytest unit tests for core functionality
└── samples/                    # optional place for example outputs & figures
```

> If you prefer a different layout (for example `src/` omitted), adjust `PYTHONPATH` or imports accordingly.

---

## Quickstart

1. Create a Python environment (recommended: Python 3.10+).

```bash
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
# .venv\Scripts\activate     # Windows (PowerShell)
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run a short demo training (uses synthetic data):

From the project root (assumes `src` is on PYTHONPATH):

```bash
# Linux / macOS
PYTHONPATH=src python -m causalstream.train --config configs/default.yaml

# Windows (PowerShell)
set PYTHONPATH=src
python -m causalstream.train --config configs/default.yaml
```

4. Run evaluation with a saved checkpoint (or without one to use random weights):

```bash
PYTHONPATH=src python -m causalstream.eval --config configs/default.yaml --ckpt runs/demo/causalstream_epoch1.pt
```

5. Run unit tests:

```bash
pip install -r requirements.txt
pytest -q
```

---

## Configuration

All runtime hyperparameters are read from the YAML config file passed to `train.py` / `eval.py`. A minimal `configs/default.yaml` should include keys such as:

```yaml
seed: 42
device: auto
data:
  n_sites: 10
  traces_per_site: 200
  w_ms: 44
  seq_windows: 128

model:
  in_channels: 6
  frontend_hidden: 128
  encoder_dim: 128
  ssm_hidden: 256
  confound_dim: 64
  num_classes: 10

train:
  batch_size: 32
  lr: 2e-3
  epochs: 3

logging:
  save_dir: runs/demo

alpha_mu: 0.99
alpha_s: 0.995
contrast_tau: 0.1
lambda_c: 0.5
lambda_cf: 0.1
```

Adjust values for your experiments.

---

## Data

* The prototype ships with a synthetic dataset generator (`SyntheticWFTraceDataset`) for quick experiments and testing.
* `data.windowize_trace()` converts packet-level traces (time, signed_size) into a fixed-size windowed matrix `M` with per-window counts and statistics.
* If you want to use your own traces, prepare them in the expected `(time_ms, signed_size)` per-packet format, and call `windowize_trace()` before feeding data to the model. Ensure anonymization and compliance with legal/ethical rules.

---

## Running experiments safely

* The code contains a simulation-only training loop and synthetic-data defaults. If you intend to test any active querying or active-defense simulations, do so only in controlled, instrumented testbeds and after obtaining all necessary approvals.
* The repository intentionally restricts active/querying functionality to simulation mode in the training script. Do not run queries against third-party or production networks.

---

## Visualization

* `viz.py` and the helper plotting functions in `src/causalstream/viz.py` produce demo plots (training curves, embeddings, activation heatmaps). These are small, dependency-light helpers and avoid heavy t-SNE/UMAP defaults for fast execution.
* Example: after training, use the saved pooled features to create an embedding plot with the helper functions.

---

## Testing

* Unit tests are provided under `tests/test_core.py`. They exercise the synthetic dataset, windowing, the adaptive normalizer, and a forward/backward pass through the model.
* Run `pytest -q` from the repository root after installing dependencies.

---

## Extending / Production notes

* The `SelectiveStateSpace` module in `models.py` is a lightweight prototype. For production or high-performance research, replace it with an optimized implementation (e.g., a production SSM or a Mamba-2 wrapper).
* Replace synthetic dataset generation with a loader for your real traces. Keep preprocessing causal and streaming-friendly.
* If you integrate GPU-accelerated libraries or custom CUDA kernels, ensure reproducibility and deterministic behavior where necessary.

---

## Ethics & Responsible Use

* This project deals with traffic fingerprinting research. Use of these techniques can have privacy implications. Only run experiments on traces you are authorized to use, and follow institutional review processes.
* Active probing, querying, or effects that interfere with third-party systems are out of scope for the default code and must only be executed in managed, consented testbeds.

---

## Contributing

* Contributions are welcome (bug fixes, replacement of prototype components with optimized versions, improved datasets, better evaluation suites).
* Please provide tests for new functionality and respect the ethical guidance above.

