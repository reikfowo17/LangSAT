<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)">
  </a>
</p>

<h1 align="center"><b>CS106 - LangSAT Reproduce Project</b></h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge" />
  <img src="https://img.shields.io/badge/stable--baselines3-PPO-0091EA?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Gymnasium-SmartSAT%20Env-43B02A?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Graph-Message%20Passing-8E44AD?style=for-the-badge" />
  <img src="https://img.shields.io/badge/SATzilla-48%20Features-F39C12?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

# LangSAT Reproduce


Reproduction of the paper **LangSAT: A Novel Framework Combining NLP and Reinforcement Learning for SAT Solving** (arXiv:2512.04374v1).

The project is organized into separate modules for the CDCL baseline, the SmartSAT Gym environment, the graph message-passing policy, training pipeline, evaluation, and natural-language to DIMACS conversion.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Install](#install)
- [Run](#run)
- [Configuration](#configuration)
- [Core Modules](#core-modules)
- [Runtime Data](#runtime-data)
- [Development Notes](#development-notes)
- [License](#license)

---

## Features

- Pure Python CDCL solver with VSIDS heuristic, conflict learning, and backtracking.
- Hard-budget guard for solver (conflicts, seconds, decisions) to keep wall-clock fair across runs.
- SmartSAT Gym environment exposing variable selection as the action space on top of the CDCL search.
- Action-mask wrapper that prunes already-assigned variables and out-of-range indices.
- Bipartite graph message-passing policy (`TrueGNNSATExtractor`) that fuses signed clause-variable edges, assignment state, clause state, and 48 SATfeatPy global features.
- PPO / MaskablePPO training with reproducible seeds, train/test split, and reward callback.
- 48-feature SATzilla extractor via SATfeatPy with local-search probing, on-disk feature cache, and DIMACS normalization for SATLIB headers.
- Evaluation harness that compares SmartSAT and baseline on identical instances, including win rate, median time, decision/conflict ratio, and invalid-action rate.
- Natural-language to DIMACS pipeline using a SymPy logic transformer and an OpenAI-backed converter for free-form English text.
- Reward shaping aligned with the paper: absolute clause score on `uf20-91` (20 variables, 91 clauses).
- Smoke and full run profiles that can be toggled through environment variables.
- Kaggle reproduce notebook that clones the repo, installs dependencies, and runs training plus evaluation end-to-end.

---

## Repository Structure

```text
.
|-- notebooks/                    Reproduction notebooks
|   `-- LangSAT_Kaggle_Reproduce.ipynb   Kaggle end-to-end pipeline
|
|-- paper/                        Reference paper PDF
|
|-- docs/                         Design notes and planning documents
|
|-- results/                      Generated outputs (model, plots, metrics, split metadata)
|
|-- data/                         Dataset folder (uf20-91 CNF instances)
|
|-- src/                          Source modules
|   -- -- __init__.py
|   -- -- cdcl_baseline.py           CDCL solver with VSIDS heuristic
|   -- -- smartsat_env.py            Gym environment for SmartSAT RL agent
|   -- -- policy.py                  Graph message-passing PPO feature extractor
|   -- -- training_pipeline.py       PPO training loop and reward callback
|   -- -- evaluate.py                Win rate and solving time evaluation
|   -- -- satfeat_adapter.py         SATfeatPy 48-feature adapter and cache
|   `-- lang2logic.py              Natural language to DIMACS pipeline
|
|-- requirements.txt              Python dependencies
|-- README.md                     Project documentation
 `-- .gitignore                    Git ignore rules (cache, data, results, IDE)
```

---

## Requirements

Recommended environment:

- Python 3.10 or newer
- pip for dependency installation
- Git for cloning the SATfeatPy companion repository
- For Kaggle runs: a Kaggle environment with GPU or CPU runtime and persistent `/kaggle/working` storage
- Optional: OpenAI API key for the English-text branch of the Lang2Logic pipeline

Python dependencies are pinned in `requirements.txt`. The notable packages are:

- `torch` (>= 2.0) for the graph message-passing policy
- `stable-baselines3` (>= 2.0) and `sb3-contrib` (>= 2.0) for PPO and MaskablePPO
- `gymnasium` (>= 0.29) for the SmartSAT environment
- `sympy` (>= 1.12) and `lark` (>= 1.1) for the Lang2Logic logic parser
- `openai` (>= 1.0) for the optional English to CNF converter
- `networkx`, `python-louvain`, `powerlaw` for SATfeatPy local-search dependencies
- `matplotlib`, `seaborn`, `pandas` for plots and metrics tables

---

## Install

### 1. Clone the repository

```powershell
git clone https://github.com/reikfowo17/LangSAT.git
cd LangSAT
```

### 2. Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Clone SATfeatPy for 48-feature extraction

```powershell
git clone --depth 1 https://github.com/bprovanbessell/SATfeatPy.git C:\\tools\\SATfeatPy
$env:LANGSAT_SATFEATPY_DIR = "C:\\tools\\SATfeatPy"
```

### 4. Prepare the dataset

The repository expects the SATLIB `uf20-91` corpus (20 variables, 91 clauses, 1000 instances) under the `data/` directory. Either drop the `.cnf` files directly into `data/` or attach the `heon29/uf20-91` Kaggle dataset, then point the data root through the `LANGSAT_DATA_DIR` environment variable.

### 5. Optional: enable English to CNF conversion

```powershell
$env:OPENAI_API_KEY = "sk-..."
```

---

## Run

### Local training and evaluation

After installing dependencies, configuring SATfeatPy, and placing the dataset under `data/`, run the full reproduction locally with:

```powershell
python src\training_pipeline.py
python src\evaluate.py
```

The training script will train a PPO agent with `TrueGNNSATExtractor`, save the model under `results/`, and emit a reward curve plot. The evaluation script then loads the trained model and the same test split, runs the CDCL baseline and SmartSAT on every instance, and writes `results/metrics.json`, `results/runtime_breakdown.csv`, and several diagnostic plots.

### Smoke profile (default for laptops)

A faster smoke run caps the dataset at 20 instances and uses 2,048 PPO steps. It is enabled by default in `training_pipeline.py` when running outside Kaggle:

```powershell
$env:LANGSAT_KAGGLE_RUN_MODE = "smoke"
python src\training_pipeline.py
python src\evaluate.py
```

### Full paper profile

The full profile uses 100,000 PPO steps, the 800/200 train/test split, and the SATfeatPy 48-feature extractor. Enable it with:

```powershell
$env:LANGSAT_KAGGLE_RUN_MODE = "full"
$env:LANGSAT_TOTAL_STEPS = "100000"
python src\training_pipeline.py
python src\evaluate.py
```

### Kaggle reproduction

Use the `notebooks/LangSAT_Kaggle_Reproduce.ipynb` notebook on Kaggle. The first cell clones the repository, installs requirements, and clones SATfeatPy. Subsequent cells perform the smoke or full reproduction based on `LANGSAT_KAGGLE_RUN_MODE` and write all outputs to `/kaggle/working/results/`.

### Lang2Logic quick start

Convert propositional text directly:

```python
from lang2logic import Lang2Logic
from cdcl_baseline import solve_file

pipeline = Lang2Logic()
expr = pipeline.parse_expression("And(Or(A, B), Not(A))")
pipeline.save_dimacs(expr, "results/example.cnf")
sat, seconds = solve_file("results/example.cnf")
```

Convert free-form English text (requires `OPENAI_API_KEY`):

```python
from lang2logic import Lang2Logic

pipeline = Lang2Logic()
result = pipeline.convert("If A then B. A.")
with open("results/from_text.cnf", "w", encoding="utf-8") as f:
    f.write(result["dimacs"]["dimacs"])
```

---

## Configuration

All runtime knobs are read from environment variables, so the same scripts work in local Python, Kaggle, or CI without code changes.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LANGSAT_DATA_DIR` | `./data` (or `/kaggle/input/datasets/heon29/uf20-91` on Kaggle) | Folder containing the `uf20-91` CNF instances. |
| `LANGSAT_OUTPUT_DIR` | `./results` (or `/kaggle/working/results` on Kaggle) | Folder where the model, metrics, plots, and split metadata are written. |
| `LANGSAT_MODEL_PATH` | `<OUTPUT_DIR>/smartsat_model` | Output path for the trained PPO/MaskablePPO artifact (saved with `.zip`). |
| `LANGSAT_SPLIT_PATH` | `<OUTPUT_DIR>/data_split.json` | Train/test split JSON produced by `training_pipeline.py` and consumed by `evaluate.py`. |
| `LANGSAT_TOTAL_STEPS` | `100000` | Total PPO environment steps for the full paper profile. |
| `LANGSAT_SPLIT_STRATEGY` | `sorted` | Either `sorted` (paper default) or `shuffled` for a random split. |
| `LANGSAT_SPLIT_SEED` | `42` | RNG seed used when `LANGSAT_SPLIT_STRATEGY=shuffled`. |
| `LANGSAT_TRAIN_LOG_INTERVAL` | `5000` | Step interval for the reward logger callback. |
| `LANGSAT_TRAIN_HEARTBEAT_SECONDS` | `60` | Wall-clock interval for verbose heartbeat prints during training. |
| `LANGSAT_ENV_MAX_STEPS` | `200` (`N_VARS * 10`) | Hard cap on environment steps per episode. |
| `LANGSAT_INVALID_ACTION_PENALTY` | `2.0` | Additional reward penalty subtracted when the agent picks an already-assigned or out-of-range variable. |
| `LANGSAT_SOLVER_MAX_CONFLICTS` | `250` | Per-instance conflict budget enforced by the CDCL baseline. |
| `LANGSAT_SOLVER_MAX_SECONDS` | `5.0` | Per-instance wall-clock budget enforced by the CDCL baseline. |
| `LANGSAT_SOLVER_MAX_DECISIONS` | `20000` | Per-instance decision budget enforced by the CDCL baseline. |
| `LANGSAT_POLICY_MODE` | `rl` | SmartSAT inference mode (`rl` for the trained policy). |
| `LANGSAT_USE_SEARCH_TIME` | `0` | When `1`, evaluator reports search time without policy overhead. Must be `0` for paper-like reproduce. |
| `LANGSAT_SATFEATPY_DIR` | empty | Absolute path to a local clone of the SATfeatPy repository used for the 48-feature extractor. |
| `LANGSAT_FEATURE_CACHE_DIR` | empty | Folder for the on-disk JSON feature cache; speeds up repeat runs. |
| `LANGSAT_SATFEATPY_FULL_LOCAL_SEARCH` | `1` | When `1`, requires the full SATzilla local-search features (40 of 48). Set to `0` to allow a partial feature set. |
| `OPENAI_API_KEY` | empty | OpenAI API key used by `lang2logic` for free-form English to CNF conversion. |

Hard-coded constants in source (not environment-driven):

| Constant | Value | Source |
| --- | --- | --- |
| `N_VARS` | `20` | `smartsat_env.py` |
| `N_CLAUSES` | `91` | `smartsat_env.py` |
| `N_GLOBAL` / `N_SATZILLA_FEATURES` | `48` | `smartsat_env.py`, `satfeat_adapter.py` |
| `SATZILLA_FEATURE_ORDER` | 40 SATzilla features + 8 local-search probes | `satfeat_adapter.py` |
| `LEARNING_RATE` | `0.0002` | `training_pipeline.py` |
| `TRAIN_RATIO` | `0.8` (800 train / 200 test on uf20-91) | `training_pipeline.py` |
| `SEED` | `42` | `training_pipeline.py` |
| `CHECKPOINT_FREQ` | `10000` | `training_pipeline.py` |

---

## Core Modules

| Module | Responsibility |
| --- | --- |
| `src/cdcl_baseline.py` | Pure Python CDCL solver with VSIDS heuristic, conflict learning, non-chronological backtracking, and hard budgets for conflicts, seconds, and decisions. |
| `src/smartsat_env.py` | Gym environment that exposes variable selection as the action space, builds the observation (clause / variable / global features), and computes rewards aligned with the paper. |
| `src/policy.py` | Bipartite graph message-passing PPO feature extractor (`TrueGNNSATExtractor`) that fuses signed clause-variable edges, assignment state, clause state, and the 48 SATfeatPy global features. |
| `src/training_pipeline.py` | PPO / MaskablePPO training loop with reproducible seeds, train/test split, reward callback, and reward-curve plotting. |
| `src/evaluate.py` | Evaluation harness that compares SmartSAT and the CDCL baseline on identical instances, including win rate, median time, decision/conflict ratio, and invalid-action rate. |
| `src/satfeat_adapter.py` | SATfeatPy 48-feature adapter with local-search probing, on-disk JSON feature cache, and DIMACS normalization for SATLIB headers. |
| `src/lang2logic.py` | Natural-language to DIMACS pipeline using a SymPy logic transformer for structured input and an OpenAI-backed converter for free-form English text. |

---

## Runtime Data

- The trained PPO/MaskablePPO model is written to `<OUTPUT_DIR>/smartsat_model.zip`.
- The train/test split is written to `<OUTPUT_DIR>/data_split.json` (metadata includes the run profile, train ratio, split strategy, and split seed).
- Training reward trace is written to `<OUTPUT_DIR>/training_rewards.json` and plotted to `<OUTPUT_DIR>/training_reward_curve.png`.
- Evaluation outputs the per-instance table to `<OUTPUT_DIR>/eval_results.csv` and aggregate metrics to `<OUTPUT_DIR>/metrics.json`.
- Plots generated during evaluation: `<OUTPUT_DIR>/solving_time_comparison.png` and `<OUTPUT_DIR>/time_distribution.png`.
- A human-readable summary is written to `<OUTPUT_DIR>/summary.txt`.
- On Kaggle, all outputs land under `/kaggle/working/results/`.
- The SATfeatPy JSON feature cache lives in `LANGSAT_FEATURE_CACHE_DIR` when set, or in a per-file location next to the CNF instance otherwise.

Expected output layout after a full run:

```text
results/
  smartsat_model.zip
  data_split.json
  training_rewards.json
  training_reward_curve.png
  eval_results.csv
  metrics.json
  solving_time_comparison.png
  time_distribution.png
  summary.txt
  tb_logs/                  TensorBoard logs from PPO
```

---

## Development Notes

- All code lives under `src/`. The package is importable both as `src.*` (when `src/` is on `PYTHONPATH`) and as flat top-level modules (the scripts add `src/` to `sys.path` at startup).
- Do not commit build outputs, caches, datasets, or local IDE files. The repository `.gitignore` covers `__pycache__/`, `results/`, `data/`, and `.venv/`.
- When adding a new module, keep it under `src/` and update the **Core Modules** table in this README.
- When adding a new environment variable, expose it through `os.environ.get(...)` with a sensible default, document the default here in **Configuration**, and add a smoke test in the Kaggle notebook.
- Reward shaping is paper-aligned: absolute clause score on `uf20-91` (20 variables, 91 clauses). Keep the dataset fixed to `uf20-91` unless the paper text is updated.
- The default split strategy is `sorted` to match the paper. Switch to `shuffled` only when the change is documented and reproducible (use `LANGSAT_SPLIT_SEED`).
- The 48-feature extractor depends on a local clone of SATfeatPy at `LANGSAT_SATFEATPY_DIR`. Without it the environment falls back to zero global features and the policy degrades.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
