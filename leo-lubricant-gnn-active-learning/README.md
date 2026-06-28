# leo-lubricant-gnn-active-learning

`leo-lubricant-gnn-active-learning` is a Python research repository for environment-conditioned, degradation-aware graph neural networks with Gaussian-process uncertainty estimation and active learning for low-volatility space lubricant discovery.

The package implements a multi-task workflow for seven targets:

- vapor pressure
- viscosity
- friction coefficient
- wear rate
- mass loss
- viscosity change
- degradation index

The included CSV files are demonstration inputs for exercising the pipeline. They are not validated scientific datasets, and users need curated experimental data before drawing real scientific conclusions.

## Features

- molecular graph construction from SMILES with an RDKit-backed path and a fallback tokenizer path
- environment feature conditioning for thermo-oxidative and tribological context
- dense message-passing GNN for multi-task regression
- Gaussian-process uncertainty estimation on learned graph embeddings
- desirability scoring and acquisition functions for active learning
- preprocessing, training, evaluation, candidate ranking, and plotting scripts

## Repository Layout

```text
leo-lubricant-gnn-active-learning/
├── configs/
├── data/
├── docs/
├── notebooks/
├── results/
├── scripts/
├── src/
└── tests/
```

## Installation

### Pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Conda

```bash
conda env create -f environment.yml
conda activate leo-lubricants
pip install -e .
```

## Quickstart

```bash
python scripts/01_preprocess_data.py --help
python scripts/01_preprocess_data.py
python scripts/02_train_model.py
python scripts/03_evaluate_model.py
python scripts/04_run_active_learning.py
python scripts/05_generate_figures.py
```

The default configuration uses small settings so the sample workflow runs quickly on demonstration data.

## Data Format

Training rows are expected to contain:

- `molecule_id`
- `smiles`
- environment columns:
  - `temperature_c`
  - `vacuum_pa`
  - `radiation_gy`
  - `oxygen_ppm`
  - `load_n`
  - `sliding_speed_mps`
- target columns:
  - `vapor_pressure`
  - `viscosity`
  - `friction_coefficient`
  - `wear_rate`
  - `mass_loss`
  - `viscosity_change`
  - `degradation_index`

Candidate rows use the same identifier, structure, and environment columns, but target columns are optional.

## Preprocessing

The preprocessing pipeline:

1. validates required schema
2. canonicalizes and filters molecules
3. computes molecular descriptors
4. computes environment-conditioned features
5. scales continuous descriptors and environment variables
6. produces reproducible train, validation, and test splits

Outputs are written under `data/processed/`.

## Training

The training workflow:

1. loads processed splits
2. builds molecular graphs
3. trains the multi-task GNN
4. extracts latent embeddings
5. fits one Gaussian process per target for predictive uncertainty
6. saves the model bundle to `results/trained_models/`

## Evaluation

Evaluation reports:

- RMSE
- MAE
- R²
- Spearman correlation
- uncertainty calibration summaries

Metrics are written to `results/metrics/`, and figure generation writes into `results/figures/`.

## Active Learning

Candidate ranking combines:

- predictive uncertainty from Gaussian processes
- normalized desirability toward low-volatility, low-friction, and low-degradation behavior
- configurable exploitation and exploration weights

Ranked candidates are written to `results/candidate_rankings/`.

## Limitations

- The bundled data are demonstration-only and not suitable for scientific claims.
- The dense-graph GNN is designed for small to medium datasets and simple experimentation.
- Gaussian-process uncertainty scales best when latent dimensionality and dataset size remain moderate.
- Real deployment should include domain-specific validation, uncertainty stress testing, and external benchmarking.

## Citation

If you use this repository in academic work, please cite the project metadata in `CITATION.cff`.

