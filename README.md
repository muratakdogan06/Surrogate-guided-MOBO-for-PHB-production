# Surrogate-guided multi objective Bayesian optimization for growth-coupled polyhydroxybutyrate production in Priestia megaterium DSM319*

> **Full title:** urrogate-guided multi objective Bayesian optimization for growth-coupled polyhydroxybutyrate production in Priestia megaterium DSM319

---

## 1. Project Description

This repository provides a complete, reproducible computational pipeline that
couples **genome-scale metabolic modelling (GEM / FBA)** with **machine-learning
surrogate models** and a **Bayesian-style active-learning loop** to
systematically optimise polyhydroxybutyrate (PHB) production in *Priestia
megaterium*.



## 2. Installation

```bash
# Option A — conda (recommended)
conda env create -f environment.yml
conda activate phaopt
pip install -e .

# Option B — pip only
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

**Python ≥ 3.10** is required.

---

## 3. Pipeline Execution Order

Each script is self-contained; run them sequentially:

| # | Script | Description |
|--:|--------|-------------|
| 1 | `python scripts/01_extend_model.py` | Add PHA objective to the GEM |
| 2 | `python scripts/02_validate_model.py` | Validate model integrity & baseline FBA |
| 3 | `python scripts/03_generate_design_space.py` | Combinatorial metabolic designs |
| 4 | `python scripts/04_run_fba_dataset.py` | FBA simulations → ML dataset |
| 5 | `python scripts/05_train_surrogate.py` | Train surrogate ML models |
| 6 | `python scripts/06_run_active_learning.py` | AL loop + baselines |
| 7 | `python scripts/07_pareto_analysis.py` | Pareto frontier analysis |
| 8 | `python scripts/08_shap_analysis.py` | SHAP interpretability |
| 9 | `python scripts/09_flux_maps.py` | Flux rerouting maps |
| 10 | `python scripts/10_generate_figures.py` | (Re)generate all Figures 1–7 |

---

## 4. Configuration

All model-specific IDs are exposed in YAML — **nothing is hardcoded**.

| File | Purpose | Edit first? |
|------|---------|:-----------:|
| `configs/model_config.yaml` | SBML path, precursor IDs, biomass / exchange rxn IDs | **YES** |
| `configs/candidate_reactions.yaml` | Pathway groups for KOs / up-regs | **YES** |
| `configs/conditions.yaml` | Environmental conditions (exchange overrides) | maybe |
| `configs/active_learning.yaml` | AL hyper-parameters, surrogate settings | optional |

---

## 5. Expected Inputs

| Input | Location | Format |
|-------|----------|--------|
| Genome-scale metabolic model | `models/B_megaterium.xml` | SBML / XML |

Place your SBML file at the path set in `configs/model_config.yaml → model_path`.

---

## 6. Expected Outputs

### Models

| File | Description |
|------|-------------|
| `models/model_with_PHA.xml` | Extended SBML model |
| `models/model_extension_report.json` | Modification report |
| `models/surrogate_*.joblib` | Serialised ML surrogates |

### Data

| File | Description |
|------|-------------|
| `data/processed/design_space.parquet` | Design table |
| `data/processed/fba_results.parquet` | Full FBA results |
| `data/processed/ml_dataset.parquet` | ML-ready feature matrix |

### Results

| File | Description |
|------|-------------|
| `results/tables/validation_report.json` | Model checks |
| `results/tables/surrogate_metrics.csv` | ML performance |
| `results/tables/al_history.csv` | AL convergence trace |
| `results/tables/al_best_design.json` | Best design |
| `results/tables/pareto_front.csv` | Pareto-optimal designs |
| `results/tables/pareto_metrics.json` | Hypervolume comparison |
| `results/tables/shap_feature_importance.csv` | SHAP ranking |
| `results/tables/flux_comparison.csv` | Pathway fluxes |
| `results/tables/pipeline_summary.json` | End-to-end summary |

---

## 7. Project Structure

```
makale-2/
├── environment.yml
├── pyproject.toml
├── README.md
├── configs/
│   ├── model_config.yaml
│   ├── conditions.yaml
│   ├── candidate_reactions.yaml
│   └── active_learning.yaml
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   └── phaopt/
│       ├── __init__.py
│       ├── utils.py
│       ├── io.py
│       ├── plotting.py
│       ├── model_extension.py
│       ├── model_validation.py
│       ├── perturbation_space.py
│       ├── simulation.py
│       ├── dataset.py
│       ├── train.py
│       ├── predict.py
│       ├── uncertainty.py
│       ├── active_learning.py
│       ├── pareto.py
│       ├── shap_analysis.py
│       ├── flux_maps.py
│       └── evaluation.py
└── scripts/
    ├── 01_extend_model.py
    ├── 02_validate_model.py
    ├── 03_generate_design_space.py
    ├── 04_run_fba_dataset.py
    ├── 05_train_surrogate.py
    ├── 06_run_active_learning.py
    ├── 07_pareto_analysis.py
    ├── 08_shap_analysis.py
    ├── 09_flux_maps.py
    └── 10_generate_figures.py
```

---


## 8. Citation

> [Akdoğan, M]. Surrogate-guided multi objective Bayesian optimization for growth-coupled
> polyhydroxybutyrate production in *Priestia megaterium* DSM319 *[Metabolic Engineering]*, [2026].
> DOI: [pending].

---

## License

This project is provided for academic and research purposes under the MIT
licence.
