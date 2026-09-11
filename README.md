# MPM-Nested-CV: Mineral Prospectivity Mapping with Nested Cross-Validation

A comprehensive framework for Mineral Prospectivity Mapping (MPM) using nested cross-validation with conformal prediction for uncertainty quantification.

## Features

- **Nested Cross-Validation**: Rigorous evaluation with spatial block cross-validation to prevent data leakage
- **Conformal Prediction**: Cross-conformal calibration targeting 85% positive-class coverage, approximate under spatial autocorrelation
- **Positive-Unlabeled Learning**: BaggingPU-XGBoost for learning from positive and unlabeled samples
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization
- **SHAP Analysis**: Feature importance and model interpretability
- **Comprehensive Visualization**: Spatial maps, deposit capture curves, and uncertainty zones

## Project Structure

```
mpm_nested_cv_release/
├── DATA-LICENSE.md                  # Per-source data licence terms
├── figures/                         # Figure_2a.pdf (external schematic input for generate_figure2.py)
├── config/                          # Configuration files
│   ├── config.yaml                  # Main configuration
│   └── hyperparameter_ranges.yaml   # Hyperparameter search spaces
├── data/                            # Input data
│   ├── Yilgarn_GIS_DB.csv          # Main dataset
│   └── boundary/                    # Shapefile boundaries
├── scripts/
│   ├── orchestration/               # Main workflow scripts
│   │   ├── run_all_outer_folds.py
│   │   ├── run_single_outer_fold.py
│   │   └── run_final_aggregation.py
│   ├── fold_generation/             # CV fold creation
│   │   ├── create_outer_folds.py
│   │   └── create_inner_folds.py
│   ├── optuna_optimization/         # Hyperparameter tuning
│   │   ├── optuna_xgboost.py
│   │   └── gridsearch_baggingpu_xgboost.py
│   ├── model_training/              # Model training
│   │   ├── train_xgboost_bootstrap.py
│   │   └── train_baggingpu_xgboost_bootstrap.py
│   ├── conformal_calibration/       # Conformal prediction
│   │   ├── cross_conformal_xgboost.py
│   │   ├── cross_conformal_baggingpu.py
│   │   ├── cross_conformal_oof_xgboost.py    # Out-of-fold decision layer (paper)
│   │   └── cross_conformal_oof_baggingpu.py
│   ├── test_evaluation/             # Test set evaluation
│   │   ├── evaluate_test_fold.py
│   │   ├── evaluate_test_fold_oof.py
│   │   ├── compute_deposit_capture.py
│   │   ├── recompute_zones.py
│   │   └── recompute_zones_oof.py
│   ├── result_aggregation/          # Results aggregation
│   │   ├── aggregate_fold_results.py
│   │   ├── aggregate_oof_results.py
│   │   ├── generate_summary_statistics.py
│   │   └── compute_roc_auc.py
│   ├── visualization/               # Plotting
│   │   ├── plot_cv_metrics.py
│   │   ├── plot_zone_distribution.py
│   │   ├── plot_deposit_capture.py
│   │   ├── plot_spatial_maps.py
│   │   ├── plot_block_examples.py
│   │   └── generate_figure1.py ... generate_figure8.py  # Manuscript figures (with _figure_utils.py)
│   ├── analysis/                    # Analysis scripts
│   │   ├── shap_analysis.py
│   │   ├── variogram_analysis.py    # Spatial autocorrelation analysis
│   │   ├── r24_lift_efficiency.py   # Threshold sensitivity tables (Tables 7-8)
│   │   ├── r22_inset_calibration.py # In-set calibration (Table 4)
│   │   ├── r1324_r22_analysis.py    # Calibration and lift-configuration analysis
│   │   └── r26_triage_numbers.py    # Reproducibility-triage statistics
│   ├── preprocessing/               # Data preprocessing
│   │   └── leakage_free_preprocessor.py
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       ├── data_utils.py
│       ├── model_utils.py
│       ├── conformal_utils.py
│       ├── metrics_utils.py
│       ├── visualization_utils.py
│       └── preprocessing_utils.py
└── outputs/                         # Output directory (populated by the pipeline)
```

## Installation

### Using uv (recommended)

```bash
uv venv
uv pip install -r requirements.txt
```

### Using pip

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Configure the experiment

Edit `config/config.yaml` to set:
- Data paths
- Cross-validation parameters (number of folds, block size)
- Model parameters
- Output directories

### 2. Run the full pipeline

```bash
# Run all outer folds sequentially
uv run python scripts/orchestration/run_all_outer_folds.py

# Or run a single outer fold
uv run python scripts/orchestration/run_single_outer_fold.py --outer-fold 0

# Aggregate results after all folds complete
uv run python scripts/orchestration/run_final_aggregation.py
```

### 3. Individual steps (optional)

```bash
# Step 1: Create spatial folds
uv run python scripts/fold_generation/create_outer_folds.py
uv run python scripts/fold_generation/create_inner_folds.py --outer-fold 0

# Step 2: Hyperparameter optimization
uv run python scripts/optuna_optimization/optuna_xgboost.py --outer-fold 0

# Step 3: Train models with bootstrap
uv run python scripts/model_training/train_xgboost_bootstrap.py --outer-fold 0

# Step 4: Conformal calibration
uv run python scripts/conformal_calibration/cross_conformal_xgboost.py --outer-fold 0

# Step 5: Evaluate on test fold
uv run python scripts/test_evaluation/evaluate_test_fold.py --outer-fold 0

# Step 6: Generate visualizations
uv run python scripts/visualization/plot_spatial_maps.py
uv run python scripts/visualization/plot_deposit_capture.py
```

### Reproducing the manuscript figures and tables

- `scripts/visualization/generate_figure1.py` through `generate_figure8.py` regenerate the article figures from the pipeline outputs (`outputs/aggregated*` and per-fold zone assignments), so the pipeline must be run first. Figure 2 needs only `uv run python scripts/fold_generation/create_outer_folds.py` (seconds) plus the bundled `figures/Figure_2a.pdf` schematic.
- `generate_figure1.py` additionally requires two GSWA shapefiles placed under `data/geology_gswa/`: the 1:500,000 State Interpreted Bedrock Geology (`500k_interpgeop.*`) and `MajorCrustalBoundaries_2015.*`. They are not redistributed here because of their size; download them from the GSWA/DMIRS data portal. Its variogram panel uses `outputs/data/variogram_computed.json` when present and otherwise recomputes it from the dataset (`scripts/analysis/variogram_analysis.py`).
- `generate_figure2.py` calls the system tool `pdftoppm` (Debian/Ubuntu package `poppler-utils`).
- Figure 9 (TreeSHAP) is produced by `scripts/analysis/shap_analysis.py` and requires the trained bootstrap models, so it can be regenerated only after a full pipeline run. Figure 3 is a hand-drawn workflow diagram with no generator script.
- `scripts/analysis/r24_lift_efficiency.py`, `r22_inset_calibration.py`, `r1324_r22_analysis.py`, and `r26_triage_numbers.py` reproduce the threshold-sensitivity, calibration, and triage numbers reported in the article. They snapshot `outputs/folds`, `outputs/models`, and `outputs/conformal`, so they also require a full pipeline run first.
- `scripts/result_aggregation/compute_roc_auc.py` produces `outputs/aggregated/roc_auc_results.json` and is run manually after the final aggregation.

## Methodology

### Nested Cross-Validation

The framework implements nested spatial block cross-validation:

1. **Outer loop (K=5)**: Evaluates generalization performance
2. **Inner loop (K=5)**: Hyperparameter optimization
3. **Spatial blocking**: Prevents spatial autocorrelation leakage

### Conformal Prediction

Cross-conformal prediction provides:
- Distribution-free marginal coverage of the positive class at the nominal target (alpha = 0.15, that is an 85% target), approximate under spatial autocorrelation
- A coverage-controlled prediction-set membership indicator used in zone assignment
- Uncertainty quantification for target prioritization

### Practical Exploration Zones

Bootstrap mean probability, the relative interquartile range (rel_IQR) uncertainty measure, and conformal prediction-set membership are combined into five zones:
- **Zone 0 (Immediate)**: high probability with low uncertainty (rel_IQR <= 30%)
- **Zone 1 (Priority)**: high probability, or mid probability with low uncertainty
- **Zone 2 (Follow-up)**: mid probability, or low-mid probability with low uncertainty
- **Zone 3 (Potential)**: remaining cells inside the conformal prediction set
- **Zone 4 (Excluded)**: cells outside the coverage-controlled conformal prediction set

## Configuration

### config.yaml

```yaml
paths:
  original_data: "data/Yilgarn_GIS_DB.csv"

data:
  target_column: "Ni_mine"
  coordinate_columns: ["TMX", "TMY"]

spatial_cv:
  block_size_km: 50
  n_outer_folds: 5
  n_inner_folds: 5

bootstrap:
  n_bootstraps: 50
  random_state: 42

conformal:
  alpha: 0.15  # Target 85% coverage (1 - alpha)

zone_method: "jorc30"  # "jorc30" | "percentile_std"
```

### hyperparameter_ranges.yaml

```yaml
xgboost:
  n_estimators:
    type: "int"
    low: 100
    high: 500
  max_depth:
    type: "int"
    low: 3
    high: 8
  learning_rate:
    type: "float"
    low: 0.01
    high: 0.3
    log: true

baggingpu_gridsearch:
  max_samples:
    values: [0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
  n_estimators:
    values: [10, 20, 30, 40, 50]
```

## Output

The pipeline generates:

- `outputs/fold_*/`: Per-fold results
  - `predictions.csv`: Test predictions with uncertainty
  - `metrics.json`: Evaluation metrics
  - `models/`: Trained models
- `outputs/aggregated/`: Combined results
  - `summary_statistics.csv`: Cross-validation summary
  - `deposit_capture.csv`: Deposit capture analysis
- `outputs/figures/`: Visualization outputs

## Citation

If you use this code in your research, please cite:

```bibtex
@article{ahn2026uncertainty,
  title={Uncertainty-Aware Mineral Prospectivity Mapping with Positive-Unlabeled Learning and Conformal Calibration: Ni Mineralization in the Yilgarn Craton},
  author={Ahn, Seongin and Jo, Honggeun and Kwon, Jihoe and Park, Gyesoon and Lee, Sang-ho},
  journal={Natural Resources Research},
  year={2026},
  note={in press; DOI to be added at Online First publication}
}
```

## License

- **Code**: MIT License (see `LICENSE`; the MIT terms apply to the source code only).
- **Data**: each predictor layer follows its source custodian's terms (see `DATA-LICENSE.md`).
- **Label column (`Ni_mine`)**: derived from GSWA MINEDEX and Mineral Systems Atlas records; Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) applies to this column and its derivatives.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
