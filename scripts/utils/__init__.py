"""Utils Package"""

from .data_utils import (
    get_project_root,
    load_config,
    load_hyperparameter_ranges,
    load_original_data,
    load_outer_folds,
    load_inner_folds,
    get_train_test_data,
    get_feature_matrix,
    save_json,
    load_json,
    save_numpy,
    load_numpy
)

from .model_utils import (
    BaggingPUClassifier,
    train_xgboost,
    train_baggingpu_xgboost,
    train_bootstrap_ensemble,
    predict_bootstrap_ensemble,
    load_bootstrap_models,
    save_model,
    load_model
)

from .conformal_utils import (
    PlattCalibrator,
    PlattApplicator,
    fnr_control_direct,
    compute_prediction_sets,
    compute_fnr_practical_zones,
    assign_zones_jorc30,
    assign_zones_lift,
    compute_coverage_metrics,
    compute_deposit_capture_by_zone
)

from .metrics_utils import (
    compute_pr_auc,
    compute_roc_auc,
    compute_classification_metrics,
    compute_recall_at_fpr,
    compute_deposit_capture_curve,
    compute_top_k_capture,
    aggregate_fold_metrics
)

from .visualization_utils import (
    get_zone_colors,
    get_zone_names,
    save_figure,
    plot_fold_comparison_bar,
    plot_zone_distribution_stacked,
    plot_deposit_capture_boxplot,
    plot_spatial_map,
    plot_zone_map
)

from .preprocessing_utils import (
    LeakageFreePreprocessor
)

__all__ = [
    'get_project_root',
    'load_config',
    'load_hyperparameter_ranges',
    'load_original_data',
    'load_outer_folds',
    'load_inner_folds',
    'get_train_test_data',
    'get_feature_matrix',
    'save_json',
    'load_json',
    'save_numpy',
    'load_numpy',
    'BaggingPUClassifier',
    'train_xgboost',
    'train_baggingpu_xgboost',
    'train_bootstrap_ensemble',
    'predict_bootstrap_ensemble',
    'load_bootstrap_models',
    'save_model',
    'load_model',
    'PlattCalibrator',
    'PlattApplicator',
    'fnr_control_direct',
    'compute_prediction_sets',
    'compute_fnr_practical_zones',
    'assign_zones_jorc30',
    'assign_zones_lift',
    'compute_coverage_metrics',
    'compute_deposit_capture_by_zone',
    'compute_pr_auc',
    'compute_roc_auc',
    'compute_classification_metrics',
    'compute_recall_at_fpr',
    'compute_deposit_capture_curve',
    'compute_top_k_capture',
    'aggregate_fold_metrics',
    'get_zone_colors',
    'get_zone_names',
    'save_figure',
    'plot_fold_comparison_bar',
    'plot_zone_distribution_stacked',
    'plot_deposit_capture_boxplot',
    'plot_spatial_map',
    'plot_zone_map',
    'LeakageFreePreprocessor'
]
