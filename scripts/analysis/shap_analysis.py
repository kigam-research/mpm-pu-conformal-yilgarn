"""TreeSHAP Feature Importance Analysis for MPM Paper Appendix D"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

DATA_PATH = PROJECT_ROOT / "data" / "Yilgarn_GIS_DB.csv"
FOLDS_PATH = PROJECT_ROOT / "outputs" / "folds" / "outer_fold_assignments.csv"
MODELS_PATH = PROJECT_ROOT / "outputs" / "models"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "shap_analysis"

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


def load_data_and_preprocess(fold_id: int = 0):
    """Load data and apply preprocessing for a specific fold."""

    df = pd.read_csv(DATA_PATH)
    folds = pd.read_csv(FOLDS_PATH)

    preproc_path = MODELS_PATH / "xgboost" / f"outer_fold_{fold_id}" / "preprocessor_params.json"
    with open(preproc_path, 'r') as f:
        preproc = json.load(f)

    feature_names = preproc['feature_names']
    final_feature_names = preproc['final_feature_names']

    test_indices = folds[folds['fold_id'] == fold_id]['index'].values
    train_indices = folds[folds['fold_id'] != fold_id]['index'].values

    X_train = df.loc[train_indices, feature_names].copy()
    X_test = df.loc[test_indices, feature_names].copy()
    y_train = folds[folds['fold_id'] != fold_id]['target'].values
    y_test = folds[folds['fold_id'] == fold_id]['target'].values

    def preprocess(X, params, fit=False):
        X = X.copy()

        for col in params['missing_indicator_cols']:
            X[f'{col}_missing'] = X[col].isna().astype(int)

        for col, val in params['impute_values'].items():
            X[col] = X[col].fillna(val)

        for col, (low, high) in params['winsorize_bounds'].items():
            X[col] = X[col].clip(low, high)

        for col in params['proximity_features']:
            X[col] = np.log1p(X[col])

        for i, col in enumerate(params['final_feature_names']):
            if col in X.columns:
                idx = params['final_feature_names'].index(col)
                center = params['scaler_center'][idx]
                scale = params['scaler_scale'][idx]
                if scale > 0:
                    X[col] = (X[col] - center) / scale

        return X[params['final_feature_names']]

    X_train_processed = preprocess(X_train, preproc)
    X_test_processed = preprocess(X_test, preproc)

    return X_train_processed, X_test_processed, y_train, y_test, final_feature_names


def load_xgboost_model(fold_id: int = 0, bootstrap_id: int = 0):
    """Load XGBoost bootstrap model."""
    model_path = MODELS_PATH / "xgboost" / f"outer_fold_{fold_id}" / "bootstrap_models" / f"bootstrap_{bootstrap_id:03d}.joblib"
    return joblib.load(model_path)


def load_baggingpu_model(fold_id: int = 0, bootstrap_id: int = 0):
    """Load BaggingPU-XGBoost bootstrap model."""
    model_path = MODELS_PATH / "baggingpu_xgboost" / f"outer_fold_{fold_id}" / "bootstrap_models" / f"bootstrap_{bootstrap_id:03d}.joblib"
    return joblib.load(model_path)


def extract_xgboost_from_baggingpu(baggingpu_model):
    """Extract XGBoost base estimators from BaggingPU model."""
    if hasattr(baggingpu_model, 'estimators_'):
        return baggingpu_model.estimators_
    else:
        raise ValueError("Cannot extract estimators from BaggingPU model")


def compute_shap_values(model, X, model_type='xgboost'):
    """Compute SHAP values using TreeExplainer."""

    if model_type == 'xgboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
    elif model_type == 'baggingpu':
        base_estimators = extract_xgboost_from_baggingpu(model)
        shap_values_list = []

        for est in base_estimators:
            explainer = shap.TreeExplainer(est)
            sv = explainer.shap_values(X)
            shap_values_list.append(sv)

        shap_values = np.mean(shap_values_list, axis=0)

    return shap_values


def compute_global_importance(shap_values, feature_names):
    """Compute global feature importance (mean |SHAP|)."""
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False)
    return importance_df


def run_shap_analysis(fold_id: int = 0, n_samples: int = None, n_bootstraps: int = 5):
    """Run SHAP analysis for XGBoost and BaggingPU models."""

    print(f"Loading data for fold {fold_id}...")
    X_train, X_test, y_train, y_test, feature_names = load_data_and_preprocess(fold_id)

    if n_samples is not None and len(X_test) > n_samples:
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test.iloc[sample_idx]
    else:
        X_sample = X_test

    print(f"Using {len(X_sample)} samples for SHAP analysis (total test: {len(X_test)})")

    xgb_importance_list = []
    bag_importance_list = []

    for b in range(n_bootstraps):
        print(f"\nProcessing bootstrap {b}...")

        print("  Computing XGBoost SHAP values...")
        xgb_model = load_xgboost_model(fold_id, b)
        xgb_shap = compute_shap_values(xgb_model, X_sample.values, 'xgboost')
        xgb_imp = compute_global_importance(xgb_shap, feature_names)
        xgb_imp['bootstrap'] = b
        xgb_importance_list.append(xgb_imp)

        print("  Computing BaggingPU SHAP values...")
        bag_model = load_baggingpu_model(fold_id, b)
        bag_shap = compute_shap_values(bag_model, X_sample.values, 'baggingpu')
        bag_imp = compute_global_importance(bag_shap, feature_names)
        bag_imp['bootstrap'] = b
        bag_importance_list.append(bag_imp)

    xgb_all = pd.concat(xgb_importance_list)
    bag_all = pd.concat(bag_importance_list)

    xgb_mean = xgb_all.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    xgb_mean.columns = ['feature', 'xgb_mean', 'xgb_std']
    xgb_mean = xgb_mean.sort_values('xgb_mean', ascending=False)

    bag_mean = bag_all.groupby('feature')['importance'].agg(['mean', 'std']).reset_index()
    bag_mean.columns = ['feature', 'bag_mean', 'bag_std']

    results = xgb_mean.merge(bag_mean, on='feature')
    results['rank_xgb'] = range(1, len(results) + 1)
    results = results.sort_values('bag_mean', ascending=False)
    results['rank_bag'] = range(1, len(results) + 1)
    results = results.sort_values('xgb_mean', ascending=False)

    results.to_csv(OUTPUT_PATH / f"feature_importance_fold{fold_id}.csv", index=False)
    print(f"\nResults saved to {OUTPUT_PATH / f'feature_importance_fold{fold_id}.csv'}")

    return results, X_sample, xgb_shap, bag_shap, feature_names


def create_comparison_plot(results, output_path):
    """Create feature importance comparison plot."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))

    results_sorted = results.sort_values('xgb_mean', ascending=True)

    ax1 = axes[0]
    y_pos = np.arange(len(results_sorted))
    ax1.barh(y_pos, results_sorted['xgb_mean'], xerr=results_sorted['xgb_std'],
             color='steelblue', alpha=0.8, capsize=3)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(results_sorted['feature'], fontsize=9)
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('XGBoost Feature Importance')
    ax1.grid(axis='x', alpha=0.3)

    results_sorted_bag = results.sort_values('bag_mean', ascending=True)
    ax2 = axes[1]
    ax2.barh(y_pos, results_sorted_bag['bag_mean'], xerr=results_sorted_bag['bag_std'],
             color='darkorange', alpha=0.8, capsize=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(results_sorted_bag['feature'], fontsize=9)
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('BaggingPU-XGBoost Feature Importance')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / 'feature_importance_comparison.pdf', bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path / 'feature_importance_comparison.png'}")


def create_rank_comparison_plot(results, output_path):
    """Create rank difference scatter plot."""

    fig, ax = plt.subplots(figsize=(10, 8))

    results['rank_diff'] = results['rank_xgb'] - results['rank_bag']

    colors = ['green' if d > 0 else 'red' if d < 0 else 'gray' for d in results['rank_diff']]

    ax.scatter(results['rank_xgb'], results['rank_bag'], c=colors, s=100, alpha=0.7)

    ax.plot([0, 26], [0, 26], 'k--', alpha=0.5, label='Same rank')

    for _, row in results.iterrows():
        if abs(row['rank_diff']) >= 3:
            ax.annotate(row['feature'], (row['rank_xgb'], row['rank_bag']),
                       fontsize=8, alpha=0.8)

    ax.set_xlabel('XGBoost Rank')
    ax.set_ylabel('BaggingPU Rank')
    ax.set_title('Feature Importance Rank Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 26)
    ax.set_ylim(0, 26)

    plt.tight_layout()
    plt.savefig(output_path / 'rank_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Rank comparison saved to {output_path / 'rank_comparison.png'}")


def analyze_all_folds(n_samples=None, n_bootstraps=50):
    """Run analysis across all 5 folds and aggregate."""

    all_results = []
    total_models = 5 * n_bootstraps

    print(f"\n{'='*60}")
    print(f"FULL SHAP ANALYSIS: {total_models} models (5 folds × {n_bootstraps} bootstraps)")
    print(f"{'='*60}")

    for fold_id in range(5):
        print(f"\n{'='*60}")
        print(f"Analyzing Fold {fold_id+1}/5")
        print('='*60)

        results, _, _, _, _ = run_shap_analysis(fold_id, n_samples, n_bootstraps)
        results['fold'] = fold_id
        all_results.append(results)

        results.to_csv(OUTPUT_PATH / f'feature_importance_fold{fold_id}.csv', index=False)
        print(f"Fold {fold_id} results saved.")

    combined = pd.concat(all_results, ignore_index=True)

    overall = combined.groupby('feature').agg({
        'xgb_mean': ['mean', 'std'],
        'bag_mean': ['mean', 'std']
    }).reset_index()
    overall.columns = ['feature', 'xgb_overall_mean', 'xgb_overall_std',
                       'bag_overall_mean', 'bag_overall_std']
    overall = overall.sort_values('xgb_overall_mean', ascending=False)
    overall['rank_xgb'] = range(1, len(overall) + 1)
    overall = overall.sort_values('bag_overall_mean', ascending=False)
    overall['rank_bag'] = range(1, len(overall) + 1)
    overall = overall.sort_values('xgb_overall_mean', ascending=False)

    overall.to_csv(OUTPUT_PATH / 'feature_importance_all_folds.csv', index=False)
    print(f"\nOverall results saved to {OUTPUT_PATH / 'feature_importance_all_folds.csv'}")

    return overall, combined


def create_figure_c1(results, output_path):
    """Create Figure C1: All features SHAP comparison in single chart."""
    import matplotlib
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

    results_sorted = results.sort_values('xgb_overall_mean', ascending=True)

    results_sorted = results_sorted[~results_sorted['feature'].str.endswith('_missing')]

    n_features = len(results_sorted)
    y_pos = np.arange(n_features)
    bar_height = 0.35

    fig, ax = plt.subplots(figsize=(10, 10))

    bars1 = ax.barh(y_pos + bar_height/2, results_sorted['xgb_overall_mean'],
                    height=bar_height, color='#2166AC', alpha=0.85,
                    xerr=results_sorted['xgb_overall_std'],
                    capsize=2, ecolor='gray', error_kw={'linewidth': 0.8},
                    label='XGBoost')

    bars2 = ax.barh(y_pos - bar_height/2, results_sorted['bag_overall_mean'],
                    height=bar_height, color='#D6604D', alpha=0.85,
                    xerr=results_sorted['bag_overall_std'],
                    capsize=2, ecolor='gray', error_kw={'linewidth': 0.8},
                    label='BaggingPU-XGBoost')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_sorted['feature'], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
    ax.set_ylabel('')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_xlim(left=0)

    plt.tight_layout()

    output_file = output_path / 'shap_comparison_all_features.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path / 'shap_comparison_all_features.pdf', bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Figure C1 saved to: {output_file}")
    return output_file


def regenerate_figure_c1():
    """Regenerate Figure C1 using existing computed results."""
    csv_path = OUTPUT_PATH / 'feature_importance_all_folds.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Computed results not found: {csv_path}")

    results = pd.read_csv(csv_path)
    create_figure_c1(results, OUTPUT_PATH)
    print("Figure C1 regenerated successfully.")


def main():
    """Main entry point."""

    print("="*60)
    print("TreeSHAP Feature Importance Analysis")
    print("XGBoost vs BaggingPU-XGBoost Comparison")
    print("FULL ANALYSIS: 5 Folds x 50 Bootstraps = 250 Models")
    print("2000 samples per fold (standard SHAP practice)")
    print("="*60)

    print("\n[Phase 1] Full SHAP analysis across all folds and bootstraps")
    overall_results, combined_results = analyze_all_folds(
        n_samples=2000,
        n_bootstraps=50
    )

    print("\n[Phase 2] Creating Figure C1...")
    create_figure_c1(overall_results, OUTPUT_PATH)

    print("\n[Phase 3] Creating additional visualizations...")
    create_comparison_plot(overall_results.rename(columns={
        'xgb_overall_mean': 'xgb_mean', 'xgb_overall_std': 'xgb_std',
        'bag_overall_mean': 'bag_mean', 'bag_overall_std': 'bag_std'
    }), OUTPUT_PATH)
    create_rank_comparison_plot(overall_results.rename(columns={
        'xgb_overall_mean': 'xgb_mean', 'xgb_overall_std': 'xgb_std',
        'bag_overall_mean': 'bag_mean', 'bag_overall_std': 'bag_std'
    }), OUTPUT_PATH)

    print("\n" + "="*60)
    print("SUMMARY: Top 5 Features (250 models average)")
    print("="*60)

    print("\nXGBoost (mean |SHAP| across 250 models):")
    for i, row in overall_results.head(5).iterrows():
        print(f"  {row['rank_xgb']:2d}. {row['feature']:<35} {row['xgb_overall_mean']:.6f} (+/- {row['xgb_overall_std']:.6f})")

    results_bag = overall_results.sort_values('bag_overall_mean', ascending=False)
    print("\nBaggingPU-XGBoost (mean |SHAP| across 250 models):")
    for i, row in results_bag.head(5).iterrows():
        print(f"  {row['rank_bag']:2d}. {row['feature']:<35} {row['bag_overall_mean']:.6f} (+/- {row['bag_overall_std']:.6f})")

    print("\n" + "="*60)
    print("RANK CHANGES (XGBoost -> BaggingPU)")
    print("="*60)

    overall_results['rank_change'] = overall_results['rank_xgb'] - overall_results['rank_bag']
    significant_changes = overall_results[abs(overall_results['rank_change']) >= 3].sort_values('rank_change', ascending=False)

    print("\nFeatures with rank change >= 3:")
    for _, row in significant_changes.iterrows():
        direction = "UP" if row['rank_change'] > 0 else "DOWN"
        print(f"  {row['feature']:<35} {direction} {abs(int(row['rank_change']))} "
              f"(XGB: {int(row['rank_xgb'])} -> Bag: {int(row['rank_bag'])})")

    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total models analyzed: 250 (5 folds x 50 bootstraps)")
    print(f"Total test samples used: 10,000 (5 folds x 2,000 each)")
    print(f"BaggingPU internal estimators: 20 per model (5,000 total)")
    print(f"\nAll outputs saved to: {OUTPUT_PATH}")

    return overall_results


if __name__ == "__main__":
    results = main()
