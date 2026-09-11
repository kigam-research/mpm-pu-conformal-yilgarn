#!/usr/bin/env python3
"""Run Single Outer Fold Pipeline"""

import argparse
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config, load_json, save_json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, args: list) -> bool:
    """Run a Python script with arguments."""
    cmd = ["uv", "run", "python", str(script_path)] + args
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with return code {e.returncode}")
        return False


def update_fold_status(
    status_path: Path,
    method: str,
    fold: int,
    step: str,
    success: bool,
    error_msg: str = None
):
    """Update fold status tracking file."""
    if status_path.exists():
        status = load_json(status_path)
    else:
        status = {'fold_status': {}}

    if method not in status['fold_status']:
        status['fold_status'][method] = {}

    fold_key = f'outer_fold_{fold}'
    if fold_key not in status['fold_status'][method]:
        status['fold_status'][method][fold_key] = {
            'inner_folds_created': False,
            'optuna_completed': False,
            'bootstrap_completed': False,
            'conformal_completed': False,
            'evaluation_completed': False,
            'status': 'pending',
            'last_updated': None,
            'error_message': None
        }

    fold_status = status['fold_status'][method][fold_key]

    if step == 'inner_folds':
        fold_status['inner_folds_created'] = success
    elif step == 'optuna':
        fold_status['optuna_completed'] = success
    elif step == 'bootstrap':
        fold_status['bootstrap_completed'] = success
    elif step == 'conformal':
        fold_status['conformal_completed'] = success
    elif step == 'evaluation':
        fold_status['evaluation_completed'] = success

    fold_status['last_updated'] = datetime.now().isoformat()

    if not success:
        fold_status['status'] = 'failed'
        fold_status['error_message'] = error_msg
    elif all([
        fold_status['inner_folds_created'],
        fold_status['optuna_completed'],
        fold_status['bootstrap_completed'],
        fold_status['conformal_completed'],
        fold_status['evaluation_completed']
    ]):
        fold_status['status'] = 'completed'
    else:
        fold_status['status'] = 'in_progress'

    save_json(status, status_path)


def run_single_outer_fold(
    outer_fold: int,
    method: str = None,
    force_restart: bool = False,
    config: dict = None
):
    """Run complete pipeline for a single outer fold."""
    logger.info("=" * 70)
    logger.info(f"Single Outer Fold Pipeline - Fold {outer_fold}")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    methods = [method] if method else config['methods']
    status_path = PROJECT_ROOT / config['output_dirs']['folds'].replace('folds', '') / "fold_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)

    scripts_dir = PROJECT_ROOT / "scripts"

    for m in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"Method: {m.upper()}")
        logger.info(f"{'='*50}")

        if status_path.exists() and not force_restart:
            status = load_json(status_path)
            fold_status = status.get('fold_status', {}).get(m, {}).get(f'outer_fold_{outer_fold}', {})
            if fold_status.get('status') == 'completed':
                logger.info(f"Fold {outer_fold} already completed for {m}. Skipping.")
                continue

        inner_folds_path = (
            PROJECT_ROOT / config['output_dirs']['folds'] /
            "inner_folds" / f"outer_fold_{outer_fold}" / "inner_fold_assignments.csv"
        )

        if not inner_folds_path.exists() or force_restart:
            logger.info("\n[Step 1/6] Creating inner folds...")
            success = run_script(
                scripts_dir / "fold_generation" / "create_inner_folds.py",
                ["--outer-fold", str(outer_fold)]
            )
            update_fold_status(status_path, m, outer_fold, 'inner_folds', success)
            if not success:
                logger.error("Inner fold creation failed!")
                return False
        else:
            logger.info("\n[Step 1/6] Inner folds already exist. Skipping.")
            update_fold_status(status_path, m, outer_fold, 'inner_folds', True)

        if m == 'xgboost':
            optuna_script = "optuna_xgboost.py"
            optuna_output = (
                PROJECT_ROOT / config['output_dirs']['optuna'] /
                m / f"outer_fold_{outer_fold}" / "best_params.json"
            )
            step_name = "Optuna optimization"
        elif m == 'baggingpu_xgboost':
            optuna_script = "gridsearch_baggingpu_xgboost.py"
            optuna_output = (
                PROJECT_ROOT / 'outputs' / 'gridsearch' /
                m / f"outer_fold_{outer_fold}" / "best_params.json"
            )
            step_name = "Grid Search optimization"

        if not optuna_output.exists() or force_restart:
            logger.info(f"\n[Step 2/6] Running {step_name}...")
            success = run_script(
                scripts_dir / "optuna_optimization" / optuna_script,
                ["--outer-fold", str(outer_fold)]
            )
            update_fold_status(status_path, m, outer_fold, 'optuna', success)
            if not success:
                logger.error(f"{step_name} failed!")
                return False
        else:
            logger.info(f"\n[Step 2/6] {step_name} results exist. Skipping.")
            update_fold_status(status_path, m, outer_fold, 'optuna', True)

        bootstrap_script = {
            'xgboost': "train_xgboost_bootstrap.py",
            'baggingpu_xgboost': "train_baggingpu_xgboost_bootstrap.py"
        }[m]

        bootstrap_output = (
            PROJECT_ROOT / config['output_dirs']['models'] /
            m / f"outer_fold_{outer_fold}" / "bootstrap_predictions.npz"
        )

        if not bootstrap_output.exists() or force_restart:
            logger.info("\n[Step 3/6] Training bootstrap ensemble...")
            success = run_script(
                scripts_dir / "model_training" / bootstrap_script,
                ["--outer-fold", str(outer_fold)]
            )
            update_fold_status(status_path, m, outer_fold, 'bootstrap', success)
            if not success:
                logger.error("Bootstrap training failed!")
                return False
        else:
            logger.info("\n[Step 3/6] Bootstrap models exist. Skipping.")
            update_fold_status(status_path, m, outer_fold, 'bootstrap', True)

        conformal_script = {
            'xgboost': "cross_conformal_xgboost.py",
            'baggingpu_xgboost': "cross_conformal_baggingpu.py"
        }[m]

        conformal_output = (
            PROJECT_ROOT / config['output_dirs']['conformal'] /
            m / f"outer_fold_{outer_fold}" / "median_params.json"
        )

        if not conformal_output.exists() or force_restart:
            logger.info("\n[Step 4/6] Running cross-conformal calibration...")
            success = run_script(
                scripts_dir / "conformal_calibration" / conformal_script,
                ["--outer-fold", str(outer_fold)]
            )
            update_fold_status(status_path, m, outer_fold, 'conformal', success)
            if not success:
                logger.error("Cross-conformal calibration failed!")
                return False
        else:
            logger.info("\n[Step 4/6] Conformal results exist. Skipping.")
            update_fold_status(status_path, m, outer_fold, 'conformal', True)

        eval_output = (
            PROJECT_ROOT / config['output_dirs']['test_evaluation'] /
            m / f"outer_fold_{outer_fold}" / "metrics.json"
        )

        if not eval_output.exists() or force_restart:
            logger.info("\n[Step 5/6] Evaluating test fold...")
            success = run_script(
                scripts_dir / "test_evaluation" / "evaluate_test_fold.py",
                ["--outer-fold", str(outer_fold), "--method", m]
            )
            update_fold_status(status_path, m, outer_fold, 'evaluation', success)
            if not success:
                logger.error("Test evaluation failed!")
                return False
        else:
            logger.info("\n[Step 5/6] Evaluation results exist. Skipping.")
            update_fold_status(status_path, m, outer_fold, 'evaluation', True)

        logger.info("\n[Step 6/6] Computing deposit capture analysis...")
        success = run_script(
            scripts_dir / "test_evaluation" / "compute_deposit_capture.py",
            ["--outer-fold", str(outer_fold), "--method", m]
        )

        if not success:
            logger.warning("Deposit capture analysis failed (non-critical)")

        logger.info(f"\n{'='*50}")
        logger.info(f"Fold {outer_fold} complete for {m.upper()}!")
        logger.info(f"{'='*50}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete pipeline for a single outer fold"
    )
    parser.add_argument(
        "--outer-fold",
        type=int,
        required=True,
        help="Outer fold index (0-4)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method (default: all methods)"
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart from beginning"
    )

    args = parser.parse_args()
    run_single_outer_fold(args.outer_fold, args.method, args.force_restart)


if __name__ == "__main__":
    main()
