#!/usr/bin/env python3
"""Run All Outer Folds Pipeline"""

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


def run_outer_fold(fold: int, method: str = None, force_restart: bool = False) -> bool:
    """Run pipeline for a single outer fold."""
    cmd = [
        "uv", "run", "python",
        str(PROJECT_ROOT / "scripts" / "orchestration" / "run_single_outer_fold.py"),
        "--outer-fold", str(fold)
    ]

    if method:
        cmd.extend(["--method", method])
    if force_restart:
        cmd.append("--force-restart")

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Fold {fold} failed with return code {e.returncode}")
        return False


def get_fold_status(status_path: Path, method: str, fold: int) -> str:
    """Get status for a specific fold."""
    if not status_path.exists():
        return 'pending'

    status = load_json(status_path)
    fold_status = status.get('fold_status', {}).get(method, {}).get(f'outer_fold_{fold}', {})
    return fold_status.get('status', 'pending')


def run_all_outer_folds(
    method: str = None,
    resume: bool = False,
    force_restart: bool = False,
    config: dict = None
):
    """Run complete pipeline for all outer folds."""
    logger.info("=" * 70)
    logger.info("Running All Outer Folds Pipeline")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    n_folds = config['spatial_cv']['n_outer_folds']
    methods = [method] if method else config['methods']
    status_path = PROJECT_ROOT / "outputs" / "fold_status.json"

    if force_restart and status_path.exists():
        status_path.unlink()
        logger.info("Cleared existing status file for fresh start")

    start_time = datetime.now()
    logger.info(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Folds: 0-{n_folds-1}")
    logger.info(f"Resume mode: {resume}")

    results = {m: {'completed': [], 'failed': [], 'skipped': []} for m in methods}

    for m in methods:
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Method: {m.upper()}")
        logger.info(f"{'='*70}")

        for fold in range(n_folds):
            fold_start = datetime.now()
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold}/{n_folds-1} - {m.upper()}")
            logger.info(f"{'='*50}")

            if resume and not force_restart:
                current_status = get_fold_status(status_path, m, fold)
                if current_status == 'completed':
                    logger.info(f"Fold {fold} already completed. Skipping.")
                    results[m]['skipped'].append(fold)
                    continue

            success = run_outer_fold(fold, m, force_restart)

            fold_duration = datetime.now() - fold_start
            logger.info(f"Fold {fold} duration: {fold_duration}")

            if success:
                results[m]['completed'].append(fold)
                logger.info(f"✓ Fold {fold} completed successfully")
            else:
                results[m]['failed'].append(fold)
                logger.error(f"✗ Fold {fold} failed")
                logger.info("Continuing to next fold...")

    total_duration = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total duration: {total_duration}")

    all_success = True
    for m in methods:
        completed = len(results[m]['completed'])
        failed = len(results[m]['failed'])
        skipped = len(results[m]['skipped'])
        total = completed + failed + skipped

        logger.info(f"\n{m.upper()}:")
        logger.info(f"  Completed: {completed}/{total}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Skipped: {skipped}")

        if results[m]['failed']:
            logger.info(f"  Failed folds: {results[m]['failed']}")
            all_success = False

    if all_success:
        logger.info("\n✓ All folds completed successfully!")
        logger.info("Run aggregation scripts next:")
        logger.info("  uv run python scripts/orchestration/run_final_aggregation.py")
    else:
        logger.warning("\n⚠ Some folds failed. Check logs and re-run with --resume")

    return all_success


def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline for all outer folds"
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=['xgboost', 'baggingpu_xgboost'],
        help="Model method (default: all methods)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (skip completed folds)"
    )
    parser.add_argument(
        "--force-restart",
        action="store_true",
        help="Force restart all folds from beginning"
    )

    args = parser.parse_args()
    run_all_outer_folds(args.method, args.resume, args.force_restart)


if __name__ == "__main__":
    main()
