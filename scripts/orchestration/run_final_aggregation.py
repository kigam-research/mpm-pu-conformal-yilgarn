#!/usr/bin/env python3
"""Run Final Aggregation and Visualization"""

import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, args: list = None) -> bool:
    """Run a Python script with optional arguments."""
    cmd = ["uv", "run", "python", str(script_path)]
    if args:
        cmd.extend(args)
    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed with return code {e.returncode}")
        return False


def run_final_aggregation(config: dict = None):
    """Run final aggregation and visualization."""
    logger.info("=" * 70)
    logger.info("Final Aggregation and Visualization")
    logger.info("=" * 70)

    if config is None:
        config = load_config()

    scripts_dir = PROJECT_ROOT / "scripts"
    start_time = datetime.now()

    methods = config.get('methods', ['xgboost', 'baggingpu_xgboost'])

    logger.info("\n[Step 0] Recomputing Practical Zones (Prob & Lift)")
    logger.info("-" * 50)

    recompute_results = []
    for method in methods:
        logger.info(f"\nRecomputing zones for {method}...")
        success = run_script(
            scripts_dir / "test_evaluation" / "recompute_zones.py",
            args=["--method", method, "--zone-mode", "all"]
        )
        recompute_results.append((f"Recompute Zones ({method})", success))
        if success:
            logger.info(f"✓ Zone recomputation completed for {method}")
        else:
            logger.warning(f"✗ Zone recomputation failed for {method}")

    steps = [
        ("Aggregate Fold Results", "result_aggregation/aggregate_fold_results.py"),
        ("Generate Summary Statistics", "result_aggregation/generate_summary_statistics.py"),
        ("Plot CV Metrics", "visualization/plot_cv_metrics.py"),
        ("Plot Zone Distribution", "visualization/plot_zone_distribution.py"),
        ("Plot Deposit Capture", "visualization/plot_deposit_capture.py"),
        ("Plot Spatial Maps", "visualization/plot_spatial_maps.py"),
    ]

    results = recompute_results.copy()

    for i, (name, script) in enumerate(steps, 1):
        logger.info(f"\n[Step {i}/{len(steps)}] {name}")
        logger.info("-" * 50)

        success = run_script(scripts_dir / script)
        results.append((name, success))

        if success:
            logger.info(f"✓ {name} completed")
        else:
            logger.warning(f"✗ {name} failed (continuing...)")

    logger.info("\n[Step 7] Plotting Block Examples (all methods)")
    logger.info("-" * 50)

    for method in methods:
        logger.info(f"\nPlotting block examples for {method}...")
        success = run_script(
            scripts_dir / "visualization" / "plot_block_examples.py",
            args=["--method", method]
        )
        results.append((f"Plot Block Examples ({method})", success))
        if success:
            logger.info(f"✓ Block examples completed for {method}")
        else:
            logger.warning(f"✗ Block examples failed for {method}")

    total_duration = datetime.now() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total duration: {total_duration}")

    successful = sum(1 for _, s in results if s)
    logger.info(f"Steps completed: {successful}/{len(results)}")

    for name, success in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    logger.info("\n" + "-" * 50)
    logger.info("Output Locations:")
    logger.info("-" * 50)

    outputs = [
        ("Aggregated results", "outputs/aggregated/"),
        ("Practical Zone (Prob)", "outputs/aggregated/*/practical_zone/prob/"),
        ("Practical Zone (Lift)", "outputs/aggregated/*/practical_zone/lift/"),
        ("Figures", "outputs/figures/"),
        ("Zone Analysis Figures", "outputs/figures/zone_analysis/"),
        ("Reports", "reports/"),
    ]

    for name, path in outputs:
        if '*' in path:
            import glob
            full_paths = list(PROJECT_ROOT.glob(path.lstrip('/')))
            if full_paths:
                n_files = sum(1 for p in full_paths for _ in p.rglob("*") if _.is_file())
                logger.info(f"  {name}: {len(full_paths)} dirs, {n_files} files")
            else:
                logger.info(f"  {name}: not created")
        else:
            full_path = PROJECT_ROOT / path
            if full_path.exists():
                n_files = sum(1 for _ in full_path.rglob("*") if _.is_file())
                logger.info(f"  {name}: {full_path} ({n_files} files)")
            else:
                logger.info(f"  {name}: {full_path} (not created)")

    report_path = PROJECT_ROOT / "reports" / "nested_cv_validation_report.md"
    if report_path.exists():
        logger.info(f"\n✓ Validation report ready: {report_path}")
    else:
        logger.warning("\n⚠ Validation report not generated")

    return all(s for _, s in results)


def main():
    run_final_aggregation()


if __name__ == "__main__":
    main()
