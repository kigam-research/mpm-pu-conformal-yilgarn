#!/usr/bin/env python3
"""Variogram Analysis for Target Variable (Ni_mine)"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import transform as shapely_transform
from pathlib import Path
import json
import argparse
import warnings
import logging
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore', category=RuntimeWarning)


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_PATH = PROJECT_ROOT / "data" / "Yilgarn_GIS_DB.csv"
    GADM_PATH = PROJECT_ROOT / "data" / "boundary" / "gadm41_AUS_0.shp"
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    RESULTS_PATH = OUTPUT_DIR / "data" / "variogram_computed.json"

    N_LAGS = 20
    MAXLAG_KM = 300.0
    DOWNSAMPLE_FACTOR = 2
    RANDOM_SEED = 42


def load_variogram_results() -> Optional[Dict]:
    """Load previously computed variogram results if available."""
    if Config.RESULTS_PATH.exists():
        with open(Config.RESULTS_PATH, 'r') as f:
            return json.load(f)
    return None


def save_variogram_results(results: Dict, output_path: Path = None) -> None:
    """Save variogram results to JSON."""
    if output_path is None:
        output_path = Config.RESULTS_PATH

    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {}
    for k, v in results.items():
        if k in ['variogram_object', 'lag_centers', 'semivariance', 'pair_counts']:
            continue
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            serializable[k] = float(v)
        else:
            serializable[k] = v

    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"  Saved: {output_path}")


def downsample_for_variogram(
    df: pd.DataFrame,
    coord_cols: Tuple[str, str] = ("TMX", "TMY"),
    target_col: str = "Ni_mine",
    factor: int = 2,
    preserve_positives: bool = True,
    random_state: int = 42
) -> pd.DataFrame:
    """Downsample the dataset for variogram analysis."""
    np.random.seed(random_state)
    x_col, y_col = coord_cols

    unique_x = np.sort(df[x_col].unique())
    unique_y = np.sort(df[y_col].unique())
    print(f"  Original grid: {len(unique_x)} x {len(unique_y)} = {len(df):,} cells")

    selected_x = unique_x[::factor]
    selected_y = unique_y[::factor]
    print(f"  Downsampled grid: {len(selected_x)} x {len(selected_y)} = {len(selected_x) * len(selected_y):,} cells (theoretical)")

    mask_grid = df[x_col].isin(selected_x) & df[y_col].isin(selected_y)

    if preserve_positives:
        mask_positive = df[target_col] == 1
        mask = mask_grid | mask_positive
        n_positive_total = mask_positive.sum()
        n_positive_on_grid = (mask_grid & mask_positive).sum()
        n_positive_added = n_positive_total - n_positive_on_grid
        print(f"  Positive samples: {n_positive_total} total, {n_positive_on_grid} on grid, {n_positive_added} added")
    else:
        mask = mask_grid

    df_downsampled = df[mask].copy().reset_index(drop=True)
    print(f"  Final downsampled size: {len(df_downsampled):,} ({len(df_downsampled)/len(df)*100:.1f}% of original)")

    return df_downsampled


def compute_variogram(
    data_path: Path = None,
    target_col: str = "Ni_mine",
    coord_cols: Tuple[str, str] = ("TMX", "TMY"),
    n_lags: int = None,
    maxlag_km: float = None,
    downsample_factor: int = None,
    models: List[str] = ['spherical', 'exponential', 'gaussian']
) -> Dict:
    """Compute variogram using skgstat library."""
    from skgstat import Variogram

    if data_path is None:
        data_path = Config.DATA_PATH
    if n_lags is None:
        n_lags = Config.N_LAGS
    if maxlag_km is None:
        maxlag_km = Config.MAXLAG_KM
    if downsample_factor is None:
        downsample_factor = Config.DOWNSAMPLE_FACTOR

    print("=" * 70)
    print("VARIOGRAM COMPUTATION")
    print("=" * 70)

    print(f"\nLoading data from {data_path.name}")
    data = pd.read_csv(data_path)
    x_col, y_col = coord_cols
    n_total = len(data)
    n_positive = (data[target_col] == 1).sum()
    print(f"Total samples: {n_total:,}")
    print(f"Positive samples ({target_col}=1): {n_positive:,}")

    print(f"\nApplying grid-based downsampling (factor={downsample_factor})...")
    df_downsampled = downsample_for_variogram(
        data,
        coord_cols=coord_cols,
        target_col=target_col,
        factor=downsample_factor,
        preserve_positives=True,
        random_state=Config.RANDOM_SEED
    )

    sample_size = len(df_downsampled)
    coords = df_downsampled[[x_col, y_col]].values
    values = df_downsampled[target_col].values
    maxlag = maxlag_km * 1000

    print(f"\nComputing variogram...")
    print(f"  n_lags: {n_lags}")
    print(f"  maxlag: {maxlag_km} km")

    best_result = None
    best_rmse = np.inf

    for model in models:
        print(f"  Trying model: {model}...", end=" ")
        try:
            V = Variogram(
                coordinates=coords,
                values=values,
                n_lags=n_lags,
                maxlag=maxlag,
                model=model,
                estimator='matheron',
                fit_method='trf'
            )

            params = V.parameters
            rmse = V.rmse
            range_m = params[0] if len(params) > 0 else np.nan
            sill = params[1] if len(params) > 1 else np.nan
            nugget = params[2] if len(params) > 2 else np.nan
            range_km = range_m / 1000

            print(f"RMSE: {rmse:.6f}, Range: {range_km:.1f} km")

            if rmse < best_rmse:
                best_rmse = rmse
                best_result = {
                    'model': model,
                    'effective_range_km': float(range_km),
                    'range_m': float(range_m),
                    'sill': float(sill),
                    'nugget': float(nugget),
                    'rmse': float(rmse),
                    'n_samples': int(len(values)),
                    'bins_km': (np.array(V.bins) / 1000).tolist(),
                    'experimental': V.experimental.tolist(),
                }

        except Exception as e:
            print(f"Error: {e}")
            continue

    if best_result is None:
        raise RuntimeError("All variogram computations failed!")

    best_result['computation_metadata'] = {
        'original_n_samples': int(n_total),
        'used_n_samples': int(sample_size),
        'downsample_factor': int(downsample_factor),
        'n_lags': int(n_lags),
        'maxlag_km': float(maxlag_km),
    }

    print("\n" + "=" * 70)
    print("VARIOGRAM RESULTS")
    print("=" * 70)
    print(f"  Best model: {best_result['model']}")
    print(f"  Effective Range: {best_result['effective_range_km']:.1f} km")
    print(f"  Sill: {best_result['sill']:.4f}")
    print(f"  Nugget: {best_result['nugget']:.4f}")
    if best_result['sill'] > 0:
        print(f"  Nugget/Sill Ratio: {best_result['nugget']/best_result['sill']:.4f}")
    print("=" * 70)

    return best_result


def load_data_for_visualization():
    """Load and prepare data for visualization."""
    print("Loading data...")
    df = pd.read_csv(Config.DATA_PATH)
    print(f"  Total samples: {len(df):,}")
    df['X_km'] = df['TMX'] / 1000
    df['Y_km'] = df['TMY'] / 1000
    return df


def load_boundary_km():
    """Load GADM Australia boundary and clip to Yilgarn extent."""
    print("Loading boundary...")
    gdf = gpd.read_file(Config.GADM_PATH)

    data_extent = box(12800000, -4100000, 14150000, -2800000)
    geom = gdf.geometry.iloc[0]
    clipped = geom.intersection(data_extent)

    if clipped.geom_type == 'MultiPolygon':
        largest = max(clipped.geoms, key=lambda p: p.area)
    else:
        largest = clipped

    def convert_to_km(x, y, z=None):
        return (x / 1000, y / 1000)

    return shapely_transform(convert_to_km, largest)


def create_spatial_map_figure(df, boundary_geom_km, output_path):
    """Create spatial distribution map figure."""
    print("\nCreating spatial distribution map...")

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    ax.scatter(df['X_km'], df['Y_km'],
               c='#f0f0f0', s=0.3, alpha=0.5, rasterized=True)

    deposits = df[df['Ni_mine'] == 1]
    ax.scatter(deposits['X_km'], deposits['Y_km'],
               c='red', s=10, alpha=0.9, label=f'Ni Deposits (n={len(deposits)})',
               edgecolors='darkred', linewidths=0.3, zorder=3)

    if boundary_geom_km.geom_type == 'Polygon':
        x_coords, y_coords = boundary_geom_km.exterior.xy
        ax.plot(x_coords, y_coords, color='black', linewidth=1.5,
                label='Yilgarn Craton', zorder=2)

    ax.set_xlabel('X (km)', fontsize=10)
    ax.set_ylabel('Y (km)', fontsize=10)
    ax.set_title('Spatial Distribution of Ni Deposits', fontsize=11, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def create_variogram_plot_figure(results: Dict, output_path: Path):
    """Create variogram analysis figure using computed results."""
    print("\nCreating variogram analysis plot...")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    bins_km = np.array(results.get('bins_km', []))
    experimental = np.array(results.get('experimental', []))
    effective_range = results['effective_range_km']
    sill = results['sill']
    nugget = results['nugget']
    model_name = results['model']

    if len(bins_km) > 0 and len(experimental) > 0:
        ax.scatter(bins_km, experimental, c='steelblue', s=50,
                   label='Empirical', zorder=3, edgecolors='navy', linewidths=0.5)

    h_fit = np.linspace(0, max(bins_km) if len(bins_km) > 0 else 150, 200)

    if model_name == 'gaussian':
        a = effective_range / np.sqrt(3)
        gamma_fit = nugget + (sill - nugget) * (1 - np.exp(-3 * (h_fit / a) ** 2))
    elif model_name == 'spherical':
        a = effective_range
        gamma_fit = np.where(
            h_fit <= a,
            nugget + (sill - nugget) * (1.5 * (h_fit / a) - 0.5 * (h_fit / a) ** 3),
            sill
        )
    else:
        a = effective_range / 3
        gamma_fit = nugget + (sill - nugget) * (1 - np.exp(-h_fit / a))

    ax.plot(h_fit, gamma_fit, 'r-', linewidth=2, label=f'{model_name.capitalize()} model')

    ax.axhline(y=sill, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axhline(y=nugget, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    ax.axvline(x=effective_range, color='orange', linestyle='--', linewidth=2,
               label=f'Effective Range ({effective_range:.1f} km)')

    ax.set_xlabel('Distance (km)', fontsize=10)
    ax.set_ylabel('Semivariance', fontsize=10)
    ax.set_title('Target Variogram Analysis', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_xlim(0, max(bins_km) * 1.1 if len(bins_km) > 0 else 150)
    ax.set_ylim(0, sill * 1.15 if sill > 0 else 0.2)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(output_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close(fig)


def main():
    """Main execution: compute variogram if needed, then generate visualizations."""
    print("=" * 70)
    print("Variogram Analysis for Target Variable (Ni_mine)")
    print("=" * 70)

    figures_dir = Config.OUTPUT_DIR / "figures"
    data_dir = Config.OUTPUT_DIR / "data"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    results = load_variogram_results()

    if results is None:
        print("\nNo existing results found. Computing variogram...")
        results = compute_variogram()
        save_variogram_results(results)
    else:
        print(f"\nLoaded existing results from {Config.RESULTS_PATH.name}")
        print(f"  Model: {results['model']}")
        print(f"  Effective Range: {results['effective_range_km']:.1f} km")

    print("\n" + "-" * 70)
    print("Generating Visualizations")
    print("-" * 70)

    df = load_data_for_visualization()
    boundary_geom_km = load_boundary_km()

    create_spatial_map_figure(
        df, boundary_geom_km,
        figures_dir / "spatial_distribution.png"
    )

    create_variogram_plot_figure(
        results,
        figures_dir / "variogram_analysis.png"
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Model: {results['model']}")
    print(f"  Effective Range: {results['effective_range_km']:.1f} km")
    print(f"  Sill: {results['sill']:.4f}")
    print(f"  Nugget: {results['nugget']:.4f}")
    print(f"\nOutputs saved to: {Config.OUTPUT_DIR}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Variogram Analysis for Ni_mine Target Variable"
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recompute even if results exist"
    )
    parser.add_argument(
        "--downsample-factor",
        type=int,
        default=Config.DOWNSAMPLE_FACTOR,
        help=f"Downsampling factor (default: {Config.DOWNSAMPLE_FACTOR})"
    )
    parser.add_argument(
        "--n-lags",
        type=int,
        default=Config.N_LAGS,
        help=f"Number of lag bins (default: {Config.N_LAGS})"
    )
    parser.add_argument(
        "--maxlag-km",
        type=float,
        default=Config.MAXLAG_KM,
        help=f"Maximum lag distance in km (default: {Config.MAXLAG_KM})"
    )
    args = parser.parse_args()

    if args.recompute and Config.RESULTS_PATH.exists():
        Config.RESULTS_PATH.unlink()
        print("Deleted existing results for recomputation.")

    Config.N_LAGS = args.n_lags
    Config.MAXLAG_KM = args.maxlag_km
    Config.DOWNSAMPLE_FACTOR = args.downsample_factor

    main()
