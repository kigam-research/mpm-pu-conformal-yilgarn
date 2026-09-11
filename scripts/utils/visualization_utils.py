#!/usr/bin/env python3
"""Visualization Utilities"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14
})


def get_zone_colors() -> List[str]:
    """Get standard colors for zones 0-4."""
    return ['#d62728', '#ff7f0e', '#ffbb78', '#98df8a', '#2ca02c']


def get_zone_names() -> List[str]:
    """Get standard names for zones 0-4 (JORC 30%)."""
    return [
        'IMMEDIATE',
        'PRIORITY',
        'FOLLOW_UP',
        'POTENTIAL',
        'EXCLUDED'
    ]


def save_figure(
    fig: plt.Figure,
    output_path: Union[str, Path],
    formats: List[str] = ['png', 'svg'],
    dpi: int = 300
) -> None:
    """Save figure in multiple formats."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure: {path}")


def plot_fold_comparison_bar(
    fold_values: Dict[int, float],
    metric_name: str,
    output_path: Optional[Path] = None,
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """Plot bar chart comparing metric across folds."""
    fig, ax = plt.subplots(figsize=figsize)

    folds = sorted(fold_values.keys())
    values = [fold_values[f] for f in folds]

    bars = ax.bar(folds, values, color='steelblue', alpha=0.8, edgecolor='navy')

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Threshold: {threshold:.2f}')
        ax.legend()

    ax.set_xlabel('Outer Fold')
    ax.set_ylabel(metric_name)
    ax.set_title(f'{metric_name} by Fold')
    ax.set_xticks(folds)

    mean_val = np.mean(values)
    ax.axhline(y=mean_val, color='green', linestyle=':', linewidth=1.5,
               label=f'Mean: {mean_val:.3f}')
    ax.legend()

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_zone_distribution_stacked(
    fold_zone_counts: Dict[int, Dict[int, int]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot stacked bar chart of zone distribution by fold."""
    fig, ax = plt.subplots(figsize=figsize)

    folds = sorted(fold_zone_counts.keys())
    n_zones = 5
    colors = get_zone_colors()
    zone_names = get_zone_names()

    zone_data = {z: [] for z in range(n_zones)}
    for fold in folds:
        total = sum(fold_zone_counts[fold].values())
        for z in range(n_zones):
            count = fold_zone_counts[fold].get(z, 0)
            zone_data[z].append(count / total * 100)

    bottom = np.zeros(len(folds))
    for z in range(n_zones):
        ax.bar(folds, zone_data[z], bottom=bottom, label=f'Zone {z}: {zone_names[z]}',
               color=colors[z], edgecolor='white', linewidth=0.5)
        bottom += np.array(zone_data[z])

    ax.set_xlabel('Outer Fold')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Zone Distribution by Fold')
    ax.set_xticks(folds)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_deposit_capture_boxplot(
    fold_capture_rates: Dict[int, Dict[int, float]],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """Plot box plot of deposit capture rates by zone."""
    fig, ax = plt.subplots(figsize=figsize)

    n_zones = 5
    colors = get_zone_colors()
    zone_names = get_zone_names()

    data = []
    for z in range(n_zones):
        zone_rates = [fold_capture_rates[f].get(z, 0) for f in fold_capture_rates]
        data.append(zone_rates)

    bp = ax.boxplot(data, patch_artist=True, labels=[f'Zone {z}' for z in range(n_zones)])

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for z in range(n_zones):
        x = np.random.normal(z + 1, 0.04, size=len(data[z]))
        ax.scatter(x, data[z], alpha=0.6, s=30, color='black', zorder=3)

    ax.set_ylabel('Deposit Capture Rate (%)')
    ax.set_title('Deposit Capture by Zone (across folds)')

    ax.set_xticklabels([f'Zone {z}\n{zone_names[z][:10]}' for z in range(n_zones)],
                        fontsize=8)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_spatial_map(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = 'magma',
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
    deposit_mask: Optional[np.ndarray] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """Plot spatial map with optional deposit overlay."""
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(x, y, c=values, cmap=cmap, s=1, alpha=0.8,
                         vmin=vmin, vmax=vmax)

    if deposit_mask is not None:
        ax.scatter(x[deposit_mask], y[deposit_mask], c='black', s=10,
                   marker='.', label='Deposits', zorder=5)
        ax.legend(loc='upper right')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig


def plot_zone_map(
    x: np.ndarray,
    y: np.ndarray,
    zones: np.ndarray,
    title: str = 'Practical Zone Map',
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 10),
    deposit_mask: Optional[np.ndarray] = None
) -> plt.Figure:
    """Plot zone classification map."""
    fig, ax = plt.subplots(figsize=figsize)

    colors = get_zone_colors()
    zone_names = get_zone_names()

    cmap = mcolors.ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    scatter = ax.scatter(x, y, c=zones, cmap=cmap, norm=norm, s=1, alpha=0.8)

    if deposit_mask is not None:
        ax.scatter(x[deposit_mask], y[deposit_mask], c='black', s=10,
                   marker='.', label='Deposits', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(title)
    ax.set_aspect('equal')

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black',
                            label=f'Zone {i}: {zone_names[i]}')
                      for i in range(5)]
    if deposit_mask is not None:
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], marker='.', color='w',
                                       markerfacecolor='black', markersize=10,
                                       label='Deposits'))

    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    if output_path:
        save_figure(fig, output_path)

    return fig
