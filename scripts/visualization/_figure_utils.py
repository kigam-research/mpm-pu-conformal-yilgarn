"""Shared journal-figure utilities for the NNAR paper figures."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nnar")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform


def setup_journal_matplotlib() -> None:
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["font.family"] = "DejaVu Sans"


setup_journal_matplotlib()


def geometry_to_km(geometry):
    return shapely_transform(lambda x, y, z=None: (x / 1000.0, y / 1000.0), geometry)


def load_boundary_km(boundary_path: Path, df_meters: pd.DataFrame) -> object:
    gdf = gpd.read_file(boundary_path)
    geom = gdf.geometry.union_all()
    extent = box(
        df_meters["TMX"].min() if "TMX" in df_meters else df_meters["X"].min(),
        df_meters["TMY"].min() if "TMY" in df_meters else df_meters["Y"].min(),
        df_meters["TMX"].max() if "TMX" in df_meters else df_meters["X"].max(),
        df_meters["TMY"].max() if "TMY" in df_meters else df_meters["Y"].max(),
    )
    clipped = geom.intersection(extent)
    return geometry_to_km(clipped)


def iter_polygons(geometry) -> Iterable[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif isinstance(geometry, MultiPolygon):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from iter_polygons(part)


def iter_lines(geometry) -> Iterable[LineString]:
    if geometry.is_empty:
        return
    if isinstance(geometry, LineString):
        yield geometry
    elif isinstance(geometry, MultiLineString):
        yield from geometry.geoms
    elif isinstance(geometry, GeometryCollection):
        for part in geometry.geoms:
            yield from iter_lines(part)


def plot_boundary(
    ax: Axes,
    geometry,
    *,
    facecolor: str | None = None,
    edgecolor: str = "black",
    linewidth: float = 0.9,
    alpha: float = 1.0,
    zorder: int = 3,
) -> None:
    for polygon in iter_polygons(geometry):
        x, y = polygon.exterior.xy
        if facecolor:
            ax.fill(x, y, facecolor=facecolor, edgecolor="none", alpha=alpha, zorder=zorder - 1)
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, zorder=zorder)
        for interior in polygon.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, color=edgecolor, linewidth=linewidth * 0.5, zorder=zorder)
    for line in iter_lines(geometry):
        x, y = line.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)


def add_north_arrow(ax: Axes, x: float = 0.92, y: float = 0.88, size: float = 0.12) -> None:
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - size),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="black", linewidth=1.2, shrinkA=0, shrinkB=0),
        zorder=10,
    )


def add_scale_bar(
    ax: Axes,
    length_km: float = 200.0,
    location: tuple[float, float] = (0.08, 0.075),
    linewidth: float = 2.0,
) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    tick_h = (ylim[1] - ylim[0]) * 0.015

    ax.plot([x0, x0 + length_km], [y0, y0], color="black", linewidth=linewidth, zorder=10)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=linewidth, zorder=10)
    ax.plot(
        [x0 + length_km, x0 + length_km],
        [y0 - tick_h, y0 + tick_h],
        color="black",
        linewidth=linewidth,
        zorder=10,
    )
    ax.text(
        x0 + length_km / 2,
        y0 + tick_h * 1.8,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75),
        zorder=10,
    )


def add_panel_label(ax: Axes, label: str, x: float = 0.015, y: float = 0.975) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.35", linewidth=0.4),
        zorder=20,
    )


def format_map_axis(ax: Axes, df: pd.DataFrame, *, label_edges: bool = True) -> None:
    ax.set_xlim(df["X_km"].min(), df["X_km"].max())
    ax.set_ylim(df["Y_km"].min(), df["Y_km"].max())
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=8, direction="out", length=3, width=0.6)
    if label_edges:
        ax.set_xlabel("Easting (km)", fontsize=9)
        ax.set_ylabel("Northing (km)", fontsize=9)
    ax.grid(True, color="0.85", linewidth=0.45, linestyle="-", zorder=0)


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 600, pad_inches: float = 0.06) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)


def copy_pair(src_stem: Path, dst_stem: Path) -> None:
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        shutil.copy2(src_stem.with_suffix(suffix), dst_stem.with_suffix(suffix))


def percentile_bounds(values: np.ndarray, lower: float = 2.0, upper: float = 98.0) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(valid, [lower, upper])
    if np.isclose(lo, hi):
        pad = abs(lo) * 0.05 if lo else 1.0
        return lo - pad, hi + pad
    return float(lo), float(hi)
