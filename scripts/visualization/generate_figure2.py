#!/usr/bin/env python3
"""Generate journal-ready Figure 2(b) spatial fold map."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-nnar")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "DejaVu Sans"

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from PIL import Image, ImageChops, ImageOps
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon, box
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    folds_path: Path = project_root / "outputs" / "folds" / "outer_fold_assignments.csv"
    boundary_path: Path = project_root / "data" / "boundary" / "gadm41_AUS_0.shp"
    final_dir: Path = project_root / "figures"
    figure2_dir: Path = project_root / "outputs" / "figures" / "figure2"
    fold_distribution_dir: Path = project_root / "outputs" / "figures" / "fold_distribution"
    figure2a_pdf: Path = final_dir / "Figure_2a.pdf"
    block_size_km: float = 50.0
    cell_size_km: float = 5.0
    dpi: int = 600
    render_dpi: int = 300


FOLD_COLORS = {
    0: "#e41a1c",
    1: "#377eb8",
    2: "#4daf4a",
    3: "#984ea3",
    4: "#ff7f00",
}


def load_fold_assignments() -> pd.DataFrame:
    if not Config.folds_path.exists():
        raise FileNotFoundError(f"Outer fold assignments not found: {Config.folds_path}")
    df = pd.read_csv(Config.folds_path)
    df["X_km"] = df["X"] / 1000.0
    df["Y_km"] = df["Y"] / 1000.0
    return df


def geometry_to_km(geometry):
    return shapely_transform(lambda x, y, z=None: (x / 1000.0, y / 1000.0), geometry)


def load_boundary_km(df: pd.DataFrame):
    gdf = gpd.read_file(Config.boundary_path)
    geom = gdf.geometry.union_all()
    extent = box(df["X"].min(), df["Y"].min(), df["X"].max(), df["Y"].max())
    return geometry_to_km(geom.intersection(extent))


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


def plot_boundary(ax: Axes, geometry, *, edgecolor: str = "0.18", linewidth: float = 0.75) -> None:
    for polygon in iter_polygons(geometry):
        x, y = polygon.exterior.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, zorder=6)
        for interior in polygon.interiors:
            xi, yi = interior.xy
            ax.plot(xi, yi, color=edgecolor, linewidth=linewidth * 0.55, zorder=6)
    for line in iter_lines(geometry):
        x, y = line.xy
        ax.plot(x, y, color=edgecolor, linewidth=linewidth, zorder=6)


def add_panel_label(ax: Axes, label: str) -> None:
    ax.text(
        0.018,
        0.972,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="0.35", linewidth=0.4),
        zorder=20,
    )


def add_north_arrow(ax: Axes, x: float = 0.92, y: float = 0.87, size: float = 0.12) -> None:
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
        zorder=15,
    )


def add_scale_bar(
    ax: Axes,
    length_km: float = 200.0,
    location: tuple[float, float] = (0.08, 0.075),
    linewidth: float = 1.9,
) -> None:
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x0 = xlim[0] + (xlim[1] - xlim[0]) * location[0]
    y0 = ylim[0] + (ylim[1] - ylim[0]) * location[1]
    tick_h = (ylim[1] - ylim[0]) * 0.013
    ax.plot([x0, x0 + length_km], [y0, y0], color="black", linewidth=linewidth, zorder=15)
    ax.plot([x0, x0], [y0 - tick_h, y0 + tick_h], color="black", linewidth=linewidth, zorder=15)
    ax.plot([x0 + length_km, x0 + length_km], [y0 - tick_h, y0 + tick_h], color="black", linewidth=linewidth, zorder=15)
    ax.text(
        x0 + length_km / 2,
        y0 + tick_h * 1.7,
        f"{int(length_km)} km",
        ha="center",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.78),
        zorder=16,
    )


def format_map_axis(ax: Axes, df: pd.DataFrame) -> None:
    pad = Config.cell_size_km / 2
    ax.set_xlim(df["X_km"].min() - pad, df["X_km"].max() + pad)
    ax.set_ylim(df["Y_km"].min() - pad, df["Y_km"].max() + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Easting (km)", fontsize=9)
    ax.set_ylabel("Northing (km)", fontsize=9)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.tick_params(axis="both", labelsize=8, direction="out", length=3, width=0.6)
    ax.grid(True, color="0.86", linewidth=0.45, zorder=0)


def block_summary(df: pd.DataFrame) -> pd.DataFrame:
    blocks = (
        df.groupby("block_id")
        .agg(
            x_min=("X_km", "min"),
            x_max=("X_km", "max"),
            y_min=("Y_km", "min"),
            y_max=("Y_km", "max"),
            fold_id=("fold_id", "first"),
            n_samples=("index", "count"),
            n_positive=("target", "sum"),
        )
        .reset_index()
    )
    half_cell = Config.cell_size_km / 2
    blocks["x0"] = blocks["x_min"] - half_cell
    blocks["x1"] = blocks["x_max"] + half_cell
    blocks["y0"] = blocks["y_min"] - half_cell
    blocks["y1"] = blocks["y_max"] + half_cell
    return blocks


def draw_fold_blocks(ax: Axes, blocks: pd.DataFrame) -> None:
    for _, block in blocks.iterrows():
        color = FOLD_COLORS[int(block["fold_id"])]
        rect = Rectangle(
            (block["x0"], block["y0"]),
            block["x1"] - block["x0"],
            block["y1"] - block["y0"],
            facecolor=color,
            edgecolor="white",
            linewidth=0.18,
            alpha=0.86,
            zorder=2,
        )
        ax.add_patch(rect)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=Config.dpi, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(stem.with_suffix(".png"), dpi=Config.dpi, bbox_inches="tight", pad_inches=0.06)


def copy_pair(src_stem: Path, dst_stem: Path) -> None:
    dst_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        shutil.copy2(src_stem.with_suffix(suffix), dst_stem.with_suffix(suffix))


def create_figure2b() -> None:
    df = load_fold_assignments()
    boundary = load_boundary_km(df)
    blocks = block_summary(df)
    stats = df.groupby("fold_id").agg(n=("index", "count"), pos=("target", "sum"), blocks=("block_id", "nunique"))

    fig, ax = plt.subplots(figsize=(6.7, 6.15))
    draw_fold_blocks(ax, blocks)

    deposits = df[df["target"] == 1]
    ax.scatter(
        deposits["X_km"],
        deposits["Y_km"],
        marker="o",
        s=5.0,
        c="black",
        edgecolors="none",
        alpha=0.82,
        rasterized=True,
        zorder=8,
    )
    plot_boundary(ax, boundary)
    format_map_axis(ax, df)
    add_panel_label(ax, "(b)")
    add_north_arrow(ax)
    add_scale_bar(ax, length_km=200.0)

    ax.set_title("Spatial distribution of outer CV folds", fontsize=11, fontweight="bold", pad=7)

    legend_handles = [
        Patch(
            facecolor=FOLD_COLORS[i],
            edgecolor="white",
            linewidth=0.4,
            label=f"Fold {i}",
        )
        for i in range(5)
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="none",
            markersize=3.5,
            label="Positive analysis cells",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.012, 0.5),
        ncol=1,
        fontsize=6.2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="0.7",
        borderpad=0.35,
        handlelength=1.25,
        labelspacing=0.38,
        handletextpad=0.35,
    )

    fig.tight_layout()
    stem = Config.figure2_dir / "outer_folds_spatial"
    save_figure(fig, stem)
    copy_pair(stem, Config.final_dir / "Figure_2b")
    copy_pair(stem, Config.fold_distribution_dir / "outer_folds_spatial")
    plt.close(fig)

    print("Figure 2(b) regenerated.")
    print(f"  Final: {Config.final_dir / 'Figure_2b.pdf'}")
    print(f"  Provenance: {stem.with_suffix('.pdf')}")
    print("  Fold statistics:")
    print(stats.to_string())


def crop_white_margin(image: Image.Image, pad_px: int = 18) -> Image.Image:
    rgb = image.convert("RGB")
    background = Image.new("RGB", rgb.size, "white")
    diff = ImageChops.difference(rgb, background)
    bbox = diff.getbbox()
    if bbox is None:
        return rgb
    left = max(bbox[0] - pad_px, 0)
    upper = max(bbox[1] - pad_px, 0)
    right = min(bbox[2] + pad_px, rgb.size[0])
    lower = min(bbox[3] + pad_px, rgb.size[1])
    return rgb.crop((left, upper, right, lower))


def render_pdf_page(pdf_path: Path, tmpdir: Path) -> Image.Image:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Missing PDF for Figure 2 assembly: {pdf_path}")

    prefix = tmpdir / pdf_path.stem
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            str(Config.render_dpi),
            str(pdf_path),
            str(prefix),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return crop_white_margin(Image.open(prefix.with_suffix(".png")))


def save_combined_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=Config.dpi, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(stem.with_suffix(".png"), dpi=Config.dpi, bbox_inches="tight", pad_inches=0.04)


def create_combined_figure2() -> None:
    """Create the single-file Figure 2 required by Springer-style assembly."""
    with tempfile.TemporaryDirectory(prefix="nnar_figure2_") as tmp:
        tmpdir = Path(tmp)
        panel_a = render_pdf_page(Config.figure2a_pdf, tmpdir)
        panel_b = render_pdf_page(Config.final_dir / "Figure_2b.pdf", tmpdir)

    fig = plt.figure(figsize=(13.7, 6.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], left=0.01, right=0.99, bottom=0.02, top=0.98, wspace=0.035)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

    for ax, image in zip(axes, [panel_a, panel_b]):
        ax.imshow(image)
        ax.axis("off")

    axes[0].text(
        0.036,
        0.973,
        "(a)",
        transform=axes[0].transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="0.35", linewidth=0.4),
        zorder=20,
    )

    final_stem = Config.final_dir / "Figure_2"
    provenance_stem = Config.figure2_dir / "Figure_2_combined"
    save_combined_figure(fig, final_stem)
    save_combined_figure(fig, provenance_stem)
    plt.close(fig)

    print(f"  Combined single-file Figure 2: {final_stem.with_suffix('.pdf')}")


if __name__ == "__main__":
    create_figure2b()
    create_combined_figure2()
