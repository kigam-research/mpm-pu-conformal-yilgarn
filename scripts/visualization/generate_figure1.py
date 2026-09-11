#!/usr/bin/env python3
"""Generate journal-ready Figure 1 panels."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import box

from _figure_utils import (
    add_north_arrow,
    add_panel_label,
    add_scale_bar,
    copy_pair,
    format_map_axis,
    geometry_to_km,
    load_boundary_km as _load_boundary_km,
    percentile_bounds,
    plot_boundary,
    save_figure as _save_figure,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass(frozen=True)
class Config:
    project_root: Path = Path(__file__).resolve().parents[2]
    data_path: Path = project_root / "data" / "Yilgarn_GIS_DB.csv"
    boundary_path: Path = project_root / "data" / "boundary" / "gadm41_AUS_0.shp"
    geology_dir: Path = project_root / "data" / "geology_gswa"
    geology_polygons_path: Path = geology_dir / "500k_interpgeop.shp"
    crustal_boundaries_path: Path = geology_dir / "MajorCrustalBoundaries_2015.shp"
    final_dir: Path = project_root / "figures"
    figure1_dir: Path = project_root / "outputs" / "figures" / "figure1"
    data_out_dir: Path = project_root / "outputs" / "data"
    variogram_json: Path = data_out_dir / "variogram_computed.json"

    block_size_km: float = 50.0

    n_lags: int = 20
    maxlag_km: float = 300.0
    downsample_factor: int = 2
    random_seed: int = 42

    dpi: int = 600
    geology_clip_margin_m: float = 25_000.0
    ultramafic_near_distance_m: float = 5_000.0


FEATURES: list[tuple[str, str, str, str]] = [
    ("Prox_Ultramafic Source", "Ultramafic proximity", "Distance (km)", "viridis_r"),
    ("Prox_MajorCrustal", "Major-crustal proximity", "Distance (km)", "viridis_r"),
    ("Prox_Fault", "Fault proximity", "Distance (km)", "viridis_r"),
    ("Interpreted Bedrock Age (MA)", "Interpreted bedrock age", "Age (Ma)", "plasma"),
    ("worms_mag_1km", "Magnetic worms (1 km)", "Lineament density", "magma"),
    ("worms_grav_1km", "Gravity worms (1 km)", "Lineament density", "magma"),
    ("AuSREM", "AuSREM LAB thickness", "Thickness (km)", "cividis"),
    ("K_pct", "Potassium", "K (%)", "YlOrBr"),
]


ULTRAMAFIC_HOST_CLASS = "Ultramafic volcanic"
MAFIC_VOLCANIC_CLASS = "Mafic volcanic"
MAFIC_INTRUSIVE_CLASS = "Mafic to ultramafic intrusive"
GRANITOID_CLASS = "Granitoid & felsic intrusive"
FELSIC_VOLCANIC_CLASS = "Felsic volcanic"
SEDIMENTARY_CLASS = "Sedimentary & metasedimentary"
OTHER_CLASS = "Other/undivided"
GREENSTONE_CLASSES = {ULTRAMAFIC_HOST_CLASS, MAFIC_VOLCANIC_CLASS, MAFIC_INTRUSIVE_CLASS}

LITHOLOGY_CLASSES: list[str] = [
    ULTRAMAFIC_HOST_CLASS,
    MAFIC_VOLCANIC_CLASS,
    MAFIC_INTRUSIVE_CLASS,
    GRANITOID_CLASS,
    FELSIC_VOLCANIC_CLASS,
    SEDIMENTARY_CLASS,
    OTHER_CLASS,
]

LITHOLOGY_DRAW_ORDER: list[str] = [
    OTHER_CLASS,
    SEDIMENTARY_CLASS,
    GRANITOID_CLASS,
    FELSIC_VOLCANIC_CLASS,
    MAFIC_INTRUSIVE_CLASS,
    MAFIC_VOLCANIC_CLASS,
    ULTRAMAFIC_HOST_CLASS,
]

LITHOLOGY_PALETTE: dict[str, str] = {
    ULTRAMAFIC_HOST_CLASS: "#83a883",
    MAFIC_VOLCANIC_CLASS: "#8fb8ad",
    MAFIC_INTRUSIVE_CLASS: "#8d9cb8",
    GRANITOID_CLASS: "#c9ada7",
    FELSIC_VOLCANIC_CLASS: "#b8aec8",
    SEDIMENTARY_CLASS: "#c8bd8d",
    OTHER_CLASS: "#dddddd",
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(Config.data_path)
    df["X_km"] = df["TMX"] / 1000.0
    df["Y_km"] = df["TMY"] / 1000.0
    return df


def load_boundary_km(df: pd.DataFrame):
    return _load_boundary_km(Config.boundary_path, df)


def target_extent_m(df: pd.DataFrame, margin_m: float = Config.geology_clip_margin_m):
    return box(
        df["TMX"].min() - margin_m,
        df["TMY"].min() - margin_m,
        df["TMX"].max() + margin_m,
        df["TMY"].max() + margin_m,
    )


def group_rocktype(rocktype: object) -> str:
    if pd.isna(rocktype):
        return OTHER_CLASS

    text = str(rocktype).strip().lower()
    if "ultramafic volcanic" in text:
        return ULTRAMAFIC_HOST_CLASS
    if "mafic volcanic" in text:
        return MAFIC_VOLCANIC_CLASS
    if "mafic intrusive" in text or "ultramafic intrusive" in text or text == "meta-igneous mafic":
        return MAFIC_INTRUSIVE_CLASS
    if "felsic volcanic" in text:
        return FELSIC_VOLCANIC_CLASS
    if "granitic" in text or "felsic intrusive" in text:
        return GRANITOID_CLASS
    if "sedimentary" in text or "metasedimentary" in text or "carbonate" in text:
        return SEDIMENTARY_CLASS
    return OTHER_CLASS


def read_gswa_layer_3857(path: Path, extent) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required GSWA shapefile is missing: {path}")

    gdf = gpd.read_file(path, engine="pyogrio")
    if gdf.crs is None:
        raise ValueError(f"Required GSWA shapefile has no CRS: {path}")

    gdf = gdf.to_crs("EPSG:3857").clip(extent)
    return gdf.loc[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()


def scale_geodataframe_to_km(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    scaled = gdf.copy()
    scaled["geometry"] = scaled.geometry.apply(geometry_to_km)
    scaled.crs = None
    return scaled


def load_geology_context(
    df: pd.DataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, dict[str, object]]:
    extent = target_extent_m(df)
    geology_m = read_gswa_layer_3857(Config.geology_polygons_path, extent)
    crustal_m = read_gswa_layer_3857(Config.crustal_boundaries_path, extent)
    positive_cells = df.loc[df["Ni_mine"] == 1, ["TMX", "TMY"]].reset_index(drop=True)
    if len(positive_cells) != 818:
        raise ValueError(f"Expected 818 positive Ni cells, found {len(positive_cells)}")
    deposit_markers_m = gpd.GeoDataFrame(
        positive_cells,
        geometry=gpd.points_from_xy(positive_cells["TMX"], positive_cells["TMY"]),
        crs="EPSG:3857",
    )

    geology_m["lithology_group"] = geology_m["ROCKTYPE1"].apply(group_rocktype)
    group_counts = geology_m["lithology_group"].value_counts().reindex(LITHOLOGY_CLASSES, fill_value=0)

    greenstone_geoms = geology_m.loc[geology_m["lithology_group"].isin(GREENSTONE_CLASSES), "geometry"]
    if greenstone_geoms.empty:
        greenstone_on_count = 0
    else:
        greenstone_union = greenstone_geoms.union_all()
        greenstone_on_count = int(deposit_markers_m.geometry.intersects(greenstone_union).sum())
    deposit_count = int(len(deposit_markers_m))
    greenstone_fraction = greenstone_on_count / deposit_count

    summary = {
        "crs": "EPSG:3857",
        "geology_polygons": int(len(geology_m)),
        "crustal_boundaries": int(len(crustal_m)),
        "deposit_count": deposit_count,
        "deposit_marker_source": "Yilgarn_GIS_DB.csv Ni_mine==1 TMX/TMY",
        "greenstone_on_count": greenstone_on_count,
        "greenstone_fraction": greenstone_fraction,
        "group_counts": {label: int(group_counts.loc[label]) for label in LITHOLOGY_CLASSES},
    }
    return (
        scale_geodataframe_to_km(geology_m),
        scale_geodataframe_to_km(crustal_m),
        scale_geodataframe_to_km(deposit_markers_m),
        summary,
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    _save_figure(fig, stem, dpi=Config.dpi, pad_inches=0.06)


def create_study_area_panel(df: pd.DataFrame, boundary_km) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(6.6, 6.2))

    geology_km, crustal_km, deposit_markers_km, summary = load_geology_context(df)

    ax.set_facecolor("#f8f8f8")

    for lithology_class in LITHOLOGY_DRAW_ORDER:
        layer = geology_km.loc[geology_km["lithology_group"] == lithology_class]
        if layer.empty:
            continue
        layer.plot(
            ax=ax,
            color=LITHOLOGY_PALETTE[lithology_class],
            edgecolor="none",
            linewidth=0,
            rasterized=True,
            zorder=1,
        )

    if not crustal_km.empty:
        crustal_km.plot(ax=ax, color="#252525", linewidth=0.85, linestyle="-", zorder=4)

    plot_boundary(ax, boundary_km, facecolor=None, edgecolor="0.12", linewidth=1.0, alpha=1.0, zorder=5)

    ax.scatter(
        deposit_markers_km.geometry.x,
        deposit_markers_km.geometry.y,
        s=36,
        c="white",
        marker="*",
        alpha=0.92,
        linewidths=0,
        rasterized=True,
        zorder=6,
    )
    ax.scatter(
        deposit_markers_km.geometry.x,
        deposit_markers_km.geometry.y,
        s=22,
        c="#b2182b",
        marker="*",
        edgecolors="white",
        linewidths=0.25,
        alpha=0.95,
        rasterized=True,
        zorder=7,
    )

    format_map_axis(ax, df)
    add_north_arrow(ax)
    add_scale_bar(ax, Config.block_size_km * 4, location=(0.36, 0.055))
    add_panel_label(ax, "(a)")
    ax.set_title(
        "Interpreted bedrock geology and known Ni deposits, Yilgarn Craton",
        fontsize=10.6,
        fontweight="bold",
        pad=7,
    )

    lithology_handles = [
        Patch(facecolor=LITHOLOGY_PALETTE[label], edgecolor="none", label=label)
        for label in LITHOLOGY_CLASSES
        if summary["group_counts"][label] > 0
    ]
    lithology_legend = ax.legend(
        handles=lithology_handles,
        loc="lower left",
        title="Lithology",
        title_fontsize=7.1,
        fontsize=6.7,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.7",
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.2,
        handletextpad=0.45,
    )
    lithology_legend.set_zorder(20)
    ax.add_artist(lithology_legend)

    context_handles = [
        Line2D([0], [0], color="#252525", linewidth=1.1, label="Major crustal boundaries"),
        Line2D([0], [0], color="0.12", linewidth=1.0, label="Craton outline"),
        Line2D(
            [0],
            [0],
            marker="*",
            color="none",
            markerfacecolor="#b2182b",
            markeredgecolor="white",
            markeredgewidth=0.35,
            markersize=8,
            label="Known Ni deposits",
        ),
    ]
    context_legend = ax.legend(
        handles=context_handles,
        loc="lower right",
        fontsize=6.9,
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.7",
        borderpad=0.45,
        labelspacing=0.35,
        handlelength=1.4,
        handletextpad=0.5,
    )
    context_legend.set_zorder(20)

    fig.tight_layout()
    out = Config.figure1_dir / "study_area_deposits"
    save_figure(fig, out)
    copy_pair(out, Config.final_dir / "Figure_1a")
    plt.close(fig)
    return summary


def downsample_for_variogram(df: pd.DataFrame) -> pd.DataFrame:
    unique_x = np.sort(df["TMX"].unique())
    unique_y = np.sort(df["TMY"].unique())
    selected_x = unique_x[:: Config.downsample_factor]
    selected_y = unique_y[:: Config.downsample_factor]
    grid_mask = df["TMX"].isin(selected_x) & df["TMY"].isin(selected_y)
    positive_mask = df["Ni_mine"] == 1
    return df.loc[grid_mask | positive_mask].copy().reset_index(drop=True)


def compute_variogram(df: pd.DataFrame, force: bool = False) -> dict:
    if Config.variogram_json.exists() and not force:
        with Config.variogram_json.open("r", encoding="utf-8") as f:
            return json.load(f)

    from skgstat import Variogram

    sampled = downsample_for_variogram(df)
    coords = sampled[["TMX", "TMY"]].to_numpy()
    values = sampled["Ni_mine"].to_numpy()
    maxlag_m = Config.maxlag_km * 1000.0

    model_summaries: dict[str, dict[str, float]] = {}
    gaussian_result: dict | None = None

    for model in ("spherical", "exponential", "gaussian"):
        variogram = Variogram(
            coordinates=coords,
            values=values,
            n_lags=Config.n_lags,
            maxlag=maxlag_m,
            model=model,
            estimator="matheron",
            fit_method="trf",
        )
        params = variogram.parameters
        summary = {
            "effective_range_km": float(params[0] / 1000.0),
            "sill": float(params[1]),
            "nugget": float(params[2]) if len(params) > 2 else 0.0,
            "rmse": float(variogram.rmse),
        }
        model_summaries[model] = summary
        if model == "gaussian":
            gaussian_result = {
                "model": "gaussian",
                **summary,
                "bins_km": (np.asarray(variogram.bins) / 1000.0).tolist(),
                "experimental": np.asarray(variogram.experimental).tolist(),
                "pair_counts": np.asarray(variogram.bin_count).astype(int).tolist(),
            }

    if gaussian_result is None:
        raise RuntimeError("Gaussian variogram computation failed")

    gaussian_result["all_model_fits"] = model_summaries
    gaussian_result["computation_metadata"] = {
        "original_n_samples": int(len(df)),
        "used_n_samples": int(len(sampled)),
        "positive_samples": int(df["Ni_mine"].sum()),
        "downsample_factor": Config.downsample_factor,
        "n_lags": Config.n_lags,
        "maxlag_km": Config.maxlag_km,
        "random_seed": Config.random_seed,
    }

    Config.data_out_dir.mkdir(parents=True, exist_ok=True)
    with Config.variogram_json.open("w", encoding="utf-8") as f:
        json.dump(gaussian_result, f, indent=2)

    return gaussian_result


def gaussian_curve(h_km: np.ndarray, effective_range_km: float, sill: float, nugget: float) -> np.ndarray:
    a = effective_range_km / np.sqrt(3.0)
    return nugget + (sill - nugget) * (1.0 - np.exp(-3.0 * (h_km / a) ** 2))


def create_variogram_panel(df: pd.DataFrame, force: bool = False) -> dict:
    results = compute_variogram(df, force=force)

    bins_km = np.asarray(results["bins_km"], dtype=float)
    experimental = np.asarray(results["experimental"], dtype=float)
    effective_range = float(results["effective_range_km"])
    sill = float(results["sill"])
    nugget = float(results["nugget"])

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.scatter(
        bins_km,
        experimental,
        s=34,
        color="#2c7fb8",
        edgecolor="white",
        linewidth=0.45,
        label="Empirical semivariance",
        zorder=3,
    )

    x_max = max(Config.maxlag_km, float(np.nanmax(bins_km)) if len(bins_km) else Config.maxlag_km)
    h_fit = np.linspace(0.0, x_max, 400)
    ax.plot(
        h_fit,
        gaussian_curve(h_fit, effective_range, sill, nugget),
        color="#d95f0e",
        linewidth=2.0,
        label="Gaussian fit",
        zorder=4,
    )

    ax.axhline(sill, color="0.45", linestyle="--", linewidth=1.0, label=f"Sill = {sill:.3f}")
    ax.axvline(
        effective_range,
        color="#756bb1",
        linestyle="--",
        linewidth=1.5,
        label=f"Effective range = {effective_range:.1f} km",
    )
    ax.axvline(
        Config.block_size_km,
        color="#238b45",
        linestyle=":",
        linewidth=1.7,
        label=f"Spatial block = {Config.block_size_km:.0f} km",
    )

    ax.set_xlabel("Separation distance (km)", fontsize=10)
    ax.set_ylabel("Semivariance", fontsize=10)
    ax.set_title("Empirical variogram of the binary Ni deposit label", fontsize=11, fontweight="bold", pad=7)
    ax.grid(True, color="0.88", linewidth=0.55)
    ax.tick_params(axis="both", labelsize=8.5, direction="out", length=3, width=0.6)
    ax.set_xlim(0, x_max * 1.02)
    y_max = max(float(np.nanmax(experimental)) if len(experimental) else sill, sill) * 1.18
    ax.set_ylim(0, y_max)
    add_panel_label(ax, "(b)")

    ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.92, edgecolor="0.75")

    fig.tight_layout()
    out = Config.figure1_dir / "target_variogram"
    save_figure(fig, out)
    copy_pair(out, Config.final_dir / "Figure_1b")
    plt.close(fig)
    return results


def feature_grid(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[column].copy()
    if column.startswith("Prox_"):
        values = values / 1000.0
    table = (
        pd.DataFrame({"TMX": df["TMX"], "TMY": df["TMY"], "value": values})
        .pivot(index="TMY", columns="TMX", values="value")
        .sort_index()
        .sort_index(axis=1)
    )
    return table.columns.to_numpy() / 1000.0, table.index.to_numpy() / 1000.0, table.to_numpy()


def draw_feature_axis(
    fig: plt.Figure,
    ax: Axes,
    df: pd.DataFrame,
    boundary_km,
    column: str,
    title: str,
    unit: str,
    cmap: str | Colormap,
    *,
    row: int,
    col: int,
) -> None:
    x, y, z = feature_grid(df, column)
    vmin, vmax = percentile_bounds(z)
    extent = (x.min(), x.max(), y.min(), y.max())

    image = ax.imshow(
        np.ma.masked_invalid(z),
        extent=extent,
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    plot_boundary(ax, boundary_km, facecolor=None, edgecolor="0.15", linewidth=0.45, zorder=5)
    format_map_axis(ax, df, label_edges=False)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=3)

    if row == 1:
        ax.set_xlabel("Easting (km)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_xticklabels([])
    if col == 0:
        ax.set_ylabel("Northing (km)", fontsize=7.6, labelpad=1.5)
    else:
        ax.set_yticklabels([])
    ax.tick_params(labelsize=6.4, length=2.2, width=0.45)

    cbar = fig.colorbar(image, ax=ax, shrink=0.82, aspect=18, pad=0.018)
    cbar.set_label(unit, fontsize=6.9, labelpad=2)
    cbar.ax.tick_params(labelsize=6.1, length=2.0, width=0.4)
    cbar.locator = mticker.MaxNLocator(4)
    cbar.update_ticks()


def create_feature_grid_panel(df: pd.DataFrame, boundary_km) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(12.8, 6.05))
    fig.subplots_adjust(left=0.045, right=0.988, bottom=0.085, top=0.895, wspace=0.16, hspace=0.075)

    for idx, (column, title, unit, cmap) in enumerate(FEATURES):
        row, col = divmod(idx, 4)
        draw_feature_axis(fig, axes[row, col], df, boundary_km, column, title, unit, cmap, row=row, col=col)

    add_panel_label(axes[0, 0], "(c)")
    add_north_arrow(axes[0, 3], x=0.88, y=0.87, size=0.10)
    add_scale_bar(axes[1, 0], length_km=200.0, location=(0.08, 0.08), linewidth=1.4)
    fig.suptitle("Representative geological and geophysical predictor layers", fontsize=12, fontweight="bold", y=0.985)

    out = Config.figure1_dir / "feature_grid_2x4"
    save_figure(fig, out)
    copy_pair(out, Config.final_dir / "Figure_1c")
    plt.close(fig)


def create_combined_preview() -> None:
    """Create a single-canvas preview for the Springer no-subfigure workflow."""
    pngs = [
        Config.final_dir / "Figure_1a.png",
        Config.final_dir / "Figure_1b.png",
        Config.final_dir / "Figure_1c.png",
    ]
    if not all(path.exists() for path in pngs):
        return

    images = [plt.imread(path) for path in pngs]
    fig = plt.figure(figsize=(13.8, 12.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.08], hspace=0.035, wspace=0.03)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])]
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis("off")
    out = Config.final_dir / "Figure_1"
    save_figure(fig, out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate journal-ready Figure 1 assets.")
    parser.add_argument("--recompute-variogram", action="store_true", help="Recompute variogram cache.")
    parser.add_argument(
        "--skip-combined-preview",
        action="store_true",
        help="Do not create figures/Figure_1.{pdf,png} preview composite.",
    )
    args = parser.parse_args()

    Config.final_dir.mkdir(parents=True, exist_ok=True)
    Config.figure1_dir.mkdir(parents=True, exist_ok=True)
    Config.data_out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data()
    boundary_km = load_boundary_km(df)

    study_summary = create_study_area_panel(df, boundary_km)
    results = create_variogram_panel(df, force=args.recompute_variogram)
    create_feature_grid_panel(df, boundary_km)
    if not args.skip_combined_preview:
        create_combined_preview()

    print("Figure 1 assets regenerated.")
    print(f"  Figure 1(a): {Config.final_dir / 'Figure_1a.pdf'}")
    print(f"  Figure 1(b): {Config.final_dir / 'Figure_1b.pdf'}")
    print(f"  Figure 1(c): {Config.final_dir / 'Figure_1c.pdf'}")
    print(f"  Combined preview: {Config.final_dir / 'Figure_1.pdf'}")
    print(
        "  Variogram: "
        f"Gaussian range={results['effective_range_km']:.1f} km, "
        f"sill={results['sill']:.4f}, nugget={results['nugget']:.4f}, "
        f"rmse={results['rmse']:.4f}"
    )
    print(
        "  Geology map self-check: "
        f"crs={study_summary['crs']}, "
        f"polygons={study_summary['geology_polygons']:,}, "
        f"major_crustal_boundaries={study_summary['crustal_boundaries']:,}, "
        f"deposits={study_summary['deposit_count']:,}, "
        f"marker_source={study_summary['deposit_marker_source']}, "
        f"within_greenstone={study_summary['greenstone_on_count']:,} "
        f"({study_summary['greenstone_fraction']:.1%})"
    )


if __name__ == "__main__":
    main()
