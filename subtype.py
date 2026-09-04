"""subtype.py — サブタイプ平均ベクトルの UMAP 可視化。

centroid 補正後の埋め込みを使い、(施設 × サブタイプ) の平均ベクトル (~30点) で
UMAP をフィット・プロット。各点にサブタイプ略称を注釈。

Usage:
    uv run python subtype.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap as umap_lib

from utils.display import ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data


def compute_means(merged: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (source, subtype), grp in merged.groupby(["source", "subtype"]):
        mean_vec = np.stack(grp["embedding"].values).mean(axis=0)
        records.append({
            "source": source,
            "subtype": subtype,
            "embedding": mean_vec,
            "n": len(grp),
        })
    return pd.DataFrame(records)


def run(means: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    umap_cfg = cfg.get("umap", {})
    markers = umap_cfg.get("markers", ["o", "s", "^", "D", "v", "P", "X", "*"])

    sub_list = ordered_subtypes(set(means["subtype"]), cfg)
    src_list = sorted(means["source"].unique())
    sub_map = subtype_color_map(sub_list, cfg)
    src_marker = {s: markers[i % len(markers)] for i, s in enumerate(src_list)}

    X = np.stack(means["embedding"].values)
    n_neighbors = min(umap_cfg.get("n_neighbors", 15), len(X) - 1)
    print(f"  fitting UMAP on {len(X)} mean vectors (n_neighbors={n_neighbors})...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        coords = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=umap_cfg.get("min_dist", 0.01),
            random_state=umap_cfg.get("random_state", 42),
            n_jobs=1,
        ).fit_transform(X)

    subtypes = means["subtype"].values
    sources  = means["source"].values

    fig, ax = plt.subplots(figsize=(14, 11))
    for i, (src, sub) in enumerate(zip(sources, subtypes)):
        ax.scatter(coords[i, 0], coords[i, 1],
                   facecolors=sub_map[sub], edgecolors="white", linewidths=0.5,
                   s=300, marker=src_marker[src], zorder=3)
        ax.annotate(
            shorten(sub, cfg),
            (coords[i, 0], coords[i, 1]),
            fontsize=18, ha="center", va="bottom",
            xytext=(0, 6), textcoords="offset points",
            color=sub_map[sub],
        )

    from matplotlib.lines import Line2D
    shape_handles = [
        Line2D([0], [0], marker=src_marker[s], color="gray",
               linestyle="None", markersize=16, label=s)
        for s in src_list
    ]
    ax.legend(handles=shape_handles, loc="best", fontsize=14)
    ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "subtype_mean_umap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  saved: {path}")


def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "subtype"
    merged = load_data(cfg, "centroid")
    if merged.empty:
        print("SKIP: no data")
        return
    print(f"  {len(merged)} cases loaded")
    means = compute_means(merged)
    print(f"  {len(means)} (source, subtype) groups")
    run(means, out_dir, cfg)


if __name__ == "__main__":
    main()
