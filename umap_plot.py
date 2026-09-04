"""umap_plot.py — TITAN スライド埋め込みの UMAP 可視化。

original / centroid / gan ディレクトリからそれぞれ読み込み、
サブフォルダ別に出力する。

Usage:
    uv run python umap_plot.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap as umap_lib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.display import ordered_subtypes, shorten, subtype_color_map
from utils.loader import load_config, load_data


def run_umap(merged: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    umap_cfg = cfg.get("umap", {})
    src_colors: dict = cfg.get("display", {}).get("colors", {}).get("sources", {})

    print("  fitting UMAP...")
    X = np.stack(merged["embedding"].values)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        coords = umap_lib.UMAP(
            n_components=2,
            n_neighbors=umap_cfg.get("n_neighbors", 15),
            min_dist=umap_cfg.get("min_dist", 0.01),
            random_state=umap_cfg.get("random_state", 42),
            n_jobs=1,
        ).fit_transform(X)

    subtypes = merged["subtype"].values
    sources  = merged["source"].values
    sub_list = ordered_subtypes(set(subtypes), cfg)
    src_list = sorted(set(sources))
    sub_map  = subtype_color_map(sub_list, cfg)
    markers = cfg.get("umap", {}).get("markers", ["o", "s", "^", "D", "v", "P", "X", "*"])
    src_marker = {s: markers[i % len(markers)] for i, s in enumerate(src_list)}

    # plot 1: color=subtype, shape=source
    fig, ax = plt.subplots(figsize=(14, 11))
    for src in src_list:
        for sub in sub_list:
            mask = (subtypes == sub) & (sources == src)
            if not np.any(mask):
                continue
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       facecolors=sub_map[sub], edgecolors="none",
                       alpha=0.7, s=80, marker=src_marker[src], zorder=2)

            # # temporary annotation
            # case_ids = merged["case_id"].values
            # for i in np.where(np.array(["sft" in s.lower() for s in subtypes]))[0]:
            #     ax.annotate(case_ids[i], (coords[i, 0], coords[i, 1]),
            #     fontsize=5, alpha=0.8, va="bottom", ha="center")

    color_handles = [Patch(facecolor=sub_map[s], label=shorten(s, cfg)) for s in sub_list]
    shape_handles = [Line2D([0], [0], marker=src_marker[s], color="gray",
                            linestyle="None", markersize=22, label=s) for s in src_list]
    ax.legend(handles=color_handles + shape_handles, loc="best", fontsize=22, ncol=2)
    ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "umap_subtype_source.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")

    # plot 2: color=source
    fig, ax = plt.subplots(figsize=(12, 10))
    for src in src_list:
        mask = sources == src
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   facecolors=src_colors.get(src, "gray"), edgecolors="none",
                   alpha=0.6, s=80, marker="o", label=src, zorder=2)
    ax.legend(loc="best", fontsize=22)
    ax.set_axis_off()
    plt.tight_layout()
    path = out_dir / "umap_source.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["output_dir"]) / "umap"
    variants: dict[str, str] = cfg.get("variants", {"original": "original"})

    for variant, dir_key in variants.items():
        print(f"\n[{variant}]")
        merged = load_data(cfg, dir_key)
        if merged.empty:
            print("  SKIP: no data")
            continue
        print(f"  {len(merged)} cases")
        out_dir = out_root / variant
        out_dir.mkdir(parents=True, exist_ok=True)
        run_umap(merged, out_dir, cfg)


if __name__ == "__main__":
    main()
