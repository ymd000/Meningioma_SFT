"""confusion_mtx.py — TITAN スライド埋め込みの群間比較行列（補正前後）。

Usage:
    uv run python confusion_mtx.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity as cos_sim, euclidean_distances

from utils.display import make_abbrev, ordered_subtypes, subtype_color_map
from utils.loader import load_config, load_data


def save_legend(items: list[tuple[str, str, str]], path: Path) -> None:
    """Draw legend: colored marker + display name for each item.

    items: [(marker_style, color, display_name), ...]
    """
    n = len(items)
    max_len = max(len(name) for _, _, name in items)
    width = max(4, min(10, max_len * 0.12 + 2))
    fig, ax = plt.subplots(figsize=(width, max(1.5, n * 0.38 + 0.5)))
    ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker=mkr, color="none",
                   markerfacecolor=color, markersize=14, markeredgewidth=0)
        for mkr, color, _ in items
    ]
    labels = [name for _, _, name in items]
    ax.legend(handles, labels, loc="center", frameon=False,
              fontsize=11, ncol=1, handlelength=1.5, handletextpad=0.8)
    plt.savefig(path, dpi=150, bbox_inches="tight", transparent=True); plt.close()
    print(f"  saved: {path}")


def save_colormap(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1, 16))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([0, 127.5, 255])
    ax.set_yticklabels(
        ["0", "0.5", "1"],
        fontsize=35,
        )
    ax.yaxis.tick_right()
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    plt.savefig(path, dpi=150, bbox_inches="tight", transparent=True); plt.close()
    print(f"  saved: {path}")


# ── matrix ────────────────────────────────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)


def energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return (2 * euclidean_distances(X, Y).mean()
            - euclidean_distances(X, X).mean()
            - euclidean_distances(Y, Y).mean())


METRICS = [
    # (slug, title, is_similarity, flip, fn)
    # flip=True: display as 1 − normalize(dist) so similar pairs → 1
    ("cos_mean", "Cosine Similarity between Mean Vectors", True, False,
     lambda X, Y: cos_sim(X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True))[0, 0]),
    ("euc_mean", "Euclidean Distance between Mean Vectors", False, True,
     lambda X, Y: np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))),
    ("cos_all",  "Average Cosine Similarity of All Pairs", True, False,
     lambda X, Y: cos_sim(X, Y).mean()),
    ("euc_all",  "Average Euclidean Distance of All Pairs", False, True,
     lambda X, Y: euclidean_distances(X, Y).mean()),
    ("energy",   "Energy Distance", False, False, energy_distance),
]


def build_matrix(groups: dict, compute_fn, is_similarity: bool) -> pd.DataFrame:
    labels = list(groups.keys())
    n = len(labels)
    mat = pd.DataFrame(np.full((n, n), 1.0 if is_similarity else 0.0),
                       index=labels, columns=labels)
    for l1, l2 in combinations(labels, 2):
        val = float(compute_fn(groups[l1], groups[l2]))
        mat.loc[l1, l2] = val; mat.loc[l2, l1] = val
    return mat


# ── plot ──────────────────────────────────────────────────────────────────────

def _add_axis_markers(ax: plt.Axes, labels: list[str],
                       marker_specs: dict, n: int) -> None:
    ax.set_xticklabels([""] * n)
    ax.set_yticklabels([""] * n)
    for i, lbl in enumerate(labels):
        spec = marker_specs.get(lbl, {})
        color = spec.get("color", "dimgray")
        mkr = spec.get("marker", "o")
        x_frac = (i + 0.5) / n
        y_frac = 1.0 - (i + 0.5) / n
        ax.plot([x_frac], [-0.01], marker=mkr, color=color, markersize=10,
                transform=ax.transAxes, clip_on=False, linestyle="none")
        ax.plot([-0.01], [y_frac], marker=mkr, color=color, markersize=10,
                transform=ax.transAxes, clip_on=False, linestyle="none")


def save_heatmap(mat: pd.DataFrame, title: str, path: Path,
                 is_similarity: bool = False,
                 flip: bool = False,
                 marker_specs: dict | None = None,
                 group_size: int | None = None,
                 label_order: list | None = None) -> None:
    """
    marker_specs: {label: {"color": hex, "marker": "o"/"s"}}
    When provided, draws matplotlib markers on tick positions and group lines.
    """
    if label_order:
        mat = mat.loc[label_order, label_order]
    norm_v = _normalize(mat.values.astype(float))
    display_v = (1 - norm_v) if flip else norm_v
    cmap = "viridis"
    cbar_label = "1 − normalized" if flip else "normalized"
    norm_mat = pd.DataFrame(display_v, index=mat.index, columns=mat.columns)
    n = len(mat)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), max(7, n * 1.3)))
    sns.heatmap(norm_mat, ax=ax, annot=True, fmt=".2f", annot_kws={"size": 20},
                cmap=cmap, square=True, vmin=0, vmax=1, cbar=False)
    # ax.set_xlabel("subtype", fontsize=11)
    # ax.set_ylabel("subtype", fontsize=11)
    # ax.tick_params(axis="x", labelsize=9, rotation=45)
    # ax.tick_params(axis="y", labelsize=9, rotation=0)
    # ax.set_title(title, fontsize=12, pad=50)
    if marker_specs:
        # subtype color strips (right edge and top edge of heatmap)    
        # ax.set_xticklabels([""] * n)  # ラベルを消す
        # ax.set_yticklabels([""] * n)  # ラベルを消す
        # ax.set_xticks([])   # ← ティック線を消す
        # ax.set_yticks([])   # ← ティック線を消す
        # sw = 0.35
        # for i, lbl in enumerate(mat.index.tolist()):
        #     c = marker_specs.get(lbl, {}).get("color", "lightgray")
        #     ax.add_patch(plt.Rectangle((-sw - 0.1, i), sw, 1, color=c, clip_on=False))
        #     ax.add_patch(plt.Rectangle((i, -sw - 0.1), 1, sw, color=c, clip_on=False))
        
        # source-marker glyphs embedded in tick labels
        _add_axis_markers(ax, mat.index.tolist(), marker_specs, n)
        
    if group_size:
        for k in range(group_size, n, group_size):
            ax.axvline(k, color="white", linewidth=2.5, zorder=5)
            ax.axhline(k, color="white", linewidth=2.5, zorder=5)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ── run ───────────────────────────────────────────────────────────────────────

def _run_metrics(vecs: dict, scope: str, out_dir: Path, cfg: dict,
                 marker_specs: dict | None = None,
                 group_size: int | None = None) -> None:
    label_order = list(vecs.keys())
    for slug, title, is_sim, flip, fn in METRICS:
        mat = build_matrix(vecs, fn, is_sim)
        save_heatmap(mat, f"{title} [{scope}]",
                     out_dir / f"{slug}_heatmap_{scope}.png",
                     is_sim, flip, marker_specs, group_size, label_order)
    if cfg.get("comparison", {}).get("mmd", False):
        from hyppo.ksample import MMD
        mmd_kernels = cfg.get("comparison", {}).get("mmd_kernels", [])
        for kernel in mmd_kernels:
            try:
                mat = build_matrix(
                    vecs, lambda X, Y, k=kernel: MMD(compute_kernel=k).statistic(X, Y), False)
                save_heatmap(mat, f"MMD (kernel={kernel}) [{scope}]",
                             out_dir / f"mmd_{kernel}_heatmap_{scope}.png",
                             False, False, marker_specs, group_size, label_order)
            except Exception as e:
                print(f"  MMD kernel={kernel} skipped: {e}")


def run(merged: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    save_colormap(out_dir / "colormap.png")
    _src_syms = cfg.get("display", {}).get("markers", {}).get("sources", {})
    src_marker = dict(_src_syms)

    # per-source
    for source, grp in merged.groupby("source"):
        subtypes = ordered_subtypes(set(grp["subtype"].unique()), cfg)
        sub_color_map = subtype_color_map(subtypes, cfg)
        abbrev = make_abbrev(subtypes, cfg)
        mkr = src_marker.get(str(source), "o")
        legend_items = [(mkr, sub_color_map[s], abbrev[s]) for s in subtypes]
        save_legend(legend_items, out_dir / f"legend_{source}.png")
        vecs = {abbrev[s]: np.stack(grp.loc[grp["subtype"] == s, "embedding"].values)
                for s in subtypes}
        print(f"  [{source}] {len(subtypes)} subtypes")
        _run_metrics(vecs, str(source), out_dir, cfg)

    # cross-source
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]),
        cfg)
    if len(common) < 2:
        return
    sub_color_map = subtype_color_map(common, cfg)
    raw_labels = [f"{src}__{sub}" for sub in common for src in sources]
    abbrev = make_abbrev(raw_labels, cfg)
    legend_items = [
        (src_marker.get(src, "o"), sub_color_map[sub], abbrev[f"{src}__{sub}"])
        for sub in common for src in sources
    ]
    save_legend(legend_items, out_dir / "legend_cross.png")
    vecs = {
        abbrev[f"{src}__{sub}"]: np.stack(
            merged.loc[(merged["source"] == src) & (merged["subtype"] == sub), "embedding"].values)
        for sub in common for src in sources
    }
    marker_specs = {
        abbrev[f"{src}__{sub}"]: {
            "color":  sub_color_map[sub],
            "marker": src_marker.get(src, "o"),
        }
        for sub in common for src in sources
    }
    print(f"  [cross] {len(vecs)} groups")
    _run_metrics(vecs, "cross", out_dir, cfg, marker_specs, len(sources))


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["output_dir"]) / "confusion_mtx"
    variants: dict[str, str] = cfg.get("variants", {"original": "original"})

    for variant, dir_key in variants.items():
        print(f"\n[{variant}]")
        merged = load_data(cfg, dir_key)
        if merged.empty:
            print("  SKIP: no data")
            continue
        print(f"  {len(merged)} cases")
        run(merged, out_root / variant, cfg)


if __name__ == "__main__":
    main()
