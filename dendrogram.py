"""dendrogram.py — TITAN スライド埋め込みの群間比較樹形図（補正前後）。

Usage:
    uv run python dendrogram.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import cosine_similarity as cos_sim, euclidean_distances

from utils.display import make_abbrev, ordered_subtypes
from utils.loader import load_config, load_data


def save_legend(abbrev: dict[str, str], path: Path, title: str) -> None:
    rows = list(abbrev.items())
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9, max(2.0, n * 0.38 + 1.2)))
    ax.axis("off")
    tbl = ax.table(cellText=[[ab, full] for full, ab in rows],
                   colLabels=["Abbrev", "Full name"], cellLoc="left", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.auto_set_column_width([0, 1])
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e0e0e0")
    ax.set_title(title, fontsize=12, pad=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
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
    ("cos_mean", "Cosine Similarity between Mean Vectors", True,
     lambda X, Y: cos_sim(X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True))[0, 0]),
    ("euc_mean", "Euclidean Distance between Mean Vectors", False,
     lambda X, Y: np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0))),
    ("cos_all",  "Average Cosine Similarity of All Pairs", True,
     lambda X, Y: cos_sim(X, Y).mean()),
    ("euc_all",  "Average Euclidean Distance of All Pairs", False,
     lambda X, Y: euclidean_distances(X, Y).mean()),
    ("energy",   "Energy Distance", False, energy_distance),
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

def save_dendrogram(mat: pd.DataFrame, title: str, path: Path,
                    is_similarity: bool = False,
                    marker_specs: dict | None = None) -> None:
    """
    marker_specs: {label: {"color": hex, "marker": "o"/"s",
                            "subtype": full_name, "source": src, "short": abbrev}}
    When provided, draws subtype-colored source-shaped markers below each leaf
    and Rectangle frames around adjacent same-subtype pairs.
    """
    norm_v = _normalize(mat.values.astype(float))
    dist_v = (1 - norm_v) if is_similarity else norm_v
    np.fill_diagonal(dist_v, 0)
    Z = linkage(squareform(dist_v, checks=False), method="ward")
    n = len(mat)
    fig, ax = plt.subplots(figsize=(24, 6))
    result = dendrogram(
        Z, labels=mat.index.tolist(), ax=ax,
        no_labels=True, 
        leaf_rotation=0, leaf_font_size=15,
        color_threshold=0, above_threshold_color="gray",
    )
    # ax.set_title(title, fontsize=30)
    # ax.set_xlabel("subtype", fontsize=24)
    # ax.set_ylabel("distance", fontsize=24)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.tick_params(axis="y", labelsize=16)

    if not marker_specs:
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved: {path}")
        return

    ivl = result["ivl"]
    n_leaves = len(ivl)
    # scipy dendrogram places leaf i at x = 5 + 10*i
    leaf_xs = [5 + 10 * i for i in range(n_leaves)]

    _, ymax = ax.get_ylim()
    band     = ymax * 0.12        # マーカー帯の高さ（y単位）
    marker_y = -band * 0.55       # 帯の中央
    pad_x    = 2.0                # x単位（葉間隔 = 10）
    pad_y    = band * 0.12        # y単位

    # ── pair frames ───────────────────────────────────────────────────────────
    for i in range(n_leaves - 1):
        s1 = marker_specs.get(ivl[i], {}).get("subtype")
        s2 = marker_specs.get(ivl[i + 1], {}).get("subtype")
        if s1 and s1 == s2:
            color = marker_specs[ivl[i]]["color"]
            xi, xj = leaf_xs[i], leaf_xs[i + 1]
            x0 = xi - 5 + pad_x
            w  = (xj + 5 - pad_x) - x0
            y0 = marker_y - band * 0.40
            h  = (0.0 - pad_y) - y0
            if w > 0 and h > 0:
                ax.add_patch(Rectangle(
                    (x0, y0), w, h,
                    linewidth=2.0, edgecolor=color, facecolor="none",
                    linestyle="--", clip_on=False, zorder=4,
                ))

    # ── subtype-colored, source-shaped markers ─────────────────────────────────
    for i, lbl in enumerate(ivl):
        spec = marker_specs.get(lbl, {})
        ax.scatter([leaf_xs[i]], [marker_y],
                   color=spec.get("color", "gray"),
                   marker=spec.get("marker", "o"),
                   s=250, zorder=6, clip_on=False)

    ax.set_ylim(marker_y - ymax * 0.05, ymax)

    # ── legend ─────────────────────────────────────────────────────────────────
    # seen_sources: dict[str, str] = {}
    # for spec in marker_specs.values():
    #     src = spec.get("source", "")
    #     if src and src not in seen_sources:
    #         seen_sources[src] = spec.get("marker", "o")
    # handles: list = [
    #     Line2D([0], [0], marker=mk, linestyle="none",
    #            markerfacecolor="dimgray", markeredgecolor="dimgray",
    #            markersize=18, label=src)
    #     for src, mk in seen_sources.items()
    # ]
    # seen_subtypes: dict[str, tuple[str, str]] = {}
    # for lbl in ivl:
    #     spec = marker_specs.get(lbl, {})
    #     sub = spec.get("subtype", "")
    #     if sub and sub not in seen_subtypes:
    #         seen_subtypes[sub] = (spec.get("color", "gray"), spec.get("short", sub[:4]))
    # for _sub, (color, short) in seen_subtypes.items():
    #     handles.append(Line2D([0], [0], marker="o", linestyle="none",
    #                            markerfacecolor=color, markeredgecolor=color,
    #                            markersize=18, label=short))
    # handles.append(Patch(fill=False, linestyle="--", edgecolor="gray",
    #                      linewidth=2.5, label="adj. pair"))
    # ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
    #           fontsize=17, title="Legend", title_fontsize=19,
    #           framealpha=0.9, ncol=1,
    #           handletextpad=1.2, labelspacing=1.0, borderpad=1.2)

# ── save ───────────────────────────────────────────────────────────────────────

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {path}")


# ── run ───────────────────────────────────────────────────────────────────────

def _run_metrics(vecs: dict, scope: str, out_dir: Path, cfg: dict,
                 marker_specs: dict | None = None) -> None:
    for slug, title, is_sim, fn in METRICS:
        mat = build_matrix(vecs, fn, is_sim)
        save_dendrogram(mat, f"{title} [{scope}]",
                        out_dir / f"{slug}_dendrogram_{scope}.png",
                        is_sim, marker_specs)
    if cfg.get("comparison", {}).get("mmd", False):
        from hyppo.ksample import MMD
        mmd_kernels = cfg.get("comparison", {}).get("mmd_kernels", [])
        for kernel in mmd_kernels:
            try:
                mat = build_matrix(
                    vecs, lambda X, Y, k=kernel: MMD(compute_kernel=k).statistic(X, Y), False)
                save_dendrogram(mat, f"MMD (kernel={kernel}) [{scope}]",
                                out_dir / f"mmd_{kernel}_dendrogram_{scope}.png",
                                False, marker_specs)
            except Exception as e:
                print(f"  MMD kernel={kernel} skipped: {e}")


def run(merged: pd.DataFrame, out_dir: Path, cfg: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shortened: dict = cfg.get("display", {}).get("shortened", {}).get("subtypes", {})
    subtype_colors: dict = cfg.get("display", {}).get("colors", {}).get("subtypes", {})
    _src_syms = cfg.get("display", {}).get("markers", {}).get("sources", {})
    src_marker = dict(_src_syms)

    # per-source (no cross-site distinction needed)
    for source, grp in merged.groupby("source"):
        subtypes = ordered_subtypes(set(grp["subtype"].unique()), cfg)
        abbrev = make_abbrev(subtypes, cfg)
        save_legend(abbrev, out_dir / f"legend_{source}.png", f"Legend [{source}]")
        vecs = {abbrev[s]: np.stack(grp.loc[grp["subtype"] == s, "embedding"].values)
                for s in subtypes}
        print(f"  [{source}] {len(subtypes)} subtypes")
        _run_metrics(vecs, str(source), out_dir, cfg)

    # cross-source: color = subtype, shape = source
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]),
        cfg)
    if len(common) < 2:
        return
    raw_labels = [f"{src}__{sub}" for sub in common for src in sources]
    abbrev = make_abbrev(raw_labels, cfg)
    save_legend(abbrev, out_dir / "legend_cross.png", "Legend [cross]")
    vecs = {
        abbrev[f"{src}__{sub}"]: np.stack(
            merged.loc[(merged["source"] == src) & (merged["subtype"] == sub), "embedding"].values)
        for sub in common for src in sources
    }
    marker_specs = {
        abbrev[f"{src}__{sub}"]: {
            "color":   subtype_colors.get(sub, "gray"),
            "marker":  src_marker.get(src, "o"),
            "subtype": sub,
            "source":  src,
            "short":   shortened.get(sub, sub[:4].capitalize()),
        }
        for sub in common for src in sources
    }
    print(f"  [cross] {len(vecs)} groups")
    _run_metrics(vecs, "cross", out_dir, cfg, marker_specs)


def main() -> None:
    cfg = load_config()
    out_root = Path(cfg["output_dir"]) / "dendrogram"
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
