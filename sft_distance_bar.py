"""sft_distance_bar.py — SFT からの距離横棒グラフ（補正後・0〜1正規化）。

confusion_mtx.py の cross 行列と同一ロジックで (共通サブタイプ数 × 施設数) の行列を構築し、
confusion_mtx と同じ正規化（_normalize + flip）を全体に適用してから
SFT 行 × 各サブタイプ列のセルを平均する。
棒グラフの値は confusion_mtx のヒートマップセル値と直接対応する。

Usage:
    uv run python sft_distance_bar.py
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity as cos_sim, euclidean_distances

from utils.display import make_abbrev, ordered_subtypes, subtype_color_map
from utils.loader import load_config, load_data


# ── metrics (confusion_mtx.py と同定義) ──────────────────────────────────────

def _energy_distance(X: np.ndarray, Y: np.ndarray) -> float:
    return (2 * euclidean_distances(X, Y).mean()
            - euclidean_distances(X, X).mean()
            - euclidean_distances(Y, Y).mean())


# slug → (fn(X, Y), is_similarity, flip)
# flip=True: confusion_mtx と同様に display = 1 − normalize(dist)
METRICS: dict[str, tuple] = {
    "euc_mean": (lambda X, Y: np.linalg.norm(X.mean(axis=0) - Y.mean(axis=0)), False, True),
    "cos_mean": (lambda X, Y: cos_sim(X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True))[0, 0], True, False),
    "euc_all":  (lambda X, Y: euclidean_distances(X, Y).mean(), False, True),
    "cos_all":  (lambda X, Y: cos_sim(X, Y).mean(), True, False),
    "energy":   (_energy_distance, False, False),
}


# ── matrix (confusion_mtx.py と同一) ─────────────────────────────────────────

def _normalize(v: np.ndarray) -> np.ndarray:
    vmin, vmax = v.min(), v.max()
    return (v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v)


def build_matrix(groups: dict, compute_fn, is_similarity: bool) -> pd.DataFrame:
    labels = list(groups.keys())
    n = len(labels)
    mat = pd.DataFrame(np.full((n, n), 1.0 if is_similarity else 0.0),
                       index=labels, columns=labels)
    for l1, l2 in combinations(labels, 2):
        val = float(compute_fn(groups[l1], groups[l2]))
        mat.loc[l1, l2] = val
        mat.loc[l2, l1] = val
    return mat


# ── compute ───────────────────────────────────────────────────────────────────

def compute_sft_bar_values(
    merged: pd.DataFrame,
    cfg: dict,
    metric_slug: str,
) -> tuple[dict[str, float], dict[str, str]]:
    """cross 行列を全体で正規化（confusion_mtx と同一）してから SFT 行を平均する。

    Returns:
        bar_values: {subtype: ヒートマップセル値の平均}  confusion_mtx の表示値と直接対応
        color_map:  {subtype: hex color}
    """
    if metric_slug not in METRICS:
        raise ValueError(f"Unknown metric '{metric_slug}'. Choose from: {list(METRICS)}")
    compute_fn, is_similarity, flip = METRICS[metric_slug]

    # confusion_mtx.py の cross セクションと同一ロジック
    sources = sorted(merged["source"].unique())
    common = ordered_subtypes(
        set.intersection(*[set(merged.loc[merged["source"] == s, "subtype"]) for s in sources]),
        cfg,
    )
    print(f"  common subtypes ({len(common)}): {common}")

    raw_labels = [f"{src}__{sub}" for sub in common for src in sources]
    abbrev = make_abbrev(raw_labels, cfg)
    vecs = {
        abbrev[f"{src}__{sub}"]: np.stack(
            merged.loc[(merged["source"] == src) & (merged["subtype"] == sub), "embedding"].values
        )
        for sub in common for src in sources
    }

    mat = build_matrix(vecs, compute_fn, is_similarity)

    # confusion_mtx.py と同一: 全体で正規化 → flip 適用
    norm_v = _normalize(mat.values.astype(float))
    display_v = (1.0 - norm_v) if flip else norm_v
    display_mat = pd.DataFrame(display_v, index=mat.index, columns=mat.columns)

    # SFT 先頭 2 行ラベル（common の先頭が SFT）
    sft_sub = next((s for s in common if "sft" in s.lower()), None)
    if sft_sub is None:
        raise ValueError("SFT not found in common subtypes")
    sft_labels = [abbrev[f"{src}__{sft_sub}"] for src in sources]

    # 全 13 サブタイプについて SFT 行 2 × サブタイプ列 2 の 4 セルを平均
    bar_values: dict[str, float] = {}
    for sub in common:
        sub_labels = [abbrev[f"{src}__{sub}"] for src in sources]
        cells = [display_mat.loc[sft_lbl, sub_lbl]
                 for sft_lbl in sft_labels for sub_lbl in sub_labels]
        bar_values[sub] = float(np.mean(cells))
    other_subtypes = [s for s in common if "sft" not in s.lower()]
    color_map = subtype_color_map(other_subtypes, cfg)
    return {sub: bar_values[sub] for sub in other_subtypes}, color_map


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_bar(
    norm_values: dict[str, float],
    color_map: dict[str, str],
    cfg: dict,
    out_path: Path,
) -> None:
    shortened = cfg.get("display", {}).get("shortened", {}).get("subtypes", {})

    items = sorted(norm_values.items(), key=lambda x: x[1], reverse=True)
    labels_full = [it[0] for it in items]
    values = [it[1] for it in items]
    colors = [color_map.get(lb, "#888888") for lb in labels_full]
    labels = [shortened.get(lb, lb) for lb in labels_full]

    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.55 + 0.8)))
    bars = ax.barh(range(len(labels)), values, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=13)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center", ha="left", fontsize=11,
        )

    ax.set_xlim(0, 1.18)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    sft_cfg = cfg.get("sft_distance", {})
    variant = sft_cfg.get("variant", "centroid")
    metric_slug = sft_cfg.get("metric", "euc_mean")
    dir_key = cfg.get("variants", {}).get(variant, variant)
    out_dir = Path(cfg["output_dir"]) / "sft_distance"

    print(f"[sft_distance_bar] variant={variant}  metric={metric_slug}")
    merged = load_data(cfg, dir_key)
    if merged.empty:
        print("  SKIP: no data")
        return
    print(f"  {len(merged)} cases, sources: {sorted(merged['source'].unique())}")

    bar_values, color_map = compute_sft_bar_values(merged, cfg, metric_slug)
    print(f"  {len(bar_values)} subtypes computed")

    out_path = out_dir / f"sft_distance_bar_{variant}_{metric_slug}.png"
    plot_bar(bar_values, color_map, cfg, out_path)


if __name__ == "__main__":
    main()
