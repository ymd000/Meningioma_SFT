"""centroid.py — 重心補正（centroid shift）の双方向適用・可視化・評価。

補正済み埋め込みを HDF5 に書き込む（lp.py 等が参照する）。

出力:
    pca_source.png    — 補正前・2方向補正後の PCA 3パネル
    shift_norm.png    — 双方向シフトノルム棒グラフ
    asw_batch_bio.png — ASW-batch / ASW-bio の 3状態比較
    HDF5              — embedding.{src}.centroid/ に書き込み

Usage:
    uv run python centroid.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── config / data ─────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> pd.DataFrame:
    frames = []
    for ds_name, ds_cfg in cfg["embedding"].items():
        h5_key = f"{ds_cfg['tile_model']}/{cfg['keys']['slide_feature']}"
        emb_dir = Path(ds_cfg["original"])
        subtype_csv = Path(cfg["label"][ds_name]["subtype"])
        df_sub = pd.read_csv(subtype_csv)
        if "case_id" not in df_sub.columns:
            raise ValueError(f"[{ds_name}] 'case_id' column not found in {subtype_csv}")
        col_candidates = cfg.get("subtype_cols", ["subtype"])
        col = next((c for c in col_candidates if c in df_sub.columns), None)
        if col is None:
            raise ValueError(f"[{ds_name}] subtype column not found in {subtype_csv}")
        df_sub = df_sub.rename(columns={col: "subtype"})[["case_id", "subtype"]]
        records = []
        for h5_path in sorted(emb_dir.glob("*.h5")):
            with h5py.File(h5_path, "r") as f:
                if h5_key not in f:
                    continue
                records.append({"case_id": h5_path.stem, "embedding": f[h5_key][:]})
        if not records:
            print(f"  WARN [{ds_name}]: no embeddings under '{h5_key}'")
            continue
        df = pd.DataFrame(records).merge(df_sub, on="case_id")
        df["source"] = ds_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── alignment ─────────────────────────────────────────────────────────────────

def compute_all_shifts(merged: pd.DataFrame) -> dict[str, np.ndarray]:
    """各ソースを他ソース空間へシフトするベクトルを双方向で計算する。

    shifts[src] = mean_other - mean_src
    ノルムは両方向で等しくなる。
    """
    sources = sorted(merged["source"].unique())
    means = {src: np.stack(merged.loc[merged["source"] == src, "embedding"].values).mean(0)
             for src in sources}
    shifts: dict[str, np.ndarray] = {}
    for src in sources:
        other = next(s for s in sources if s != src)
        shifts[src] = means[other] - means[src]
        print(f"  [{src} -> {other}] shift norm: {np.linalg.norm(shifts[src]):.4f}")
    return shifts


def _apply_shift(merged: pd.DataFrame, src: str, shift: np.ndarray) -> pd.DataFrame:
    """merged の src 行だけシフトしたコピーを返す。"""
    out = merged.copy()
    mask = out["source"] == src
    out.loc[mask, "embedding"] = out.loc[mask, "embedding"].apply(lambda v, s=shift: v + s)
    return out


# ── HDF5 書き込み ─────────────────────────────────────────────────────────────

def write_centroid_embeddings(shifts: dict[str, np.ndarray], cfg: dict) -> None:
    """補正済み埋め込みを centroid dir の HDF5 に書き込む。"""
    overwrite = cfg.get("centroid", {}).get("overwrite", False)
    sources = list(cfg["embedding"].keys())

    for src, shift in shifts.items():
        h5_key = f"{cfg['embedding'][src]['tile_model']}/{cfg['keys']['slide_feature']}"
        other = next(s for s in sources if s != src)
        raw_dir  = Path(cfg["embedding"][src]["original"])
        out_dir  = Path(cfg["embedding"][src]["centroid"])
        out_dir.mkdir(parents=True, exist_ok=True)

        n_written = n_skipped = 0
        for h5_path in sorted(raw_dir.glob("*.h5")):
            out_path = out_dir / h5_path.name
            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue
            with h5py.File(h5_path, "r") as f:
                if h5_key not in f:
                    continue
                emb = f[h5_key][:]
            with h5py.File(out_path, "w") as f:
                f.create_dataset(h5_key, data=emb + shift)
            n_written += 1

        print(f"  [{src} -> {other}] written={n_written}  skipped={n_skipped}  -> {out_dir}")


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_pca2(df: pd.DataFrame, pre_n: int | None) -> tuple[np.ndarray, PCA]:
    X = np.stack(df["embedding"].values)
    if pre_n and X.shape[1] > pre_n:
        X = PCA(n_components=pre_n, random_state=42).fit_transform(X)
    pca2 = PCA(n_components=2, random_state=42)
    return pca2.fit_transform(X), pca2


def _reduce(df: pd.DataFrame, n: int | None) -> np.ndarray:
    X = np.stack(df["embedding"].values)
    if n and X.shape[1] > n:
        return PCA(n_components=n, random_state=42).fit_transform(X)
    return X


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_pca(merged: pd.DataFrame, states: list[tuple[str, pd.DataFrame]],
             out_dir: Path, cfg: dict) -> None:
    """before + 2方向補正後の 3パネル PCA。

    states: [("before", df_raw), ("after A→B", df), ("after B→A", df)]
    """
    pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
    src_colors: dict = cfg.get("display", {}).get("colors", {}).get("sources", {})
    sources = sorted(merged["source"].unique())

    fig, axes = plt.subplots(1, len(states), figsize=(7 * len(states), 6))
    for ax, (title, df) in zip(axes, states):
        coords, pca2 = _to_pca2(df, pre_n)
        for src in sources:
            mask = df["source"].values == src
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       color=src_colors.get(src, "gray"),
                       alpha=0.6, s=50, label=src, edgecolors="none")
        var = pca2.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=9)
        ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("PCA — colored by source", fontsize=13)
    plt.tight_layout()
    path = out_dir / "pca_source.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


def plot_shift_stats(shifts: dict[str, np.ndarray], sources: list[str],
                     out_dir: Path) -> None:
    """双方向シフトノルム棒グラフ。ノルムは両方向で等しい。"""
    src_a, src_b = sources
    norm = float(np.linalg.norm(shifts[src_a]))  # both directions equal
    labels = [f"{src_a}→{src_b}", f"{src_b}→{src_a}"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, [norm, norm], color=["#4477AA", "#EE6677"],
                  edgecolor="white", width=0.4)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + norm * 0.01,
                f"{norm:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylabel("L2 norm of shift vector", fontsize=11)
    ax.set_title("Centroid shift magnitude (bidirectional)", fontsize=12)
    ax.set_ylim(0, norm * 1.3)
    plt.tight_layout()
    path = out_dir / "shift_norm.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


def plot_asw(states: list[tuple[str, pd.DataFrame]], out_dir: Path, cfg: dict) -> None:
    """3状態の ASW-batch / ASW-bio 比較。"""
    pre_n = cfg.get("centroid", {}).get("pca_n_components", 50)
    records = []
    for label, df in states:
        X = _reduce(df, pre_n)
        sample = min(2000, len(X))
        src_labels = df["source"].values
        sub_labels = df["subtype"].values
        asw_batch = (silhouette_score(X, src_labels, sample_size=sample, random_state=42)
                     if len(set(src_labels)) >= 2 else float("nan"))
        asw_bio   = (silhouette_score(X, sub_labels, sample_size=sample, random_state=42)
                     if len(set(sub_labels)) >= 2 else float("nan"))
        records.append({"phase": label, "asw_batch": asw_batch, "asw_bio": asw_bio})
        print(f"  [{label}] ASW-batch={asw_batch:.4f}  ASW-bio={asw_bio:.4f}")

    df_r = pd.DataFrame(records)
    phases = df_r["phase"].tolist()
    colors = ["#aaaaaa", "#4477AA", "#EE6677"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, note in zip(axes,
                              ["asw_batch", "asw_bio"],
                              ["→ 0 is best (sources mixed)",
                               "→ 1 is best (subtypes separated)"]):
        vals = df_r[col].values
        bars = ax.bar(phases, vals, color=colors[:len(phases)],
                      edgecolor="white", width=0.5)
        for bar, v in zip(bars, vals):
            offset = max(abs(vals)) * 0.02 if not np.isnan(vals).all() else 0.02
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + (offset if v >= 0 else -offset),
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{col}\n{note}", fontsize=10)
        ax.set_ylabel("Silhouette score", fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        lim = max(abs(vals)) * 1.4 if not np.isnan(vals).all() else 1.0
        ax.set_ylim(-lim, lim)
    plt.suptitle("Batch effect metrics — 3 states", fontsize=12)
    plt.tight_layout()
    path = out_dir / "asw_batch_bio.png"
    plt.savefig(path, dpi=150); plt.close()
    print(f"  saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    out_dir = Path(cfg["output_dir"]) / "centroid"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    merged = load_data(cfg)
    print(f"  {len(merged)} cases")

    sources = sorted(merged["source"].unique())
    src_a, src_b = sources

    print("\n[compute shifts]")
    shifts = compute_all_shifts(merged)

    # 3状態の DataFrame を in-memory で作成（可視化用）
    # state_a: src_a が src_b 空間にシフト、src_b は raw
    merged_a = _apply_shift(merged, src_a, shifts[src_a])
    # state_b: src_b が src_a 空間にシフト、src_a は raw
    merged_b = _apply_shift(merged, src_b, shifts[src_b])

    states = [
        ("before (raw)", merged),
        (f"after ({src_a}→{src_b} space)", merged_a),
        (f"after ({src_b}→{src_a} space)", merged_b),
    ]

    print("\n[PCA visualization]")
    plot_pca(merged, states, out_dir, cfg)

    print("\n[shift stats]")
    plot_shift_stats(shifts, sources, out_dir)

    print("\n[ASW metrics]")
    plot_asw(states, out_dir, cfg)

    print("\n[write centroid embeddings]")
    write_centroid_embeddings(shifts, cfg)


if __name__ == "__main__":
    main()
