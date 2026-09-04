"""dataset.py — SFT/Meningioma 症例数円グラフ。

Usage:
    uv run python dataset.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_counts(cfg: dict) -> dict[str, pd.Series]:
    col_candidates = cfg.get("subtype_cols", ["subtype"])
    result = {}
    for ds_name, ds_cfg in cfg["label"].items():
        df = pd.read_csv(ds_cfg["subtype"])
        col = next((c for c in col_candidates if c in df.columns), None)
        if col is None:
            raise ValueError(f"[{ds_name}] subtype column not found in {ds_cfg['subtype']}")
        result[ds_name] = df[col].value_counts()
    return result


def build_piecharts(cfg: dict) -> None:
    shortened: dict[str, str] = cfg["display"]["shortened"]["subtypes"]
    colors_map: dict[str, str] = cfg["display"]["colors"]["subtypes"]
    sources: list[str] = list(cfg["label"].keys())

    counts = load_counts(cfg)
    out_dir = Path(cfg["output_dir"]) / "dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in sources:
        series = counts[ds]
        st_list = [st for st in shortened if series.get(st, 0) > 0]
        values  = [int(series[st]) for st in st_list]
        colors  = [colors_map.get(st, "#aaaaaa") for st in st_list]
        abbrs   = [shortened[st] for st in st_list]
        total   = sum(values)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(values, colors=colors, startangle=90)
        ax.set_aspect("equal")
        fig.tight_layout(pad=0)

        out = out_dir / f"pie_{ds}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
        plt.close(fig)
        print(f"保存先: {out}  (n={total})")

    # 凡例: 全 source に登場するサブタイプをまとめて別ファイル出力
    all_st = [st for st in shortened if any(counts[ds].get(st, 0) > 0 for ds in sources)]
    patches = [
        plt.Rectangle((0, 0), 1, 1, fc=colors_map.get(st, "#aaaaaa"))
        for st in all_st
    ]
    legend_texts = [f"{shortened[st]}  {st}" for st in all_st]

    fig, ax = plt.subplots(figsize=(4, len(all_st) * 0.28 + 0.3))
    ax.set_axis_off()
    ax.legend(patches, legend_texts, loc="center", fontsize=8, frameon=False,
              handlelength=1.2, handleheight=1.0, borderpad=0)
    fig.tight_layout(pad=0.2)

    out = out_dir / "legend.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"保存先: {out}")


if __name__ == "__main__":
    cfg = load_config()
    build_piecharts(cfg)
