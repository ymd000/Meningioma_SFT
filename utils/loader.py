"""loader.py — config / embedding データの読み込み共通処理。"""
from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_data(cfg: dict, dir_key: str) -> pd.DataFrame:
    """slide-level 埋め込みとサブタイプラベルを結合して返す。

    dir_key: config の embedding.*.{dir_key} を参照する。
    reference が設定されている場合、reference 側は dir_key によらず original を使う。
    """
    reference: str | None = cfg.get("reference")
    frames = []
    for ds_name, ds_cfg in cfg["embedding"].items():
        h5_key = f"{ds_cfg['tile_model']}/{cfg['keys']['slide_feature']}"
        effective_key = "original" if (reference == ds_name and dir_key != "original") else dir_key
        if effective_key not in ds_cfg:
            continue
        emb_dir = Path(ds_cfg[effective_key])
        if not emb_dir.exists():
            print(f"  WARN [{ds_name}]: dir not found: {emb_dir}")
            continue
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
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
