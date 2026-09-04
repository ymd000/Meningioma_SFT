"""titan.py — TITAN スライド埋め込みを一括生成し HDF5 に書き込む。

生成キー: {tile_model}/aggregates/titan/feature
このキーを umap.py / confusion_mtx.py / dendrogram.py が読む。

Usage:
    uv run python titan.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def run(cfg: dict) -> None:
    titan_cfg = cfg["titan"]
    overwrite: bool = titan_cfg.get("overwrite", False)
    variants: list[str] = titan_cfg.get("variants", ["original"])

    device_spec: str = titan_cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_spec == "auto" else device_spec
    print(f"device: {device}")

    from wsi_toolbox.presets.slide.titan import create_titan_model
    model = create_titan_model().to(device).eval()

    keys = cfg["keys"]
    for dir_key in variants:
        for ds_name, ds_cfg in cfg["embedding"].items():
            if dir_key not in ds_cfg:
                continue
            emb_dir = Path(ds_cfg[dir_key])
            if not emb_dir.exists():
                print(f"\n[{dir_key}/{ds_name}] WARN: dir not found: {emb_dir}")
                continue
            tile_model: str = ds_cfg["tile_model"]
            target_key = f"{tile_model}/{keys['slide_feature']}"

            h5_files = sorted(emb_dir.glob("*.h5"))
            print(f"\n[{dir_key}/{ds_name}] {len(h5_files)} files  ->  {target_key}")

            for h5_path in h5_files:
                with h5py.File(h5_path, "r") as f:
                    if target_key in f and not overwrite:
                        print(f"  skip  {h5_path.name}")
                        continue

                    tile_grp = f.get(tile_model)
                    if tile_grp is None or keys["tile_features"] not in tile_grp:
                        print(f"  WARN  {h5_path.name}: '{tile_model}' not found")
                        continue

                    features = tile_grp[keys["tile_features"]][:]
                    coords = tile_grp[keys["tile_coords"]][:]
                    patch_size_lv0 = int(tile_grp.attrs["patch_size"])

                feat_t = torch.from_numpy(features).float().unsqueeze(0).to(device)
                coord_t = torch.from_numpy(coords).long().unsqueeze(0).to(device)

                with torch.inference_mode():
                    emb = model.encode_slide_from_patch_features(feat_t, coord_t, patch_size_lv0)
                emb_np = emb.squeeze(0).cpu().numpy().astype(np.float32)

                with h5py.File(h5_path, "a") as f:
                    if target_key in f:
                        del f[target_key]
                    f.create_dataset(target_key, data=emb_np)

                print(f"  ok    {h5_path.name}  (N={features.shape[0]}, D={emb_np.shape[-1]})")

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
