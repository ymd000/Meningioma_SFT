"""gan.py — GAN補正済みパッチをエンコードし HDF5 に書き込む。

入力: source HDF5 の cache/{patch_size}/gan/patches
出力: {gan_dir}/{stem}.h5 の {tile_model}/features + {tile_model}/coordinates

Usage:
    uv run python gan.py
"""
from __future__ import annotations

import gc
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _encode_patches(
    patches: np.ndarray,
    model: torch.nn.Module,
    extract_fn,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """patches (N, H, W, 3) uint8 -> features (N, D) float32"""
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    use_autocast = device_type == "cuda"
    autocast_dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if use_autocast else torch.float32

    all_features: list[np.ndarray] = []
    n = len(patches)
    n_batches = (n + batch_size - 1) // batch_size

    for i in range(0, n, batch_size):
        batch = patches[i : i + batch_size]
        x = (torch.from_numpy(batch).float() / 255.0).permute(0, 3, 1, 2)
        x = x.to(device, memory_format=torch.channels_last)
        x = (x - mean) / std

        with torch.inference_mode(), torch.autocast(device_type=device_type, dtype=autocast_dtype, enabled=use_autocast):
            if extract_fn is not None:
                feats = extract_fn(model, x).cpu().numpy()
            else:
                h = model.forward_features(x)
                feats = h[:, 0, ...].cpu().numpy()

        all_features.append(feats)
        print(f"    batch {i // batch_size + 1}/{n_batches}", end="\r", flush=True)

    print()
    return np.concatenate(all_features, axis=0)


def _load_model(tile_model: str, device: str):
    from wsi_toolbox.common import create_default_model, get_config, set_default_preset

    set_default_preset(tile_model)
    wt_cfg = get_config()
    model = create_default_model().to(device, memory_format=torch.channels_last).eval()
    mean = torch.tensor(wt_cfg.norm_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(wt_cfg.norm_std).view(1, 3, 1, 1).to(device)
    return model, wt_cfg.extract_fn, mean, std


def run(cfg: dict) -> None:
    gan_cfg = cfg.get("gan", {})
    patch_size: int = gan_cfg.get("patch_size", 512)
    batch_size: int = gan_cfg.get("batch_size", 512)
    overwrite: bool = gan_cfg.get("overwrite", False)

    device_spec: str = gan_cfg.get("device", "auto")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device_spec == "auto" else device_spec
    print(f"device: {device}")

    keys = cfg["keys"]
    patches_key = f"cache/{patch_size}/{keys['gan_patches']}"
    coords_candidates = [
        f"cache/{patch_size}/{keys['gan_coords']}",
        f"cache/{patch_size}/{keys['tile_coords']}",
    ]

    loaded_model: str | None = None
    model = extract_fn = mean = std = None

    for ds_name, ds_cfg in cfg["embedding"].items():
        if "gan" not in ds_cfg:
            print(f"\n[{ds_name}] SKIP: 'gan' path not configured")
            continue

        raw_dir = Path(ds_cfg["original"])
        out_dir = Path(ds_cfg["gan"])
        out_dir.mkdir(parents=True, exist_ok=True)
        tile_model: str = ds_cfg["tile_model"]

        # load/reload model only when tile_model changes
        if loaded_model != tile_model:
            if model is not None:
                del model, mean, std
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"  loading model: {tile_model}")
            model, extract_fn, mean, std = _load_model(tile_model, device)
            loaded_model = tile_model

        feature_key = f"{tile_model}/{keys['tile_features']}"
        coord_key = f"{tile_model}/{keys['tile_coords']}"

        h5_files = sorted(raw_dir.glob("*.h5"))
        print(f"\n[{ds_name}] {len(h5_files)} files  ->  {out_dir}")

        n_written = n_skipped = n_missing = 0
        for h5_path in h5_files:
            out_path = out_dir / h5_path.name

            if out_path.exists() and not overwrite:
                n_skipped += 1
                continue

            with h5py.File(h5_path, "r") as f:
                if patches_key not in f:
                    print(f"  SKIP {h5_path.name}: '{patches_key}' not found")
                    n_missing += 1
                    continue
                patches = f[patches_key][:]

                coords: np.ndarray | None = None
                for ck in coords_candidates:
                    if ck in f:
                        coords = f[ck][:]
                        break
                if coords is None:
                    coords = np.zeros((len(patches), 2), dtype=np.int64)

            print(f"  {h5_path.name}: {len(patches)} patches")
            features = _encode_patches(patches, model, extract_fn, mean, std, device, batch_size)

            with h5py.File(out_path, "w") as f:
                grp = f.require_group(tile_model)
                grp.attrs["patch_size"] = patch_size
                grp.attrs["preset"] = tile_model
                f.create_dataset(feature_key, data=features)
                f.create_dataset(coord_key, data=coords)

            print(f"    -> {out_path.name}  shape={features.shape}")
            n_written += 1

        print(f"  written={n_written}  skipped={n_skipped}  missing={n_missing}")

    if model is not None:
        del model, mean, std
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    cfg = load_config()
    run(cfg)


if __name__ == "__main__":
    main()
