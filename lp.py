"""lp.py — TITAN スライド埋め込みに対する Linear Probe の交差データセット評価。

centroid.py で生成した補正済み HDF5 を使い、補正前後を比較する。

訓練: 片方のデータセット全体（raw）で学習
テスト: もう片方の raw（before）または centroid（after）で評価

  centroid.py を先に実行して centroid HDF5 を生成しておくこと。

出力 (output_dir/lp/):
    trained_by_{src}/before/  — raw でテスト
    trained_by_{src}/after/   — centroid でテスト
      checkpoints/last.ckpt
      logs/
      confusion_matrix.png / confusion_matrix_norm.png
      metrics.csv
    comparison.csv            — 4条件の横断比較

Usage:
    uv run python lp.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import h5py
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ── model ─────────────────────────────────────────────────────────────────────

class LinearProbeModel(L.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        lr: float = 1e-3,
        dropout: float = 0.0,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        layers: list[torch.nn.Module] = []
        if dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(embedding_dim, num_classes))
        self.classifier = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict:
        return {"logits": self.classifier(x), "attention": None}

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc",  acc,  on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self(x)
        loss = torch.nn.functional.cross_entropy(out["logits"], y)
        acc = (out["logits"].argmax(-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc",  acc,  prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


# ── config ────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ── dataset ───────────────────────────────────────────────────────────────────

class TitanDataset(Dataset):
    """TITAN スライド埋め込みデータセット（wsi-toolbox HDF5 形式）。"""

    def __init__(self, emb_dir: str | Path, h5_key: str, label_csv: str | Path):
        emb_dir = Path(emb_dir)
        label_dict: dict[str, int] = {}
        with open(label_csv) as f:
            for row in csv.DictReader(f):
                label_dict[row["case_id"]] = int(row["label"])

        self.embeddings: list[np.ndarray] = []
        self.labels: list[int] = []
        self.case_ids: list[str] = []

        for h5_path in sorted(emb_dir.glob("*.h5")):
            case_id = h5_path.stem
            if case_id not in label_dict:
                continue
            with h5py.File(h5_path, "r") as f:
                if h5_key not in f:
                    continue
                self.embeddings.append(f[h5_key][:])
            self.labels.append(label_dict[case_id])
            self.case_ids.append(case_id)

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return torch.from_numpy(self.embeddings[idx]).float(), self.labels[idx]


def _collate(batch: list[tuple[torch.Tensor, int]]):
    embs, labels = zip(*batch)
    return torch.stack(embs), torch.tensor(labels, dtype=torch.long)


# ── training ──────────────────────────────────────────────────────────────────

def _make_split(dataset: TitanDataset, val_fraction: float) -> tuple[list[int], list[int]]:
    indices = list(range(len(dataset)))
    return train_test_split(
        indices,
        test_size=val_fraction,
        stratify=dataset.labels,
        random_state=42,
    )


def _save_split(dataset: TitanDataset, train_idx: list[int], val_idx: list[int],
                path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "label", "split"])
        for i in train_idx:
            w.writerow([dataset.case_ids[i], dataset.labels[i], "train"])
        for i in val_idx:
            w.writerow([dataset.case_ids[i], dataset.labels[i], "val"])
    print(f"  saved: {path}")


def _remap_split(dataset: TitanDataset, split_csv: Path) -> tuple[list[int], list[int]]:
    """split.csv の case_id を基準に dataset のインデックスを再計算する。

    base_ds と dataset のファイル並びが異なっても正しい症例が train/val に入ることを保証する。
    split.csv に載っていない case_id（variant dir にファイルが存在しない場合）は除外する。
    """
    train_ids: set[str] = set()
    val_ids: set[str] = set()
    with open(split_csv) as f:
        for row in csv.DictReader(f):
            (train_ids if row["split"] == "train" else val_ids).add(row["case_id"])
    train_idx = [i for i, cid in enumerate(dataset.case_ids) if cid in train_ids]
    val_idx   = [i for i, cid in enumerate(dataset.case_ids) if cid in val_ids]
    return train_idx, val_idx


def train_all(dataset: TitanDataset, train_idx: list[int], val_idx: list[int],
              cfg: dict, out_dir: Path) -> str:
    """hold-out 検証付きで Linear Probe を訓練し、best.ckpt のパスを返す。"""
    lp_cfg = cfg["lp"]
    torch.set_float32_matmul_precision("high")

    train_ds, val_ds = Subset(dataset, train_idx), Subset(dataset, val_idx)

    embedding_dim = dataset.embeddings[0].shape[-1]
    num_classes   = len(set(dataset.labels))
    num_workers   = lp_cfg.get("num_workers", 0)

    model = LinearProbeModel(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        lr=lp_cfg["lr"],
        dropout=lp_cfg.get("dropout", 0.0),
        weight_decay=lp_cfg.get("weight_decay", 0.0),
    )
    train_loader = DataLoader(train_ds, batch_size=lp_cfg["batch_size"],
                              shuffle=True, num_workers=num_workers, collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=lp_cfg["batch_size"],
                              shuffle=False, num_workers=num_workers, collate_fn=_collate)

    ckpt_cb = ModelCheckpoint(
        dirpath=out_dir / "checkpoints", filename="best",
        monitor="val_loss", mode="min",
        save_top_k=1, enable_version_counter=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val_loss", mode="min",
        patience=lp_cfg.get("patience", 20),
        verbose=False,
    )
    trainer = L.Trainer(
        max_epochs=lp_cfg["max_epochs"],
        accelerator="auto",
        log_every_n_steps=1,
        logger=CSVLogger(save_dir=str(out_dir), name="logs", version=""),
        callbacks=[ckpt_cb, early_stop_cb],
        enable_progress_bar=True,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    ckpt_path = str(ckpt_cb.best_model_path)
    print(f"  checkpoint: {ckpt_path}  (stopped at epoch {trainer.current_epoch})")
    _save_training_history(out_dir / "logs" / "metrics.csv", out_dir / "training_history.png")
    return ckpt_path


def _save_training_history(csv_path: Path, out_path: Path) -> None:
    if not csv_path.exists():
        return
    import pandas as pd
    df = pd.read_csv(csv_path)
    metric_cols = [c for c in df.columns if c not in ("epoch", "step")]
    if not metric_cols:
        return
    epoch_df = (
        df.groupby("epoch")[metric_cols]
        .last()
        .reset_index()
    )
    n = len(metric_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, col in zip(axes[0], metric_cols):
        vals = epoch_df[col].dropna()
        ax.plot(epoch_df["epoch"].iloc[:len(vals)], vals)
        ax.set_xlabel("epoch"); ax.set_ylabel(col); ax.set_title(col)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {out_path}")


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(ckpt_path: str, dataset: TitanDataset, cfg: dict, out_dir: Path) -> dict:
    """チェックポイントをロードして dataset 全件を推論し、結果を保存する。"""
    lp_cfg = cfg["lp"]
    class_names: dict[int, str] | None = lp_cfg.get("class_names")
    device_spec: str = lp_cfg.get("device", "auto")
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu") if device_spec == "auto" else device_spec
    )

    model = LinearProbeModel.load_from_checkpoint(ckpt_path, map_location=device)
    model.eval()

    num_workers = lp_cfg.get("num_workers", 0)
    loader = DataLoader(dataset, batch_size=lp_cfg.get("batch_size", 64),
                        shuffle=False, num_workers=num_workers, collate_fn=_collate)
    all_labels, all_preds = [], []
    with torch.no_grad():
        for embs, labels in loader:
            preds = model(embs.to(device))["logits"].argmax(-1).cpu().tolist()
            all_labels.extend(labels.tolist())
            all_preds.extend(preds)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    acc      = float((all_labels == all_preds).mean())
    bal_acc  = float(balanced_accuracy_score(all_labels, all_preds))
    f1       = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    print(f"  acc={acc:.4f}  bal_acc={bal_acc:.4f}  f1={f1:.4f}  n={len(all_labels)}")

    _save_misclassified(dataset.case_ids, all_labels, all_preds, class_names,
                        out_dir / "misclassified.txt")

    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        cm[t, p] += 1

    labels = ([class_names[i] for i in range(num_classes)]
              if class_names else [str(i) for i in range(num_classes)])
    _save_cm(cm, labels, out_dir / "confusion_matrix.png",      normalize=False)
    _save_cm(cm, labels, out_dir / "confusion_matrix_norm.png", normalize=True)
    _save_cm_colorbar(out_dir / "confusion_matrix_colorbar.png")

    metrics = {"accuracy": acc, "balanced_accuracy": bal_acc, "f1_macro": f1, "n": len(all_labels)}
    with open(out_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in metrics.items():
            w.writerow([k, v])
    print(f"  saved: {out_dir / 'metrics.csv'}")
    return metrics


def _save_misclassified(
    case_ids: list[str],
    labels: np.ndarray,
    preds: np.ndarray,
    class_names: dict[int, str] | None,
    path: Path,
) -> None:
    def name(i: int) -> str:
        return class_names[i] if class_names and i in class_names else str(i)

    mask = labels != preds
    n_wrong = int(mask.sum())
    lines = [f"misclassified: {n_wrong} / {len(labels)}", ""]
    if n_wrong:
        w = max(len(c) for c in case_ids)
        lines.append(f"{'case_id':<{w}}  true -> pred")
        lines.append("-" * (w + 20))
        for i in np.where(mask)[0]:
            lines.append(f"{case_ids[i]:<{w}}  {name(labels[i])} -> {name(preds[i])}")
    path.write_text("\n".join(lines) + "\n")
    print(f"  saved: {path}  ({n_wrong} errors)")


def _save_cm(cm: np.ndarray, labels: list[str], path: Path, normalize: bool) -> None:
    n = cm.shape[0]
    data = (cm / cm.sum(axis=1, keepdims=True).clip(min=1)
            if normalize else cm.astype(float))
    fig, ax = plt.subplots(figsize=(max(5, n * 1.2), max(4, n)))
    im = ax.imshow(data, cmap="Blues", vmin=0, vmax=(1.0 if normalize else None))
    # ax.set_xticks(range(n)); ax.set_yticks(range(n))
    # ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    # ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    thresh = data.max() / 2
    for i in range(n):
        for j in range(n):
            v = data[i, j]
            text = f"{v:.2f}\n(n={cm[i, j]})" if normalize else str(int(cm[i, j]))
            ax.text(j, i, text, ha="center", va="center", fontsize=22,
                    color="white" if v > thresh else "black")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  saved: {path}")


def _save_cm_colorbar(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.5, 8))
    gradient = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(gradient, aspect="auto", cmap="Blues", origin="lower", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([0, 127.5, 255])
    ax.set_yticklabels(["0", "0.5", "1"], fontsize=22)
    ax.yaxis.tick_right()
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight", transparent=True); plt.close()
    print(f"  saved: {path}")


# ── comparison ────────────────────────────────────────────────────────────────

def save_comparison(all_metrics: list[dict], out_dir: Path) -> None:
    path = out_dir / "comparison.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "train_source", "test_source", "accuracy", "balanced_accuracy", "f1_macro", "n"])
        for r in all_metrics:
            w.writerow([r["variant"], r["train_source"], r["test_source"],
                        f"{r['accuracy']:.4f}", f"{r['balanced_accuracy']:.4f}",
                        f"{r['f1_macro']:.4f}", r["n"]])
    print(f"\nsaved: {path}")
    print(f"\n{'variant':10} {'train':12} {'test':12}  acc     bal_acc  f1_macro")
    print("-" * 64)
    for r in all_metrics:
        print(f"{r['variant']:10} {r['train_source']:12} {r['test_source']:12}"
              f"  {r['accuracy']:.4f}  {r['balanced_accuracy']:.4f}  {r['f1_macro']:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────

def _effective_key(src: str, variant: str, reference: str | None) -> str:
    return "original" if (reference and src == reference) else variant


def main() -> None:
    cfg = load_config()
    lp_cfg    = cfg["lp"]
    slide_key = cfg["keys"]["slide_feature"]
    reference = cfg.get("reference")
    sources   = list(cfg["embedding"].keys())
    variants  = lp_cfg.get("variants", ["original"])
    out_root  = Path(cfg["output_dir"]) / "lp"
    out_root.mkdir(parents=True, exist_ok=True)

    L.seed_everything(42, workers=True)
    evaluate_only: bool = lp_cfg.get("evaluate_only", False)
    all_metrics: list[dict] = []

    for train_src in sources:
        test_src = next(s for s in sources if s != train_src)
        train_h5_key = f"{cfg['embedding'][train_src]['tile_model']}/{slide_key}"
        test_h5_key  = f"{cfg['embedding'][test_src]['tile_model']}/{slide_key}"

        # split は original データセットで一度だけ計算・保存（variant によらず共通）
        base_ds = TitanDataset(
            cfg["embedding"][train_src]["original"], train_h5_key,
            cfg["label"][train_src]["label"],
        )
        src_out_dir = out_root / f"trained_by_{train_src}"
        if not evaluate_only:
            train_idx, val_idx = _make_split(base_ds, lp_cfg.get("val_fraction", 0.2))
            _save_split(base_ds, train_idx, val_idx, src_out_dir / "split.csv")
            print(f"\n[trained_by_{train_src}]  train={len(train_idx)}  val={len(val_idx)}")
        else:
            train_idx, val_idx = [], []

        for variant in variants:
            '''
            要はこうなってほしい
              ┌─────┬───────────────────────┬──────────────────┬──────────────────┐
              │  #  │      補正の意味       │      train       │       test       │
              ├─────┼───────────────────────┼──────────────────┼──────────────────┤
              │ 1   │ 補正なし              │ site_a/original │ site_b/original  │
              ├─────┼───────────────────────┼─────────────────┼──────────────────┤
              │ 2   │ test 側のみ centroid  │ site_a/original │ site_b/centroid  │
              ├─────┼───────────────────────┼─────────────────┼──────────────────┤
              │ 3   │ test 側のみ gan       │ site_a/original │ site_b/gan       │
              ├─────┼───────────────────────┼─────────────────┼──────────────────┤
              │ 4   │ 補正なし（逆向き）    │ site_b/original │ site_a/original  │
              ├─────┼───────────────────────┼─────────────────┼──────────────────┤
              │ 5   │ train 側のみ centroid │ site_b/centroid │ site_a/original  │
              ├─────┼───────────────────────┼─────────────────┼──────────────────┤
              │ 6   │ train 側のみ gan      │ site_b/gan      │ site_a/original  │
              └─────┴───────────────────────┴──────────────────┴──────────────────┘
            '''
            train_key = _effective_key(train_src, variant, reference)
            test_key  = _effective_key(test_src,  variant, reference)

            train_emb_dir = cfg["embedding"][train_src].get(train_key)
            test_emb_dir  = cfg["embedding"][test_src].get(test_key)

            if not test_emb_dir or not Path(test_emb_dir).exists():
                print(f"\nSKIP [{test_src}/{test_key}]: dir not found")
                continue

            out_dir = out_root / f"trained_by_{train_src}" / variant
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n=== trained_by_{train_src} / {variant} ===")
            print(f"  train: {train_src}/{train_key}")
            print(f"  test:  {test_src}/{test_key}")

            if evaluate_only:
                ckpt_path = out_dir / "checkpoints" / "best.ckpt"
                if not ckpt_path.exists():
                    print(f"  SKIP: checkpoint not found: {ckpt_path}")
                    continue
                ckpt = str(ckpt_path)
            else:
                if not train_emb_dir or not Path(train_emb_dir).exists():
                    print(f"\nSKIP [{train_src}/{train_key}]: dir not found")
                    continue
                train_ds = TitanDataset(train_emb_dir, train_h5_key, cfg["label"][train_src]["label"])
                tr_idx, vl_idx = _remap_split(train_ds, src_out_dir / "split.csv")
                ckpt = train_all(train_ds, tr_idx, vl_idx, cfg, out_dir)

            test_ds = TitanDataset(test_emb_dir, test_h5_key, cfg["label"][test_src]["label"])
            print(f"  n_test={len(test_ds)}")
            metrics = evaluate(ckpt, test_ds, cfg, out_dir)
            all_metrics.append({
                "variant":      variant,
                "train_source": train_src,
                "test_source":  test_src,
                **metrics,
            })

    if all_metrics:
        save_comparison(all_metrics, out_root)


if __name__ == "__main__":
    main()
