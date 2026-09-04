# Meningioma_SFT

MeningiomaとSolitary Fibrous Tumor (SFT) を題材にした埋め込み解析・可視化スクリプト群。

## 構成

```
config.yaml          # 全設定（パス・色・パラメータ）
gan.py               # Cycle-GAN による補正後のバッチをエンコードして HDF5 書き込みするラッパー
titan.py             # TITAN による WSI-level aggregate
centroid.py          # 重心補正（centroid shift）の可視化・評価
lp.py                # TITAN 埋め込みの Linear Probe 訓練・評価
umap_plot.py         # 補正前後の UMAP プロット
subtype.py           # サブタイプ別平均ベクトルの UMAP
confusion_mtx.py     # 施設間距離行列（補正前後）
dendrogram.py        # 施設間距離の樹形図（補正前後）
dataset.py           # データセット構成の円グラフ
sft_distance_bar.py  # SFT と各サブタイプ間の距離棒グラフ
utils/
  loader.py          # load_config / load_data（各スクリプトが共通利用）
  display.py         # 色・順序・略称ヘルパー（ordered_subtypes / subtype_color_map / make_abbrev 等）
```

## 環境構築

```bash
# 依存パッケージのインストール（wsi-toolbox を GitHub から取得）
uv sync

# config.yaml は .gitignore 対象。テンプレートをコピーしてパスなどを設定する
cp config.example.yaml config.yaml
```

## フロー

### 前提

- **施設数は 2 つであること**（`site_a` / `site_b`）。3 施設以上には対応していない。
- 以下が済んでいること。

```
NDPIファイル
  ├─ [wsi_gan] ─────────────→ HDF5 key: cache/{size}/gan/patches   (GAN補正済みパッチ)
  └─ [wsi-toolbox] ─────────→ HDF5 key: {tile_model}/features      (original patch emb)
```
[wsi_gan]: https://github.com/S-murakami1/wsi_gan
[wsi-toolbox]: https://github.com/technoplasm/wsi-toolbox

- CSV と HDF5 は `config.yaml` で指定したパスに置くこと。

```yaml
label:
  site_a:
    label:   path/to/site_a/case_labels.csv
    subtype: path/to/site_a/case_subtypes.csv
  site_b:
    label:   path/to/site_b/case_labels.csv
    subtype: path/to/site_b/case_subtypes.csv
embedding:
  site_a:
    original:   path/to/site_a/embedding   # HDF5 ファイル群のディレクトリ
    tile_model: conch15_768                # wsi-toolbox のモデル名（HDF5 キーのプレフィックス）
  site_b:
    original:   path/to/site_b/embedding
    tile_model: conch15_768
```
- HDF5 内のキー構造が `config.yaml` 通りに存在すること。 キー構造は `config.yaml` の `keys:` セクションで管理する。

| config キー | 完全キーのテンプレート | 用途 |
|---|---|---|
| `keys.gan_patches` | `cache/{gan.patch_size}/{key}` | `gan.py` が読む GAN補正済みパッチ |
| `keys.gan_coords` | `cache/{gan.patch_size}/{key}` | 同パッチの座標（フォールバックは `tile_coords`）|
| `keys.tile_features` | `{tile_model}/{key}` | `gan.py` が書く・`titan.py` が読む patch embedding |
| `keys.tile_coords` | `{tile_model}/{key}` | 同座標 |
| `keys.slide_feature` | `{tile_model}/{key}` | `titan.py` が書く・解析スクリプトが読む slide embedding |

`tile_model` を変更する場合は `embedding.*.tile_model` を更新するだけでよい。`keys:` の値を変える必要はない。

### 概略図

```
┌──────────────────────────────────┐
│  dataset.py                      │
│  subtype CSV → 円グラフ          │
└──────────────────────────────────┘

HDF5: cache/{size}/gan/patches
      │
      ▼
┌──────────────────────────────────────────────────────────┐
│  gan.py  — GAN補正パッチを tile model でエンコード       │
│  → {tile_model}/features  (GAN patch-level embedding)    │
└──────────────────────────────────────────────────────────┘
      │  GAN patch emb            original patch emb
      │                           (wsi-toolbox 済み)
      └───────────────┬───────────────────┘
                      ▼
┌──────────────────────────────────────────────────────────┐
│  titan.py  — TITAN で slide-level にアグリゲート         │
│  {tile_model}/features → {tile_model}/aggregates/titan   │
│  variants: original / gan                                │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼  slide-level embedding (HDF5)
┌──────────────────────────────────────────────────────────┐
│  centroid.py  — 施設間重心補正                           │
│  → 補正済み HDF5 を centroid/ に書き出し                 │
│  → PCA 3パネル / シフトノルム棒グラフ / ASW 評価         │
└──────────────────────────────────────────────────────────┘
                      │
           ┌──────────┴──────────┐
           │ before (original)   │ after (centroid)
           ▼                     ▼
┌──────────────────────────────────────────────────────────┐
│  lp.py  — Linear Probe 交差データセット評価              │
│  施設 A で訓練 → 施設 B でテスト（補正前後を比較）       │
│  → confusion matrix / metrics.csv / comparison.csv       │
└──────────────────────────────────────────────────────────┘
                      │
                      ▼  
──── slide-level embedding を使った各種解析 ────────────────
  umap_plot.py          補正前後の UMAP プロット
  subtype.py            サブタイプ別平均ベクトルの UMAP
  confusion_mtx.py      施設間距離行列（補正前後）
  dendrogram.py         施設間距離の樹形図（補正前後）
  sft_distance_bar.py   SFT と各サブタイプ間の距離棒グラフ
```

## 原則

- **1 figure = 1 Python file**。共通処理はファイル間で共有せず、各スクリプトに直接書く
- 設定（パス・色・パラメータ）はすべて `config.yaml` に集約し、スクリプト内にハードコードしない
- `argparse` は使わない。試行錯誤は `config.yaml` の編集で行う
