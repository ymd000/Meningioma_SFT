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

### 実行順

```bash
uv run python gan.py       # GAN 補正パッチを patch-level embedding に変換して gan/ に書き出し
uv run python titan.py     # patch-level を slide-level に集約（original / gan の 2 variant）
uv run python centroid.py  # slide-level を施設間で重心補正し centroid variant HDF5 を生成

# 以下は順不同（可視化・評価）
uv run python lp.py
uv run python umap_plot.py
uv run python subtype.py
uv run python confusion_mtx.py
uv run python dendrogram.py
uv run python sft_distance_bar.py
uv run python dataset.py   # 他スクリプトと独立。任意のタイミングで実行可
```


### 前提

- **施設数は 2 つであること**（`site_a` / `site_b` は `config.example.yaml` のプレースホルダ名。実際の施設名に置き換えて使う）。3 施設以上には対応していない。
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
    centroid:   path/to/centroid/site_a    # centroid.py が書き出す先
    gan:        path/to/gan/site_a         # gan.py が書き出す先
    tile_model: conch15_768                # wsi-toolbox のモデル名（HDF5 キーのプレフィックス）
  site_b:
    original:   path/to/site_b/embedding
    centroid:   path/to/centroid/site_b
    gan:        path/to/gan/site_b
    tile_model: conch15_768
```
- HDF5 ファイルは **variant ごとに別ディレクトリ**（`embedding.*.original` / `.gan` / `.centroid`）に配置する。3 variant は同じキー `{tile_model}/aggregates/titan/feature` に slide embedding を書くため、同一 HDF5 に共存させられない。

```
original/{case}.h5     ← 前処理ツール ([wsi_gan] / [wsi-toolbox]) の出力
├── cache/{gan.patch_size}/
│   ├── gan/patches       [wsi_gan]      → gan.py が読む
│   └── gan/coordinates   [wsi_gan]      → gan.py が読む
└── {tile_model}/
    ├── features          [wsi-toolbox]  → titan.py が読む
    ├── coordinates       [wsi-toolbox]  → titan.py が読む
    └── aggregates/titan/feature         ← titan.py が書く      → 解析が読む

gan/{case}.h5          ← gan.py + titan.py の出力
└── {tile_model}/
    ├── features                         ← gan.py が書く        → titan.py が読む
    ├── coordinates                      ← gan.py が書く        → titan.py が読む
    └── aggregates/titan/feature         ← titan.py が書く      → 解析が読む

centroid/{case}.h5     ← centroid.py の出力
└── {tile_model}/
    └── aggregates/titan/feature         ← centroid.py が書く   → 解析が読む
```

- `{gan.patch_size}` / `{tile_model}` は config の値に展開される（例: `cache/512/gan/patches`, `conch15_768/features`）

## 概略図

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
│  → 補正済み slide embedding を centroid variant HDF5 に  │
│    書き出し（配置は embedding.*.centroid で指定）        │
│  → PCA 3パネル / シフトノルム棒グラフ / ASW 評価         │
└──────────────────────────────────────────────────────────┘
                      │  (original / centroid / gan)
                      ▼
┌──────────────────────────────────────────────────────────┐
│  lp.py  — Linear Probe 交差データセット評価              │
│  施設 A→B と B→A の双方向 × 3 variant (original /        │
│  centroid / gan) を比較                                  │
│  → confusion_matrix{,_norm}.png / metrics.csv /          │
│    comparison.csv / checkpoints/ / logs/                 │
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
