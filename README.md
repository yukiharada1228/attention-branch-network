# attention-branch-network

Attention Branch Network（ABN）の実装です。`torchvision.datasets.Imagenette`（10クラス）を用いた画像分類に適用し、モデルがどこを見て予測したかを可視化できます。

![Attention Maps](outputs/abn_attentions.png?v=1)

## 概要

このプロジェクトは ABN を ResNet 系バックボーン上に実装し、Imagenette データセットでの学習・評価・可視化を行います。学習には Hugging Face `Trainer` を用い、学習率スケジュールやチェックポイント保存を簡潔に扱えるようにしています。

## 訓練結果

ResNet152 + ABN での Imagenette 10クラス分類の結果:

- **Top-1 Accuracy**: 90.47%
- **Top-5 Accuracy**: 99.21%
- **Validation Loss**: 0.6205
- **Training Epochs**: 90 epochs

## 学習済みモデル

このプロジェクトで学習したモデルがHugging Face Hubで公開されています：

**🔗 [yukiharada1228/abn-resnet152-imagenette](https://huggingface.co/yukiharada1228/abn-resnet152-imagenette)**

### モデル仕様
- **アーキテクチャ**: ResNet152 + Attention Branch Network
- **データセット**: Imagenette (10クラス)
- **性能**: Top-1 Accuracy 90.47%, Top-5 Accuracy 99.21%
- **モデルサイズ**: 73.3M parameters
- **フォーマット**: Safetensors

### 使用方法

```python
from transformers import AutoModelForImageClassification
import torchvision.transforms as T
import torch

# モデルの読み込み
model = AutoModelForImageClassification.from_pretrained(
    "yukiharada1228/abn-resnet152-imagenette",
    trust_remote_code=True,
)
model.eval()

# 前処理
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 推論
with torch.no_grad():
    outputs = model(pixel_values=pixel_values)
    logits = outputs.logits
    attention_map = model.model.attention_map  # (B,1,H,W)
```

### 可視化

学習済みモデルを使用した可視化：

```bash
uv run visualize.py --ckpt yukiharada1228/abn-resnet152-imagenette --out-dir outputs --prefix abn
```

## 主な機能

- **Imagenette 10クラス分類**: 公式の `train/val` 分割をそのまま利用
- **注意機構の可視化**: 原画像とヒートマップ重畳を横並びペアでグリッド保存（クラス数に合わせて正方に近いレイアウト。例: 10クラス → 2×5 ペア）
- **複数の ResNet 対応**: ResNet18/34/50/101/152
- **Trainer 連携**: 最良モデルの自動保存・読み込みに対応
- **チェックポイント互換**: `model.safetensors` と `checkpoint-XXXX` のどちらからでも可視化可能

## プロジェクト構造

```
attention-branch-network/
├── models/                 # ABN モデル実装（HF互換）
│   ├── __init__.py
│   ├── configuration_abn.py
│   ├── modeling_abn.py
│   └── resnet_abn_backbone.py
├── data/                  # データセット（初回実行時に自動ダウンロード）
│   └── Imagenette/
├── checkpoint/            # Trainer 出力（最良モデルや epoch ごとの ckpt）
│   └── runs/              # TensorBoard 互換ログ
├── outputs/               # 可視化結果（まとめ画像）
│   └── abn_attentions.png
├── train.py               # 学習・評価（HF Trainer）
├── visualize.py           # 注意マップ可視化
├── main.py                # エントリ（サンプル）
├── pyproject.toml         # 依存関係（uv 対応）
└── uv.lock
```

## 動作環境

- Python 3.12 以上
- CUDA 環境（GPU 推奨。`--cpu` でCPU実行可）

## セットアップ

```bash
# uv を使用（推奨）
uv sync
```

## データセット（Imagenette）

`train.py`/`visualize.py` は初回実行時に Imagenette を自動ダウンロードします。

- 既定の保存先: `./data/Imagenette`
- サイズ指定: `--imagenette-size {full|320px|160px}`（既定: `full`）

## 使い方

### 学習

```bash
uv run train.py
```

- 最良モデルは `--checkpoint` で指定したディレクトリに保存されます（例: `checkpoint/model.safetensors`）。
- 学習途中のチェックポイントは `checkpoint-XXXX/` 形式で保存されます。
- ログ: TensorBoard 互換のイベントファイルを `checkpoint/runs/` に出力します。

#### 評価のみ

```bash
uv run train.py --evaluate --checkpoint checkpoint --gpu-id 0
```

### 可視化（注意マップ）

最良モデル（`checkpoint/model.safetensors`）または任意の `checkpoint-XXXX/` を指定できます。

```bash
# 最良モデルから可視化（既定パス）
uv run visualize.py --ckpt checkpoint/model.safetensors --out-dir outputs --prefix abn

# あるエポックの ckpt を指定
uv run visualize.py --ckpt checkpoint/checkpoint-1924 --out-dir outputs --prefix abn
```

主なオプション:

- 学習（train.py）
  - `--arch {resnet18,resnet34,resnet50,resnet101,resnet152}`（既定: `resnet152`）
  - `--imagenette-root`（既定: `./data/Imagenette`） / `--imagenette-size {full|320px|160px}`（既定: `full`）
  - `-j/--workers`（既定: 4）
  - `--train-batch`（既定: 64）/`--test-batch`（既定: 100）
  - `--epochs`（既定: 90）/`--lr`（既定: 0.1）/`--momentum`（既定: 0.9）/`--wd`（既定: 1e-4）
  - `--schedule`（既定: `31 61`）/`--gamma`（既定: 0.1）
  - `--checkpoint`（出力先、既定: `checkpoint`）/`--resume`（学習再開）
  - `--evaluate`（評価のみ）/`--gpu-id`（CUDA デバイス指定）/`--push-to-hub`（任意）

- 可視化（visualize.py）
  - `--ckpt`（既定: `checkpoint/model.safetensors`。`checkpoint-XXXX/` も可）
  - `--out-dir`（既定: `outputs`）/`--prefix`（既定: `abn`）/`--dpi`（既定: 200）
  - `--attention-alpha`（0.0–1.0、既定: 1.0。1.0で単純加算）/`--no-display`
  - `--arch` / `--imagenette-root` / `--imagenette-size` / `-j/--workers` / `--gpu-id` または `--cpu`

## 可視化結果・アルゴリズム

- `outputs/{prefix}_attentions.png` に、原画像と重畳ヒートマップのペアをタイル配置で保存します（既定: `abn_attentions.png`）。

実装の要点（ABN 論文実装に準拠しつつ簡潔・高速化）:

1. 画像復元: ImageNet 統計での正規化を反転し、RGB→BGR に変換
2. アテンション: `attention[0]` を 入力解像度へ `cv2.resize`
3. カラーマップ: `cv2.COLORMAP_JET` を適用
4. 合成: `cv2.add(original_bgr, jet_map)`。`--attention-alpha` で強度調整（1.0 で単純加算）
5. レイアウト: 各クラスから1枚ずつ抽出し、左に原画像・右に重畳画像のペアをタイル配置
6. 表示: 既定で表示、`--no-display` で保存のみ

## 対応アーキテクチャ

- ResNet18
- ResNet34
- ResNet50
- ResNet101
- ResNet152

## 依存関係

- PyTorch / torchvision
- Transformers / Accelerate
- NumPy
- Matplotlib（可視化）
- OpenCV（画像処理）
- TensorBoardX（ログ出力）

`pyproject.toml` に定義済みです。`uv sync` で環境構築できます。

## ライセンス

このリポジトリの `LICENSE` を参照してください。

## Acknowledgements

This project includes code from:
"Attention Branch Network: Learning of Attention Mechanism for Visual Explanation"  
by Hiroshi Fukui, Tsubasa Hirakawa, Takayoshi Yamashita, and Hironobu Fujiyoshi,  
licensed under the MIT License.  
Original repository: [https://github.com/machine-perception-robotics-group/attention_branch_network](https://github.com/machine-perception-robotics-group/attention_branch_network)
