# attention-branch-network

Attention Branch Network（ABN）の実装です。`torchvision.datasets.Imagenette`（10クラス）を用いた画像分類に適用し、モデルがどこを見て予測したかを可視化できます。

![Attention Maps](outputs/abn_attentions.png)

## 概要

このプロジェクトは ABN を ResNet 系バックボーン上に実装し、Imagenette データセット（`full` / `320px` / `160px`）での学習・評価・可視化を行います。学習には Hugging Face `Trainer` を用い、学習率スケジュールやチェックポイント保存を簡潔に扱えるようにしています。

## 訓練結果

ResNet152 + ABN での Imagenette 10クラス分類の結果:

- **Top-1 Accuracy**: 0.8540
- **Top-5 Accuracy**: 0.9758
- **Training Epochs**: 90 epochs

## 学習済みモデル

このプロジェクトで学習したモデルがHugging Face Hubで公開されています（Imagenette 10クラス向けモデル、公開準備中）：

**🔗 [yukiharada1228/abn-resnet152-imagenette](https://huggingface.co/yukiharada1228/abn-resnet152-imagenette)**

### モデル仕様
- **アーキテクチャ**: ResNet152 + Attention Branch Network
- **データセット**: imagenette (10クラス)
- **性能**: 学習完了後に更新予定
- **フォーマット**: Safetensors

### 可視化

学習済みモデルを使用した可視化：

```bash
uv run visualize.py -c yukiharada1228/abn-resnet152-imagenette
```

## 主な機能

- **Imagenette 10クラス分類**: `torchvision.datasets.Imagenette`を利用
- **注意機構の可視化**: 原画像とヒートマップ重畳を横並びペアでグリッド保存（指定したクラス数分のサンプルを表示）
- **画像処理モジュール**: `image_processing_abn.py` で画像の前処理・後処理を統合管理
- **Imagenetteユーティリティ**: `imagenette_utils.py` に共通ラッパーとクラス名整形処理を集約
- **複数の ResNet 対応**: ResNet18/34/50/101/152
- **Trainer 連携**: 最良モデルの自動保存・読み込みに対応
- **チェックポイント互換**: `model.safetensors` から可視化可能

## プロジェクト構造

```
attention-branch-network/
├── abn/                    # ABN モデル実装（HF互換）
│   ├── __init__.py
│   ├── configuration_abn.py
│   ├── image_processing_abn.py
│   ├── modeling_abn.py
│   └── resnet_abn_backbone.py
├── checkpoint/            # Trainer 出力（最良モデルや epoch ごとの ckpt）
│   └── runs/              # TensorBoard 互換ログ
├── outputs/               # 可視化結果（まとめ画像）
│   └── abn_attentions.png
├── imagenette_utils.py    # Imagenette用ユーティリティ
├── train.py               # 学習・評価（HF Trainer）
├── visualize.py           # 注意マップ可視化
├── demo.ipynb             # Jupyter デモノートブック
├── main.py                # エントリ（サンプル）
├── pyproject.toml         # 依存関係（uv 対応）
├── uv.lock
├── LICENSE
└── NOTICE.txt
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

`train.py` / `visualize.py` は初回実行時に Imagenette を自動ダウンロードします。

- データソース: `torchvision.datasets.Imagenette`
- クラス数: 10クラス（ImageNetのサブセット）
- サイズ: `full`（既定）、`320px`、`160px`を選択可能（`--imagenette-size`）
- ディレクトリ: 既定で `data/imagenette` に保存（`--data-root` で変更可能）
- 分割: train（学習用）、val（評価用）

## 使い方

### 学習

```bash
uv run train.py
```

#### 評価のみ

```bash
uv run train.py --evaluate --checkpoint checkpoint
```

### 可視化（注意マップ）

```bash
uv run visualize.py
```

## 可視化結果・アルゴリズム

- `outputs/{prefix}_attentions.png` に、原画像と重畳ヒートマップのペアをタイル配置で保存します（既定: `abn_attentions.png`）。

実装の要点（ABN 論文実装に準拠しつつ簡潔・高速化）:

1. 画像復元: ImageNet 統計での正規化を反転し、RGB→BGR に変換
2. アテンション: `attention[0]` を 入力解像度へ `cv2.resize`
3. カラーマップ: `cv2.COLORMAP_JET` を適用
4. 合成: `cv2.add(original_bgr, jet_map)`。`--attention-alpha` で強度調整（1.0 で単純加算）
5. レイアウト: 指定したクラス数分のサンプルを抽出し、左に原画像・右に重畳画像のペアをタイル配置
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

## Citation
If you find this repository is useful. Please cite the following references.

```bibtex
@article{fukui2018cvpr,
    author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},
    journal = {Computer Vision and Pattern Recognition},
    year = {2019},
    pages = {10705-10714}
}
```

```bibtex
@article{fukui2018arxiv,
    author = {Hiroshi Fukui and Tsubasa Hirakawa and Takayoshi Yamashita and Hironobu Fujiyoshi},
    title = {Attention Branch Network: Learning of Attention Mechanism for Visual Explanation},
    journal = {arXiv preprint arXiv:1812.10025},
    year = {2018}
}  
```
