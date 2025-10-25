import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoImageProcessor, AutoModel


def denormalize_image(img_tensor, mean, std):
    """正規化された画像を元に戻す"""
    img = img_tensor.cpu().numpy().transpose((1, 2, 0))
    img = (img * np.array(std) + np.array(mean)) * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def apply_attention_overlay(image, attention_map, alpha=0.5):
    """アテンションマップを画像に重畳

    Args:
        image: 元画像 (BGR)
        attention_map: アテンションマップ (H, W)
        alpha: アテンションの強度 (0.0-1.0), デフォルト0.5
    """
    h, w = image.shape[:2]
    # アテンションマップをリサイズ
    att_resized = cv2.resize(attention_map, (w, h))
    # 0-255にスケール
    att_scaled = (att_resized * 255.0).astype(np.uint8)
    # JETカラーマップを適用
    jet_map = cv2.applyColorMap(att_scaled, cv2.COLORMAP_JET)

    # アテンションを強調するため、元のコードと同様に加算方式を使用
    # alpha=1.0で元のコードと完全一致（単純加算）
    # alpha<1.0でアテンションを減衰可能
    if alpha >= 1.0:
        # 元のコードと同じ単純加算
        overlay = cv2.add(image, jet_map)
    else:
        # アテンション強度を調整可能に
        jet_map_scaled = (jet_map * alpha).astype(np.uint8)
        overlay = cv2.add(image, jet_map_scaled)

    return overlay


def main(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # 学習済みモデルを読み込み
    os.makedirs(args.out_dir, exist_ok=True)
    model = AutoModel.from_pretrained(args.checkpoint, trust_remote_code=True)
    # 入力テンソルと同じデバイスにモデルを移動
    model.to(device)
    model.eval()

    # 評価用のImageProcessorを初期化
    image_processor = AutoImageProcessor.from_pretrained(
        args.checkpoint, trust_remote_code=True
    )

    # ImageNet-1kデータセットを読み込み
    test_data = load_dataset(
        "ILSVRC/imagenet-1k", split="validation", trust_remote_code=True
    )
    num_classes = 1000
    display_classes = args.num_classes

    # DataCollatorを作成
    class DataCollatorImageClassification:
        def __init__(self, image_processor):
            self.image_processor = image_processor

        def __call__(self, features):
            # features: list of {"image": PIL.Image, "label": int}
            images = [f["image"] for f in features]
            labels = torch.tensor([f["label"] for f in features], dtype=torch.long)

            # ImageProcessorで前処理
            processed = self.image_processor(images, return_tensors="pt")
            processed["labels"] = labels
            return processed

    data_collator = DataCollatorImageClassification(image_processor)

    loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_batch,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data_collator,
    )

    # クラス名を取得
    try:
        features = test_data.features
        if "label" in features and hasattr(features["label"], "int2str"):
            idx_to_cls = [features["label"].int2str(i) for i in range(num_classes)]
        else:
            idx_to_cls = [f"Class {i}" for i in range(num_classes)]
    except Exception as e:
        print(f"Warning: Could not extract class names from dataset: {e}")
        idx_to_cls = [f"Class {i}" for i in range(num_classes)]
    softmax = nn.Softmax(dim=1)

    print(f"データセットから各クラス1枚ずつ（計{display_classes}枚）を収集中...")

    with torch.no_grad():
        # 各クラスから一枚ずつ画像を収集
        class_data = {}

        for batch in loader:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # ABNモデルのforwardメソッドからatt_mapを直接取得
            model_outputs = model(pixel_values=images)
            outputs = model_outputs["per_logits"]  # 予測用のlogits
            attention = model_outputs["att_map"]  # アテンションマップ
            probs = softmax(outputs)
            confidences, predicted = probs.max(1)

            # 各クラスの最初の画像を1つずつ収集
            for i, label in enumerate(labels):
                cls_idx = label.item()
                if cls_idx not in class_data:
                    class_data[cls_idx] = {
                        "image": images[i],
                        "attention": attention[i],
                        "predicted": predicted[i].item(),
                        "confidence": confidences[i].item(),
                        "label": cls_idx,
                    }

                    # 指定されたクラス数分収集完了
                    if len(class_data) >= display_classes:
                        break

            if len(class_data) >= display_classes:
                break

    print(f"{len(class_data)}クラスのデータを収集完了")

    # 正規化パラメータ
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # 可視化データを準備
    vis_data = []
    for cls_idx in sorted(class_data.keys()):
        data = class_data[cls_idx]

        # 画像の前処理（BGR形式に変換）
        img_rgb = denormalize_image(data["image"], mean, std)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # アテンションマップを重畳
        att_map = data["attention"].cpu().numpy()[0]
        overlay = apply_attention_overlay(img_bgr, att_map, alpha=args.attention_alpha)

        # クラス名を取得
        true_label = (
            idx_to_cls[data["label"]] if idx_to_cls else f"Class {data['label']}"
        )
        pred_label = (
            idx_to_cls[data["predicted"]]
            if idx_to_cls
            else f"Class {data['predicted']}"
        )

        vis_data.append(
            {
                "original": img_bgr,
                "overlay": overlay,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": data["confidence"],
                "correct": data["label"] == data["predicted"],
            }
        )

    # レイアウトを計算（原画像とアテンションを横並びペアで配置）
    num_pairs = len(vis_data)
    sqrt_floor = int(math.floor(math.sqrt(num_pairs)))
    rows = max(1, sqrt_floor)
    for r in range(sqrt_floor, 1, -1):
        if num_pairs % r == 0:
            rows = r
            break
    pair_cols = int(math.ceil(num_pairs / rows))
    total_cols = pair_cols * 2

    # 図のサイズ設定（クラス数に応じて調整）
    fig_w = total_cols * 3
    fig_h = rows * 3.5

    print(f"レイアウト: {rows}行 × {pair_cols}ペア（計{total_cols}列）で配置")
    print(f"表示クラス数: {len(vis_data)}/{display_classes}")

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)

    for i, data in enumerate(vis_data):
        r = i // pair_cols
        c_pair = i % pair_cols

        # 左: 原画像
        ax_left = fig.add_subplot(rows, total_cols, r * total_cols + c_pair * 2 + 1)
        ax_left.imshow(cv2.cvtColor(data["original"], cv2.COLOR_BGR2RGB))
        ax_left.set_axis_off()
        ax_left.set_title(f"True: {data['true_label']}", fontsize=8, pad=2)

        # 右: アテンション重畳画像
        ax_right = fig.add_subplot(rows, total_cols, r * total_cols + c_pair * 2 + 2)
        ax_right.imshow(cv2.cvtColor(data["overlay"], cv2.COLOR_BGR2RGB))
        ax_right.set_axis_off()

        # 予測結果を色分けして表示
        color = "green" if data["correct"] else "red"
        title = f"Pred: {data['pred_label']}\n({data['confidence']:.2%})"
        ax_right.set_title(title, fontsize=8, pad=2, color=color)

    plt.tight_layout(pad=0.3)

    # 保存
    output_path = os.path.join(args.out_dir, f"{args.prefix}_attentions.png")
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"可視化結果を保存: {output_path}")

    # 精度を計算して表示
    correct = sum(1 for d in vis_data if d["correct"])
    accuracy = correct / len(vis_data) * 100
    print(f"表示サンプルの精度: {correct}/{len(vis_data)} ({accuracy:.1f}%)")

    if not args.no_display:
        try:
            plt.show()
        except Exception as e:
            print(f"表示エラー: {e}")

    plt.close()


def parse_args():
    p = argparse.ArgumentParser(description="ABNモデルのアテンション可視化")
    p.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="データローダーのワーカー数 (default: 4)",
    )
    p.add_argument(
        "--test-batch",
        default=100,
        type=int,
        help="テストバッチサイズ (default: 100)",
    )

    # Device
    p.add_argument("--cpu", action="store_true", help="CPUを強制使用")
    p.add_argument(
        "--gpu-id", default="0", type=str, help="CUDA_VISIBLE_DEVICES (default: 0)"
    )

    # Model
    p.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default="checkpoint",
        help="チェックポイントパス (default: checkpoint)",
    )

    # Output
    p.add_argument(
        "--out-dir",
        type=str,
        default="outputs",
        help="出力ディレクトリ (default: outputs)",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="abn",
        help="出力ファイル名のプレフィックス (default: abn)",
    )
    p.add_argument("--dpi", type=int, default=200, help="保存時のDPI (default: 200)")
    p.add_argument(
        "--attention-alpha",
        type=float,
        default=1.0,
        help="アテンション強度 (0.0-1.0, 1.0=最大強度・元のコードと同じ, default: 1.0)",
    )
    p.add_argument(
        "--no-display", action="store_true", help="画像を表示せずに保存のみ実行"
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="表示するクラス数 (default: 10)",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
