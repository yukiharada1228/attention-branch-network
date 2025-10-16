import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from train import ABNForImageClassification


def denormalize(img: np.ndarray):
    # ImageNet 統計で逆正規化 (PlantVillage 可視化に合わせる)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img.transpose(1, 2, 0) * std + mean) * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min)


def main(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # from_pretrained で学習済みモデルを読み込み
    os.makedirs(args.out_dir, exist_ok=True)
    model_wrapper = ABNForImageClassification.from_pretrained(
        args.ckpt, arch=args.arch, map_location=device, strict=False
    )
    model_wrapper.eval()

    # 評価用の変換 (train.py に準拠)
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Imagenette val split を利用
    # https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Imagenette.html
    test_data = datasets.Imagenette(
        root=args.imagenette_root,
        split="val",
        size=args.imagenette_size,
        download=True,
        transform=transform_test,
    )

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers
    )

    idx_to_cls = list(getattr(test_data, "classes", []))
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # 各クラスごとに一枚ずつ表示するために、クラスごとに画像を収集
        classes_list = list(idx_to_cls) if idx_to_cls else []
        num_classes = len(classes_list) if classes_list else 10

        # 各クラスから一枚ずつ画像を取得
        class_images = {}
        class_labels = {}
        class_attentions = {}
        class_predictions = {}
        class_confidences = {}

        # データローダーから各クラスの最初の画像を収集
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # ABN のアテンションは base_model 側の forward 出力を利用
            _, outputs, attention = model_wrapper.base_model(images)
            outputs = softmax(outputs)
            conf_data = outputs.data.topk(k=1, dim=1, largest=True, sorted=True)
            _, predicted = outputs.max(1)

            # 各クラスの最初の画像を1つずつ収集
            for i, (img, label) in enumerate(zip(images, labels)):
                cls_idx = label.item()
                if cls_idx not in class_images:
                    class_images[cls_idx] = img
                    class_labels[cls_idx] = label
                    class_attentions[cls_idx] = attention[i]
                    class_predictions[cls_idx] = predicted[i]
                    class_confidences[cls_idx] = conf_data[0][i]

                    # 全てのクラスを収集したら終了
                    if len(class_images) >= num_classes:
                        break

            if len(class_images) >= num_classes:
                break

        # 可視化用のデータを準備
        v_list = []
        att_list = []
        label_list = []
        pred_list = []
        conf_list = []

        for cls_idx in sorted(class_images.keys()):
            img = class_images[cls_idx]
            att = class_attentions[cls_idx]
            pred = class_predictions[cls_idx]
            conf = class_confidences[cls_idx]

            # 画像の前処理
            d_input = img.data.cpu().numpy()
            v_img = denormalize(d_input)

            # アテンションマップの前処理
            c_att = att.data.cpu().numpy()
            in_c, in_y, in_x = img.shape
            resize_att = cv2.resize(c_att[0], (in_x, in_y))
            resize_att = min_max(resize_att)
            resize_att *= 255.0

            v_img = np.uint8(v_img)
            resize_att = np.uint8(resize_att)
            jet_map = cv2.applyColorMap(resize_att, cv2.COLORMAP_JET)
            jet_map = cv2.addWeighted(v_img, 0.6, jet_map, 0.4, 0)

            v_list.append(v_img)
            att_list.append(jet_map)
            label_list.append(cls_idx)
            pred_list.append(pred.item())
            conf_list.append(conf.item())

        # 表示用のレイアウトを計算
        num_images = len(v_list)
        # 行数に応じて列数を自動調整
        rows = args.rows
        cols = (num_images + rows - 1) // rows  # 切り上げで列数を計算

        # Show input images
        fig_inputs = plt.figure(figsize=(cols * 2, rows * 2), dpi=args.dpi)
        plt.axis("off")

        for i, v_img in enumerate(v_list):
            ax = fig_inputs.add_subplot(rows, cols, i + 1)
            ax.imshow(v_img)
            ax.set_axis_off()
        plt.tight_layout()

        # Show attention maps
        fig_att = plt.figure(figsize=(cols * 2, rows * 2), dpi=args.dpi)
        plt.axis("off")

        for i, att_img in enumerate(att_list):
            ax = fig_att.add_subplot(rows, cols, i + 1)
            ax.imshow(att_img)
            ax.set_axis_off()
        plt.tight_layout()

        # Save figures to outputs (always)
        fig_inputs.savefig(
            os.path.join(args.out_dir, f"{args.prefix}_inputs.png"),
            dpi=args.dpi,
            bbox_inches="tight",
        )
        fig_att.savefig(
            os.path.join(args.out_dir, f"{args.prefix}_attentions.png"),
            dpi=args.dpi,
            bbox_inches="tight",
        )

        plt.show()


def parse_args():
    p = argparse.ArgumentParser()
    # Dataset (Imagenette 専用)
    p.add_argument(
        "--imagenette-size",
        default="full",
        type=str,
        choices=["full", "320px", "160px"],
        help="Imagenette のサイズバリアント",
    )
    p.add_argument(
        "--imagenette-root",
        default="./data/Imagenette",
        type=str,
        help="Imagenette のルートディレクトリ",
    )
    p.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    # Batch size naming aligned to train (--test-batch); keep --batch-size as alias
    p.add_argument(
        "--test-batch", "--batch-size", dest="test_batch", default=100, type=int
    )
    # Device options
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
    )
    # Architecture
    p.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="resnet152",
        choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        help="model architecture",
    )
    # Checkpoint path aligned with train default
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoint/model.safetensors",
        help="checkpoint dir or model.safetensors path",
    )
    # Visualization layout and saving
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--rows", type=int, default=2)
    p.add_argument(
        "--out-dir", type=str, default="outputs", help="directory to save figures"
    )
    p.add_argument(
        "--prefix", type=str, default="abn", help="filename prefix for saved figures"
    )
    p.add_argument("--dpi", type=int, default=200, help="figure DPI when saving")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
