import argparse
import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from models import ABNForImageClassification


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

        # 可視化アルゴリズムは元スクリプトと完全一致で固定（正規化やカラーマップは変更不可）

        count = 0
        for cls_idx in sorted(class_images.keys()):
            img = class_images[cls_idx]
            att = class_attentions[cls_idx]
            pred = class_predictions[cls_idx]
            conf = class_confidences[cls_idx]

            # 画像の前処理
            d_input = img.data.cpu().numpy()
            # 元コードと同一式で再現（BGR）
            v_img = (
                (
                    d_input.transpose((1, 2, 0))
                    + 0.5
                    + np.array([0.485, 0.456, 0.406], dtype=np.float32)
                )
                * np.array([0.229, 0.224, 0.225], dtype=np.float32)
            ) * 256.0
            v_img = v_img[:, :, ::-1]

            # アテンションマップの前処理
            c_att = att.data.cpu().numpy()
            in_c, in_y, in_x = img.shape
            resize_att = cv2.resize(c_att[0], (in_x, in_y))
            # 元コードと同じ手順: 0-255 スケール、PNG 経由、JET、単純加算
            resize_att = resize_att * 255.0
            # imwrite の警告回避のため uint8 へ明示変換
            v_img_u8 = np.clip(v_img, 0, 255).astype(np.uint8)
            att_u8 = np.clip(resize_att, 0, 255).astype(np.uint8)
            cv2.imwrite("stock1.png", v_img_u8)
            cv2.imwrite("stock2.png", att_u8)
            v_img_disk = cv2.imread("stock1.png")
            vis_map = cv2.imread("stock2.png", 0)
            jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            jet_map = cv2.add(v_img_disk, jet_map)
            v_img = v_img_disk
            # 一時ファイル削除
            os.remove("stock1.png")
            os.remove("stock2.png")

            v_list.append(v_img)
            att_list.append(jet_map)
            label_list.append(cls_idx)
            pred_list.append(pred.item())
            conf_list.append(conf.item())

        # 表示用のレイアウト（原画像とアテンションを横並びペアで配置）
        # クラス数（=ペア数）に合わせ、できるだけ正方形かつ割り切れるなら綺麗な分割に
        # 例: 10 -> 2x5、12 -> 3x4、15 -> 3x5
        num_pairs = len(v_list)
        sqrt_floor = int(math.floor(math.sqrt(num_pairs)))
        rows = max(1, sqrt_floor)
        for r in range(sqrt_floor, 1, -1):
            if num_pairs % r == 0:
                rows = r
                break
        pair_cols = int(math.ceil(num_pairs / rows))

        # 図全体は 2*pair_cols 列（各ペアが2カラム占有）
        total_cols = pair_cols * 2
        fig_w = total_cols * 2  # 各セルを横2インチ程度で
        fig_h = rows * 2  # 各行を縦2インチ程度で

        fig_pairs = plt.figure(figsize=(fig_w, fig_h), dpi=args.dpi)
        plt.axis("off")

        for i, (v_img, att_img) in enumerate(zip(v_list, att_list)):
            r = i // pair_cols
            c_pair = i % pair_cols
            # 左: 原画像
            ax_left = fig_pairs.add_subplot(
                rows, total_cols, r * total_cols + c_pair * 2 + 1
            )
            ax_left.imshow(cv2.cvtColor(v_img, cv2.COLOR_BGR2RGB))
            ax_left.set_axis_off()
            # 右: 重畳ヒートマップ
            ax_right = fig_pairs.add_subplot(
                rows, total_cols, r * total_cols + c_pair * 2 + 2
            )
            ax_right.imshow(cv2.cvtColor(att_img, cv2.COLOR_BGR2RGB))
            ax_right.set_axis_off()

        plt.tight_layout(pad=0.05)

        # 1枚の画像として保存（例示のレイアウトに合わせた出力名）
        fig_pairs.savefig(
            os.path.join(args.out_dir, f"{args.prefix}_attentions.png"),
            dpi=args.dpi,
            bbox_inches="tight",
            pad_inches=0.05,
        )

        try:
            plt.show()
        except Exception:
            pass


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
