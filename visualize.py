import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from safetensors.torch import load_file as safe_load_file

from models.resnet_abn import (ResNetABN, resnet18_abn, resnet34_abn,
                               resnet50_abn, resnet101_abn, resnet152_abn)


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


def build_from_arch(arch: str, num_classes: int) -> ResNetABN:
    a = (arch or "").lower()
    if a == "resnet18":
        return resnet18_abn(num_classes=num_classes)
    if a == "resnet34":
        return resnet34_abn(num_classes=num_classes)
    if a == "resnet50":
        return resnet50_abn(num_classes=num_classes)
    if a == "resnet101":
        return resnet101_abn(num_classes=num_classes)
    if a == "resnet152":
        return resnet152_abn(num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch}")


def _resolve_safetensors_path(ckpt_path: str) -> str:
    """Return path to .safetensors if input is a dir or a file; else empty."""
    if os.path.isdir(ckpt_path):
        cand = os.path.join(ckpt_path, "model.safetensors")
        if os.path.exists(cand):
            return cand
    if ckpt_path.endswith(".safetensors") and os.path.exists(ckpt_path):
        return ckpt_path
    return ""


def _strip_known_prefixes(key: str) -> str:
    prefixes = (
        "model.base_model.",
        "module.base_model.",
        "base_model.",
        "model.",
        "module.",
    )
    for p in prefixes:
        if key.startswith(p):
            return key[len(p) :]
    return key


def _load_weights_into_base_model(
    base_model: ResNetABN, ckpt_path: str, device: torch.device
):
    """Load weights into base_model from either safetensors or torch checkpoint.

    Supports:
    - Hugging Face-style `model.safetensors` containing keys prefixed with "base_model."
    - Raw safetensors state dict matching base model
    - torch.save dict with key "model" for base model state dict
    - raw torch state dict
    """
    if not ckpt_path:
        raise FileNotFoundError(
            "--ckpt が未指定です。Trainer出力ディレクトリ（例: .../checkpoint-XXXX）または model.safetensors のパスを指定してください。"
        )

    st_path = _resolve_safetensors_path(ckpt_path)
    if st_path:
        state = safe_load_file(st_path, device=str(device))
        normalized = {_strip_known_prefixes(k): v for k, v in state.items()}
        missing, unexpected = base_model.load_state_dict(normalized, strict=False)
        if unexpected:
            pass
        return

    # Fallback to torch checkpoints
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"チェックポイントが見つかりません: {ckpt_path}. ディレクトリ（model.safetensorsを含む）またはファイルを指定してください。"
        ) from e
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    normalized = {_strip_known_prefixes(k): v for k, v in state.items()}
    base_model.load_state_dict(normalized, strict=False)


def _read_normalized_state_dict(ckpt_path: str, device: torch.device):
    if not ckpt_path:
        raise FileNotFoundError(
            "--ckpt が未指定です。Trainer出力ディレクトリ（例: .../checkpoint-XXXX）または model.safetensors のパスを指定してください。"
        )
    st_path = _resolve_safetensors_path(ckpt_path)
    if st_path:
        state = safe_load_file(st_path, device=str(device))
    else:
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    normalized = {_strip_known_prefixes(k): v for k, v in state.items()}
    return normalized


def _infer_num_classes_from_state(state_dict) -> int | None:
    # Prefer classifier fc weight when available
    w = state_dict.get("fc.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) == 2:
        return int(w.shape[0])
    # Fallback to attention head conv/bn shapes
    w = state_dict.get("att_head.1.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) >= 1:
        return int(w.shape[0])
    w = state_dict.get("att_head.2.weight")
    if w is not None and hasattr(w, "shape") and len(w.shape) >= 1:
        return int(w.shape[0])
    return None


def main(args):
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )

    # Try to read checkpoint first to infer number of classes
    normalized_state = None
    inferred_num_classes = None
    if args.ckpt:
        try:
            normalized_state = _read_normalized_state_dict(args.ckpt, device)
            inferred_num_classes = _infer_num_classes_from_state(normalized_state)
        except Exception:
            normalized_state = None
            inferred_num_classes = None

    # dataset and classes (PlantVillage 15 クラスに整合; ckptから推定があれば優先)
    num_classes = inferred_num_classes if inferred_num_classes is not None else 15

    # model (train.py と同等のビルド)
    model = build_from_arch(args.arch, num_classes=num_classes).to(device)

    os.makedirs(args.out_dir, exist_ok=True)

    if normalized_state is not None:
        model.load_state_dict(normalized_state, strict=False)
    else:
        _load_weights_into_base_model(model, args.ckpt, device)
    model.eval()

    # PlantVillage 可視化用の評価変換 (train.py に準拠)
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if args.dataset != "plantvillage":
        raise ValueError("Dataset can only be plantvillage.")
    data_dir = "./data/PlantVillage"
    test_data = datasets.ImageFolder(root=data_dir, transform=transform_test)

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.test_batch, shuffle=True, num_workers=args.workers
    )

    idx_to_cls = list(getattr(test_data, "classes", []))
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        # faithfully reproduce the reference visualization flow
        images, labels = next(iter(loader))
        images = images.to(device)
        labels = labels.to(device)

        _, outputs, attention = model(images)
        outputs = softmax(outputs)
        conf_data = outputs.data.topk(k=1, dim=1, largest=True, sorted=True)
        _, predicted = outputs.max(1)

        c_att = attention.data.cpu().numpy()
        d_inputs = images.data.cpu().numpy()

        v_list = []
        att_list = []
        in_b, in_c, in_y, in_x = images.shape
        for item_img, item_att in zip(d_inputs, c_att):
            v_img = denormalize(item_img)
            resize_att = cv2.resize(item_att[0], (in_x, in_y))
            resize_att = min_max(resize_att)
            resize_att *= 255.0

            v_img = np.uint8(v_img)
            resize_att = np.uint8(resize_att)
            jet_map = cv2.applyColorMap(resize_att, cv2.COLORMAP_JET)
            jet_map = cv2.addWeighted(v_img, 0.6, jet_map, 0.4, 0)
            v_list.append(v_img)
            att_list.append(jet_map)

        # Show input images
        cols = args.cols
        rows = args.rows
        classes_list = list(idx_to_cls) if idx_to_cls else []

        fig_inputs = plt.figure(figsize=(14, 3.0), dpi=args.dpi)
        plt.title("Input image")
        plt.axis("off")
        for r in range(rows):
            for c in range(cols):
                if c >= len(labels):
                    break
                cls_idx = labels[c].item()
                ax = fig_inputs.add_subplot(r + 1, cols, c + 1)
                title_txt = (
                    classes_list[cls_idx]
                    if classes_list and cls_idx < len(classes_list)
                    else str(cls_idx)
                )
                plt.title(f"{title_txt}")
                ax.imshow(v_list[cols * r + c])
                ax.set_axis_off()
        plt.tight_layout()

        # Show attention maps
        fig_att = plt.figure(figsize=(14, 3.5), dpi=args.dpi)
        plt.title("Attention map")
        plt.axis("off")
        for r in range(rows):
            for c in range(cols):
                if c >= len(predicted):
                    break
                pred_idx = predicted[c].item()
                conf = conf_data[0][c].item()
                ax = fig_att.add_subplot(r + 1, cols, c + 1)
                ax.imshow(att_list[cols * r + c])
                pred_name = (
                    classes_list[pred_idx]
                    if classes_list and pred_idx < len(classes_list)
                    else str(pred_idx)
                )
                plt.title(f"pred: {pred_name}\nconf: {conf:.2f}")
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
    # Dataset to align with train script
    p.add_argument("-d", "--dataset", default="plantvillage", type=str)
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
        "--ckpt", type=str, default="checkpoint/model.safetensors", help="checkpoint dir or model.safetensors path"
    )
    # Visualization layout and saving
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--rows", type=int, default=1)
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
