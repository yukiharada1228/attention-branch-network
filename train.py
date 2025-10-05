import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from transformers import Trainer, TrainingArguments

from models.resnet_abn import (ResNetABN, resnet18_abn, resnet34_abn,
                               resnet50_abn, resnet101_abn, resnet152_abn)


class ABNConfig:
    def __init__(self, arch: str, dataset: str, num_labels: int):
        self._name_or_path = arch
        self.arch = arch
        self.dataset = dataset
        self.num_labels = num_labels


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


class ABNForImageClassification(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, pixel_values=None, labels=None, **kwargs):
        # pixel_values: Tensor[B, 3, 224, 224] for PlantVillage
        att_logits, per_logits, _ = self.base_model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(att_logits, labels) + self.loss_fn(per_logits, labels)
        return {"loss": loss, "logits": per_logits}


class DataCollatorImageClassification:
    def __call__(self, features):
        # features: list of (img_tensor, label)
        pixel_values = torch.stack([f[0] for f in features])
        labels = torch.tensor([f[1] for f in features], dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions may be tuple when returning (loss, logits). We need logits only.
    if isinstance(predictions, (tuple, list)):
        logits = predictions[0]
    else:
        logits = predictions
    preds_top1 = np.argmax(logits, axis=-1)
    top1 = (preds_top1 == labels).mean().item()
    # Top-5
    top5 = 0.0
    if logits.ndim == 2 and logits.shape[1] >= 5:
        idx = np.argsort(-logits, axis=-1)[:, :5]
        matches = idx == labels[:, None]
        top5 = matches.any(axis=1).mean().item()
    return {"top1": top1, "top5": top5}


def main(args):
    # Env and seed to match original behavior
    if args.gpu_id:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.benchmark = True

    # PlantVillage dataset has 15 classes
    num_classes = 15
    base_model = build_from_arch(args.arch, num_classes=num_classes)

    model = ABNForImageClassification(base_model)
    model.config = ABNConfig(
        arch=args.arch, dataset=args.dataset, num_labels=num_classes
    )

    # Transforms for PlantVillage dataset (224x224 images) - ImageNet style
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load PlantVillage dataset
    if args.dataset == "plantvillage":
        # Use the PlantVillage directory structure
        data_dir = "./data/PlantVillage"
        # Create two dataset views with different transforms
        full_train = datasets.ImageFolder(root=data_dir, transform=transform_train)
        full_test = datasets.ImageFolder(root=data_dir, transform=transform_test)

        # Split indices with 80/20 using the provided seed for reproducibility
        num_samples = len(full_train)
        split_idx = int(num_samples * 0.8)
        g = torch.Generator()
        g.manual_seed(args.manualSeed)
        perm = torch.randperm(num_samples, generator=g).tolist()
        train_indices = perm[:split_idx]
        test_indices = perm[split_idx:]

        # Build Subset datasets so that transforms differ between train/eval
        train_data = torch.utils.data.Subset(full_train, train_indices)
        test_data = torch.utils.data.Subset(full_test, test_indices)
    else:
        raise ValueError("Dataset can only be plantvillage.")

    data_collator = DataCollatorImageClassification()

    training_args = TrainingArguments(
        output_dir=args.checkpoint,
        per_device_train_batch_size=args.train_batch,
        per_device_eval_batch_size=args.test_batch,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.wd,
        dataloader_num_workers=args.workers,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="top1",
        greater_is_better=True,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to=[],
    )

    # Optimizer and step LR schedule to match epoch milestones
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd
    )
    num_update_steps_per_epoch = math.ceil(
        len(train_data) / training_args.per_device_train_batch_size
    )
    milestones_steps = [m * num_update_steps_per_epoch for m in args.schedule]

    def lr_lambda(current_step: int) -> float:
        passed = 0
        for ms in milestones_steps:
            if current_step >= ms:
                passed += 1
        return args.gamma**passed

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),
    )

    if args.evaluate:
        eval_result = trainer.evaluate()
        print("Evaluation only ->", eval_result)
        return

    train_result = trainer.train(
        resume_from_checkpoint=(args.resume if args.resume else None)
    )
    eval_result = trainer.evaluate()
    trainer.save_model()
    if args.push_to_hub:
        trainer.push_to_hub(commit_message="End of training: push best model")
    is_best_loaded = training_args.load_best_model_at_end and (
        trainer.state.best_model_checkpoint is not None
    )
    label = "best eval" if is_best_loaded else "final eval"
    print(f"{label}:", eval_result)
    print(
        "best metric:",
        trainer.state.best_metric,
        "best step:",
        trainer.state.best_global_step,
        "best ckpt:",
        trainer.state.best_model_checkpoint,
    )


def parse_args():
    p = argparse.ArgumentParser()
    # Datasets
    p.add_argument("-d", "--dataset", default="plantvillage", type=str)
    p.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    # Optimization options
    p.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    p.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    p.add_argument(
        "--train-batch", default=64, type=int, metavar="N", help="train batchsize"
    )
    p.add_argument(
        "--test-batch", default=100, type=int, metavar="N", help="test batchsize"
    )
    p.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    p.add_argument(
        "--drop",
        "--dropout",
        dest="drop",
        default=0,
        type=float,
        metavar="Dropout",
        help="Dropout ratio",
    )
    p.add_argument(
        "--schedule",
        type=int,
        nargs="+",
        default=[31, 61],
        help="Decrease learning rate at these epochs.",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="LR is multiplied by gamma on schedule.",
    )
    p.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    p.add_argument(
        "--wd",
        "--weight-decay",
        dest="wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    # Checkpoints
    p.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        type=str,
        metavar="PATH",
        help="path to save checkpoint (default: checkpoint)",
    )
    p.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
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
    p.add_argument("--manualSeed", type=int, default=42, help="manual seed (default: 42)")
    p.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    # Device options
    p.add_argument(
        "--gpu-id", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
    )
    # Extra
    p.add_argument(
        "--save-every-epochs",
        type=int,
        default=10,
        help="save a checkpoint every N epochs (default: 10)",
    )
    p.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push final model to Hugging Face Hub (default: disabled)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
