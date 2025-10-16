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

from models import ABNForImageClassification, build_from_arch


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

    # 画像前処理（ImageNet 標準統計）
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

    # Imagenette 専用（10クラス、公式の train/val split を使用）
    # https://docs.pytorch.org/vision/main/generated/torchvision.datasets.Imagenette.html
    num_classes = 10
    train_data = datasets.Imagenette(
        root=args.imagenette_root,
        split="train",
        size=args.imagenette_size,
        download=True,
        transform=transform_train,
    )
    test_data = datasets.Imagenette(
        root=args.imagenette_root,
        split="val",
        size=args.imagenette_size,
        download=True,
        transform=transform_test,
    )

    if args.evaluate:
        model = ABNForImageClassification.from_pretrained(
            args.checkpoint, arch=args.arch
        )
    else:
        base_model = build_from_arch(args.arch, num_classes=num_classes)
        model = ABNForImageClassification(base_model)

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
        report_to=["tensorboard"],
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
    p.add_argument(
        "--manualSeed", type=int, default=42, help="manual seed (default: 42)"
    )
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
