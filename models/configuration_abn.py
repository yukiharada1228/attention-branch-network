from typing import Optional

from transformers import PretrainedConfig


class AbnConfig(PretrainedConfig):
    model_type = "abn"

    def __init__(
        self,
        arch: str = "resnet152",
        num_labels: int = 10,
        id2label: Optional[dict[int, str]] = None,
        label2id: Optional[dict[str, int]] = None,
        **kwargs,
    ) -> None:
        if arch.lower() not in {
            "resnet18",
            "resnet34",
            "resnet50",
            "resnet101",
            "resnet152",
        }:
            raise ValueError(
                f"`arch` must be one of resnet18/34/50/101/152, got {arch}"
            )

        if id2label is None and label2id is None:
            id2label = {i: str(i) for i in range(num_labels)}
            label2id = {v: k for k, v in id2label.items()}
        elif id2label is None:
            id2label = {v: k for k, v in label2id.items()}
        elif label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        # 先に PretrainedConfig 側へマッピングと num_labels を渡す
        # （kwargs は順序保持されるため id2label/label2id -> num_labels の順で評価される）
        super().__init__(
            id2label=id2label, label2id=label2id, num_labels=num_labels, **kwargs
        )

        # カスタム項目
        self.arch = arch
