from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from .configuration_abn import AbnConfig
from .resnet_abn_backbone import build_from_arch


class AbnModel(PreTrainedModel):
    config_class = AbnConfig

    def __init__(self, config: AbnConfig):
        super().__init__(config)
        base_model = build_from_arch(config.arch, num_classes=config.num_labels)
        self.model = base_model

    def forward(self, pixel_values: torch.Tensor, **kwargs):
        att_logits, per_logits, att_map = self.model(pixel_values)
        return {
            "att_logits": att_logits,
            "per_logits": per_logits,
            "att_map": att_map,
        }


class AbnModelForImageClassification(PreTrainedModel):
    config_class = AbnConfig

    def __init__(self, config: AbnConfig):
        super().__init__(config)
        base_model = build_from_arch(config.arch, num_classes=config.num_labels)
        self.model = base_model
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        att_logits, per_logits, _ = self.model(pixel_values)
        loss = None
        if labels is not None:
            loss = self.loss_fn(att_logits, labels) + self.loss_fn(per_logits, labels)
        return {"loss": loss, "logits": per_logits}
