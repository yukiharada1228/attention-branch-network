from .configuration_abn import AbnConfig
from .modeling_abn import AbnModel, AbnModelForImageClassification


# AutoClass 登録ヘルパ（save_pretrained 時に auto_map を埋める）
def register_for_auto_class():
    AbnConfig.register_for_auto_class()
    AbnModel.register_for_auto_class("AutoModel")
    AbnModelForImageClassification.register_for_auto_class(
        "AutoModelForImageClassification"
    )


__all__ = [
    "AbnConfig",
    "AbnModel",
    "AbnModelForImageClassification",
]
