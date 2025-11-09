from typing import Any

from torchvision.datasets import Imagenette


class ImagenetteDictDataset(Imagenette):
    """torchvision.datasets.Imagenetteを辞書形式で返すラッパー。"""

    def __getitem__(self, idx: int):
        image, target = super().__getitem__(idx)
        return {"image": image, "label": target}


def normalize_class_name(name: Any) -> str:
    """Imagenetteのクラス名を文字列へ整形する。"""

    if isinstance(name, (tuple, list)):
        return " / ".join(str(n) for n in name)
    return str(name)
