from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from transformers import ImageProcessingMixin
from transformers.image_utils import ImageInput, make_list_of_images


class AbnImageProcessor(ImageProcessingMixin):
    """
    ABN (Attention Branch Network) 用の画像プロセッサー

    ImageNet標準の前処理を適用します：
    - リサイズ: 256px
    - 中心クロップ: 224x224
    - 正規化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        do_resize: bool = True,
        size: Dict[str, int] = None,
        do_center_crop: bool = True,
        crop_size: Dict[str, int] = None,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # デフォルト値の設定
        if size is None:
            size = {"height": 256, "width": 256}
        if crop_size is None:
            crop_size = {"height": 224, "width": 224}
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.do_resize = do_resize
        self.size = size
        self.do_center_crop = do_center_crop
        self.crop_size = crop_size
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std

    def preprocess(
        self,
        images: ImageInput,
        do_resize: Optional[bool] = None,
        size: Optional[Dict[str, int]] = None,
        do_center_crop: Optional[bool] = None,
        crop_size: Optional[Dict[str, int]] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[Union[int, float]] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        return_tensors: Optional[Union[str, torch.dtype]] = None,
        data_format: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        画像を前処理してテンソルに変換します。
        """
        # パラメータの設定
        do_resize = do_resize if do_resize is not None else self.do_resize
        size = size if size is not None else self.size
        do_center_crop = (
            do_center_crop if do_center_crop is not None else self.do_center_crop
        )
        crop_size = crop_size if crop_size is not None else self.crop_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = (
            rescale_factor if rescale_factor is not None else self.rescale_factor
        )
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std

        # 画像をリストに変換
        images = make_list_of_images(images)

        # 各画像を処理
        processed_images = []
        for image in images:
            # PIL Imageに変換
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # RGBに変換
            if image.mode != "RGB":
                image = image.convert("RGB")

            # リサイズ
            if do_resize:
                image = image.resize(
                    (size["width"], size["height"]), Image.Resampling.BILINEAR
                )

            # 中心クロップ
            if do_center_crop:
                width, height = image.size
                crop_width, crop_height = crop_size["width"], crop_size["height"]
                left = (width - crop_width) // 2
                top = (height - crop_height) // 2
                right = left + crop_width
                bottom = top + crop_height
                image = image.crop((left, top, right, bottom))

            # テンソルに変換
            image_array = torch.tensor(list(image.getdata()), dtype=torch.float32)
            image_array = image_array.view(image.size[1], image.size[0], 3)
            image_array = image_array.permute(2, 0, 1)  # HWC -> CHW

            # リスケール
            if do_rescale:
                image_array = image_array * rescale_factor

            # 正規化
            if do_normalize:
                mean = torch.tensor(image_mean).view(3, 1, 1)
                std = torch.tensor(image_std).view(3, 1, 1)
                image_array = (image_array - mean) / std

            processed_images.append(image_array)

        # バッチにスタック
        pixel_values = torch.stack(processed_images)

        return {"pixel_values": pixel_values}

    def __call__(self, images, **kwargs):
        """
        画像を前処理します。preprocessメソッドへのショートカット。
        """
        return self.preprocess(images, **kwargs)

    def post_process_semantic_segmentation(self, outputs, target_sizes=None):
        """
        セマンティックセグメンテーション用の後処理（ABNでは使用しない）
        """
        raise NotImplementedError("ABN does not support semantic segmentation")

    def post_process_instance_segmentation(self, outputs, target_sizes=None):
        """
        インスタンスセグメンテーション用の後処理（ABNでは使用しない）
        """
        raise NotImplementedError("ABN does not support instance segmentation")

    def post_process_panoptic_segmentation(self, outputs, target_sizes=None):
        """
        パノプティックセグメンテーション用の後処理（ABNでは使用しない）
        """
        raise NotImplementedError("ABN does not support panoptic segmentation")

    def post_process_object_detection(self, outputs, target_sizes=None):
        """
        オブジェクト検出用の後処理（ABNでは使用しない）
        """
        raise NotImplementedError("ABN does not support object detection")

    def to_dict(self):
        """
        プロセッサーの設定を辞書形式で返します。
        preprocessor_config.jsonの生成に使用されます。
        """
        output = super().to_dict()
        output.update(
            {
                "do_resize": self.do_resize,
                "size": self.size,
                "do_center_crop": self.do_center_crop,
                "crop_size": self.crop_size,
                "do_rescale": self.do_rescale,
                "rescale_factor": self.rescale_factor,
                "do_normalize": self.do_normalize,
                "image_mean": self.image_mean,
                "image_std": self.image_std,
            }
        )
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        辞書からプロセッサーを復元します。
        """
        # 必要なパラメータを抽出
        processor_kwargs = {
            "do_resize": config_dict.get("do_resize", True),
            "size": config_dict.get("size", {"height": 256, "width": 256}),
            "do_center_crop": config_dict.get("do_center_crop", True),
            "crop_size": config_dict.get("crop_size", {"height": 224, "width": 224}),
            "do_rescale": config_dict.get("do_rescale", True),
            "rescale_factor": config_dict.get("rescale_factor", 1 / 255),
            "do_normalize": config_dict.get("do_normalize", True),
            "image_mean": config_dict.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": config_dict.get("image_std", [0.229, 0.224, 0.225]),
        }
        processor_kwargs.update(kwargs)
        return cls(**processor_kwargs)


class AbnImageProcessorForTraining(AbnImageProcessor):
    """
    学習用の画像プロセッサー（データ拡張を含む）
    """

    def __init__(
        self,
        do_random_resized_crop: bool = True,
        random_crop_size: Dict[str, int] = None,
        do_random_horizontal_flip: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if random_crop_size is None:
            random_crop_size = {"height": 224, "width": 224}

        self.do_random_resized_crop = do_random_resized_crop
        self.random_crop_size = random_crop_size
        self.do_random_horizontal_flip = do_random_horizontal_flip

    def preprocess(
        self,
        images: ImageInput,
        do_random_resized_crop: Optional[bool] = None,
        random_crop_size: Optional[Dict[str, int]] = None,
        do_random_horizontal_flip: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        学習用の前処理（データ拡張を含む）
        """
        # パラメータの設定（常にデータ拡張を適用）
        do_random_resized_crop = (
            do_random_resized_crop
            if do_random_resized_crop is not None
            else self.do_random_resized_crop
        )
        random_crop_size = (
            random_crop_size if random_crop_size is not None else self.random_crop_size
        )
        do_random_horizontal_flip = (
            do_random_horizontal_flip
            if do_random_horizontal_flip is not None
            else self.do_random_horizontal_flip
        )

        # 画像をリストに変換
        images = make_list_of_images(images)

        # 各画像を処理
        processed_images = []
        for image in images:
            # PIL Imageに変換
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            # RGBに変換
            if image.mode != "RGB":
                image = image.convert("RGB")

            # ランダムリサイズクロップ
            if do_random_resized_crop:
                import torchvision.transforms as transforms

                transform = transforms.RandomResizedCrop(
                    (random_crop_size["height"], random_crop_size["width"]),
                    scale=(0.08, 1.0),
                    ratio=(0.75, 1.33),
                )
                image = transform(image)

            # ランダム水平フリップ
            if do_random_horizontal_flip:
                import random

                if random.random() > 0.5:
                    image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

            # テンソルに変換
            image_array = torch.tensor(list(image.getdata()), dtype=torch.float32)
            image_array = image_array.view(image.size[1], image.size[0], 3)
            image_array = image_array.permute(2, 0, 1)  # HWC -> CHW

            # リスケール
            if self.do_rescale:
                image_array = image_array * self.rescale_factor

            # 正規化
            if self.do_normalize:
                mean = torch.tensor(self.image_mean).view(3, 1, 1)
                std = torch.tensor(self.image_std).view(3, 1, 1)
                image_array = (image_array - mean) / std

            processed_images.append(image_array)

        # バッチにスタック
        pixel_values = torch.stack(processed_images)

        return {"pixel_values": pixel_values}

    def to_dict(self):
        """
        学習用プロセッサーの設定を辞書形式で返します。
        """
        output = super().to_dict()
        output.update(
            {
                "do_random_resized_crop": self.do_random_resized_crop,
                "random_crop_size": self.random_crop_size,
                "do_random_horizontal_flip": self.do_random_horizontal_flip,
            }
        )
        return output

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        辞書から学習用プロセッサーを復元します。
        """
        # 親クラスのパラメータを抽出
        parent_kwargs = {
            "do_resize": config_dict.get("do_resize", True),
            "size": config_dict.get("size", {"height": 256, "width": 256}),
            "do_center_crop": config_dict.get("do_center_crop", True),
            "crop_size": config_dict.get("crop_size", {"height": 224, "width": 224}),
            "do_rescale": config_dict.get("do_rescale", True),
            "rescale_factor": config_dict.get("rescale_factor", 1 / 255),
            "do_normalize": config_dict.get("do_normalize", True),
            "image_mean": config_dict.get("image_mean", [0.485, 0.456, 0.406]),
            "image_std": config_dict.get("image_std", [0.229, 0.224, 0.225]),
        }

        # 学習用特有のパラメータを抽出
        training_kwargs = {
            "do_random_resized_crop": config_dict.get("do_random_resized_crop", True),
            "random_crop_size": config_dict.get(
                "random_crop_size", {"height": 224, "width": 224}
            ),
            "do_random_horizontal_flip": config_dict.get(
                "do_random_horizontal_flip", True
            ),
        }

        # 全てのパラメータをマージ
        all_kwargs = {**parent_kwargs, **training_kwargs, **kwargs}
        return cls(**all_kwargs)
