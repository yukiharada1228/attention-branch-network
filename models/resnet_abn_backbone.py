import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNetABN(ResNet):
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2), num_classes=10):
        super().__init__(block, layers, num_classes=num_classes)

        feat_channels = 256 * block.expansion

        # att_layer4を作成（stride=1）
        self.inplanes = feat_channels
        self.att_layer4 = self._make_layer(block, 512, layers[3], stride=1)

        # Attention branch
        self.att_norm = nn.BatchNorm2d(512 * block.expansion)
        self.att_conv = nn.Conv2d(
            512 * block.expansion, num_classes, kernel_size=1, bias=False
        )
        self.att_norm2 = nn.BatchNorm2d(num_classes)
        self.att_conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False)
        self.att_conv3 = nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False)
        self.att_norm3 = nn.BatchNorm2d(1)
        self.att_logits_pool = nn.AdaptiveAvgPool2d(1)

        # layer4を再構築（stride=2）
        del self.layer4
        self.inplanes = feat_channels
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.attention_map = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Attention branch
        a = self.att_layer4(x)
        a = self.att_norm(a)
        a = self.att_conv(a)
        a = self.relu(self.att_norm2(a))

        # Attention map
        att_map = torch.sigmoid(self.att_norm3(self.att_conv3(a)))
        self.attention_map = att_map

        # Attention logits
        a = self.att_conv2(a)
        att_logits = self.att_logits_pool(a).flatten(1)

        # Perception branch
        rx = x * att_map + x
        rx = self.layer4(rx)
        rx = self.avgpool(rx)
        rx = torch.flatten(rx, 1)
        per_logits = self.fc(rx)

        return att_logits, per_logits, att_map


def resnet18_abn(num_classes: int = 1000) -> ResNetABN:
    return ResNetABN(
        block=BasicBlock,
        layers=(2, 2, 2, 2),
        num_classes=num_classes,
    )


def resnet34_abn(num_classes: int = 1000) -> ResNetABN:
    return ResNetABN(
        block=BasicBlock,
        layers=(3, 4, 6, 3),
        num_classes=num_classes,
    )


def resnet50_abn(num_classes: int = 1000) -> ResNetABN:
    return ResNetABN(
        block=Bottleneck,
        layers=(3, 4, 6, 3),
        num_classes=num_classes,
    )


def resnet101_abn(num_classes: int = 1000) -> ResNetABN:
    return ResNetABN(
        block=Bottleneck,
        layers=(3, 4, 23, 3),
        num_classes=num_classes,
    )


def resnet152_abn(num_classes: int = 1000) -> ResNetABN:
    return ResNetABN(
        block=Bottleneck,
        layers=(3, 8, 36, 3),
        num_classes=num_classes,
    )


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
