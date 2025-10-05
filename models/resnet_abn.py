import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet


class ResNetABN(ResNet):
    def __init__(self, block=BasicBlock, layers=(2, 2, 2, 2), num_classes=10):
        super().__init__(block, layers, num_classes=num_classes)

        feat_channels = self.layer3[0].conv1.in_channels

        self.att_norm = nn.BatchNorm2d(feat_channels)
        self.att_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
        )
        self.att_logits_pool = nn.AdaptiveAvgPool2d(1)
        self.att_map_head = nn.Sequential(
            nn.Conv2d(num_classes, num_classes, kernel_size=1, bias=False),
            nn.Conv2d(num_classes, 1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.attention_map = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        a = self.att_norm(x)
        a = self.att_head(a)
        att_logits = self.att_logits_pool(a).flatten(1)

        att_map = self.att_map_head(a)
        self.attention_map = att_map

        rx = x * att_map + x
        rx = self.layer3(rx)
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
