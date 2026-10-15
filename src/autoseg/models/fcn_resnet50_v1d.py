"""FCN on a ResNet-50 v1d backbone (final model).

Tweaks from "Bag of Tricks for Image Classification with CNNs" (He et al., 2019)
as they were written during the project:
  ResNet-C  7x7 stem replaced by three 3x3 convs (64, 64, 128)
  ResNet-D  shortcut downsample = AvgPool(stride) + 1x1 conv (stride 1)
  ResNet-B  intended, but the stride sits on the 1x1 conv1 here; the paper
            moves it to the 3x3 conv2 (which MyFcnResNet50 already does), so
            this tweak is effectively not applied. Kept as trained; see
            docs/postmortem.md.
Dilation 2 in layer3 and 4 in layer4 (output stride 8), FCN head + aux head.
"""
import torch.nn as nn
import torch.nn.functional as F

from .heads import fcn_head


class BottleneckV1d(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=1,
                               padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + identity)


class ResNet50_v1d_Backbone(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(128, 64, blocks=3, stride=1, dilation=1)
        self.layer2 = self._make_layer(256, 128, blocks=4, stride=2, dilation=1)
        self.layer3 = self._make_layer(512, 256, blocks=6, stride=1, dilation=2)
        self.layer4 = self._make_layer(1024, 512, blocks=3, stride=1, dilation=4)

    @staticmethod
    def _make_layer(in_channels, mid_channels, blocks, stride, dilation):
        out_channels = mid_channels * BottleneckV1d.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.AvgPool2d(stride, stride=stride, ceil_mode=True),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BottleneckV1d(in_channels, mid_channels, stride, dilation, downsample)]
        layers += [BottleneckV1d(out_channels, mid_channels, 1, dilation) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        return x3, x4


class FCN_ResNet50_v1d(nn.Module):
    def __init__(self, num_classes=13, in_channels=3):
        super().__init__()
        self.backbone = ResNet50_v1d_Backbone(in_channels)
        self.classifier = fcn_head(2048, 512, num_classes)
        self.aux_classifier = fcn_head(1024, 256, num_classes)

    def forward(self, x):
        size = x.shape[-2:]
        x3, x4 = self.backbone(x)
        out = F.interpolate(self.classifier(x4), size=size, mode="bilinear", align_corners=False)
        aux = F.interpolate(self.aux_classifier(x3), size=size, mode="bilinear", align_corners=False)
        return {"out": out, "aux": aux}
