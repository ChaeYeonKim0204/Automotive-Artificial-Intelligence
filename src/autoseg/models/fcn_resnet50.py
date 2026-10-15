"""FCN-ResNet50 written from scratch (no pretrained weights).

Same layout as torchvision's fcn_resnet50: ResNet-50 backbone with dilation in
layer3 (2) and layer4 (4) so the output stride is 8, an FCN head on layer4 and
an auxiliary head on layer3, both upsampled bilinearly to the input size.
"""
import torch.nn as nn
import torch.nn.functional as F

from .heads import fcn_head


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=stride,
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


class MyFcnResNet50(nn.Module):
    def __init__(self, num_classes=13, in_channels=3):
        super().__init__()
        # attribute names match the notebook so its checkpoints load
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, stride=1, dilations=[1, 1, 1])
        self.layer2 = self._make_layer(256, 128, stride=2, dilations=[1, 1, 1, 1])
        self.layer3 = self._make_layer(512, 256, stride=1, dilations=[1, 2, 2, 2, 2, 2])
        self.layer4 = self._make_layer(1024, 512, stride=1, dilations=[2, 4, 4])
        self.classifier = fcn_head(2048, 512, num_classes)
        self.aux_classifier = fcn_head(1024, 256, num_classes)

    @staticmethod
    def _make_layer(in_channels, mid_channels, stride, dilations):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(in_channels, mid_channels, stride, dilations[0], downsample)]
        layers += [Bottleneck(out_channels, mid_channels, 1, d) for d in dilations[1:]]
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        out = F.interpolate(self.classifier(x4), size=size, mode="bilinear", align_corners=False)
        aux = F.interpolate(self.aux_classifier(x3), size=size, mode="bilinear", align_corners=False)
        return {"out": out, "aux": aux}
