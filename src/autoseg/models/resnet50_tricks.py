"""FCN-ResNet50 with each Bag of Tricks tweak switchable, for the post-course ablation.

tricks: any subset of {"b", "c", "d"} (He et al., CVPR 2019)
  b  stride on the 3x3 conv2 of the bottleneck instead of the 1x1 conv1
  c  three 3x3 convs (64, 64, 128) instead of the 7x7 stem
  d  AvgPool + 1x1 conv on the shortcut instead of a strided 1x1 conv
Empty set = torchvision-style FCN-ResNet50 (MyFcnResNet50); {c, d} = the course "v1d";
{b, c, d} = ResNet50-v1d as in the paper. Dilation 2 / 4 in layer3 / layer4, FCN + aux head.
"""
import torch.nn as nn
import torch.nn.functional as F

from .heads import fcn_head


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1, dilation=1, downsample=None, stride_on_conv2=False):
        super().__init__()
        s1, s2 = (1, stride) if stride_on_conv2 else (stride, 1)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=s1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride=s2, padding=dilation, dilation=dilation, bias=False)
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


class FCN_ResNet50_Tricks(nn.Module):
    def __init__(self, num_classes=13, tricks="", in_channels=3):
        super().__init__()
        tricks = {t.strip().lower() for t in tricks.split(",") if t.strip()}
        assert tricks <= {"b", "c", "d"}, tricks
        self.tricks = tricks
        self.deep_stem, self.avg_down, self.stride_on_conv2 = "c" in tricks, "d" in tricks, "b" in tricks
        if self.deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1))
            stem_out = 128
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1))
            stem_out = 64
        self.layer1 = self._make_layer(stem_out, 64, 3, stride=1, dilation=1)
        self.layer2 = self._make_layer(256, 128, 4, stride=2, dilation=1)
        self.layer3 = self._make_layer(512, 256, 6, stride=1, dilation=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=1, dilation=4)
        self.classifier = fcn_head(2048, 512, num_classes)
        self.aux_classifier = fcn_head(1024, 256, num_classes)

    def _make_layer(self, in_channels, mid_channels, blocks, stride, dilation):
        out_channels = mid_channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or in_channels != out_channels:
            if self.avg_down and stride != 1:
                downsample = nn.Sequential(nn.AvgPool2d(stride, stride=stride, ceil_mode=True),
                                           nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
            else:
                downsample = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                                           nn.BatchNorm2d(out_channels))
        layers = [Bottleneck(in_channels, mid_channels, stride, dilation, downsample, self.stride_on_conv2)]
        layers += [Bottleneck(out_channels, mid_channels, 1, dilation, None, self.stride_on_conv2) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        out = F.interpolate(self.classifier(x4), size=size, mode="bilinear", align_corners=False)
        aux = F.interpolate(self.aux_classifier(x3), size=size, mode="bilinear", align_corners=False)
        return {"out": out, "aux": aux}
