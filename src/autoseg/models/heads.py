import torch.nn as nn


def fcn_head(in_channels, mid_channels, num_classes):
    """torchvision FCNHead: 3x3 conv - BN - ReLU - Dropout(0.1) - 1x1 conv."""
    return nn.Sequential(
        nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(mid_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(0.1),
        nn.Conv2d(mid_channels, num_classes, 1),
    )
