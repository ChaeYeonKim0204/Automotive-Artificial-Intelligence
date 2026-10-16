from .fcn_resnet50 import MyFcnResNet50
from .fcn_resnet50_v1d import FCN_ResNet50_v1d
from .resnet50_tricks import FCN_ResNet50_Tricks
from .unet import UNet

MODELS = {"fcn_resnet50": MyFcnResNet50, "fcn_resnet50_v1d": FCN_ResNet50_v1d, "fcn_resnet50_tricks": FCN_ResNet50_Tricks,
          "unet": UNet}


def build_model(name, num_classes=13, tricks="", in_channels=3):
    """tricks (e.g. "b,c,d") only applies to fcn_resnet50_tricks; in_channels=6 for two-frame input."""
    if name == "fcn_resnet50_tricks":
        return FCN_ResNet50_Tricks(num_classes=num_classes, tricks=tricks, in_channels=in_channels)
    return MODELS[name](num_classes=num_classes, in_channels=in_channels)
