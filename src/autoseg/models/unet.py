"""Classic U-Net (Ronneberger et al., 2015) with "same" padding and BatchNorm, trained from scratch.

Encoder 64-128-256-512-1024 channels (four 2x2 max-pools), decoder with 2x2 transposed
convolutions and skip concatenation, 1x1 classifier at full resolution. Inputs whose sides
are not multiples of 16 (e.g. 600x800) are zero-padded and the logits cropped back.
No auxiliary head: the training loss skips the aux term for this model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes=13, in_channels=3, base=64):
        super().__init__()
        ch = [base * 2 ** i for i in range(5)]  # 64, 128, 256, 512, 1024
        self.enc = nn.ModuleList([double_conv(in_channels, ch[0])] +
                                 [double_conv(ch[i - 1], ch[i]) for i in range(1, 5)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i], ch[i - 1], 2, stride=2) for i in range(4, 0, -1)])
        self.dec = nn.ModuleList([double_conv(ch[i], ch[i - 1]) for i in range(4, 0, -1)])
        self.classifier = nn.Conv2d(ch[0], num_classes, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, (-w) % 16, 0, (-h) % 16))
        skips = []
        for i, block in enumerate(self.enc):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([skips.pop(), up(x)], 1))
        return {"out": self.classifier(x)[..., :h, :w]}
