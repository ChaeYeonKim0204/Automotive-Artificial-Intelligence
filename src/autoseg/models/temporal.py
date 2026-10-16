"""Temporal head on top of a trained per-frame FCN (video segmentation fine-tune).

The backbone is frozen (parameters and BatchNorm statistics); only the FCN head and,
in "gru" mode, a ConvGRU branch are trained on short clips.

    gru:  z_t = head_pre(backbone(x_t));  h_t = ConvGRU(proj(z_t), h_{t-1});  f_t = z_t + out(h_t)
    none: f_t = z_t   (matched control: same trainable head, same clips, no recurrence)

``out`` is zero-initialised, so the "gru" model starts exactly as the base model.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGRUCell(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.zr = nn.Conv2d(2 * ch, 2 * ch, 3, padding=1)
        self.n = nn.Conv2d(2 * ch, ch, 3, padding=1)

    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
        z, r = torch.sigmoid(self.zr(torch.cat([x, h], 1))).chunk(2, 1)
        n = torch.tanh(self.n(torch.cat([x, r * h], 1)))
        return (1 - z) * h + z * n


class TemporalFCN(nn.Module):
    def __init__(self, base, mode="gru", hidden=128):
        super().__init__()
        assert mode in ("gru", "none"), mode
        assert hasattr(base, "backbone") and len(base.classifier) == 5, "expects FCN_ResNet50_v1d"
        self.backbone, self.mode = base.backbone, mode
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        head = copy.deepcopy(base.classifier)  # conv3x3 - BN - ReLU - Dropout - conv1x1
        self.head_pre, self.dropout, self.head_out = head[:3], head[3], head[4]
        if mode == "gru":
            c = self.head_pre[0].out_channels
            self.proj = nn.Conv2d(c, hidden, 1)
            self.cell = ConvGRUCell(hidden)
            self.out = nn.Conv2d(hidden, c, 1)
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # frozen BatchNorm statistics
        return self

    def encode(self, x):
        with torch.no_grad():
            _, x4 = self.backbone(x)
        return self.head_pre(x4)

    def fuse(self, z, h=None):
        if self.mode == "none":
            return z, None
        h = self.cell(self.proj(z), h)
        return z + self.out(h), h

    def decode(self, f, size):
        return F.interpolate(self.head_out(self.dropout(f)), size=size, mode="bilinear", align_corners=False)

    def forward(self, clip):
        """(B, T, C, H, W) -> (B, T, classes, H, W); the hidden state starts at zero for every clip."""
        h, outs = None, []
        for t in range(clip.shape[1]):
            f, h = self.fuse(self.encode(clip[:, t]), h)
            outs.append(self.decode(f, clip.shape[-2:]))
        return torch.stack(outs, 1)
