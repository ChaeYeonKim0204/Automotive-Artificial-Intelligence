"""Evaluate a checkpoint on the Validation split and optionally save prediction figures.

    python -m autoseg.evaluate --data-root "2D Segmentation" --model fcn_resnet50_v1d \
        --ckpt models/fcn_resnet50_v1d.pt --figures results/figures --num-figures 3

--bias-json applies a decision adjustment saved from autoseg.calibrate:
    {"flip": true, "bias": {"Buildings": -2.3, ...}}  ->  argmax_c [log p_c + bias_c],
with p averaged over the image and its horizontal flip when "flip" is true.
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import CLASS_NAMES, colorize_mask, make_loaders
from .engine import evaluate
from .models import MODELS, build_model


class Adjusted(torch.nn.Module):
    """Wraps a model so that "out" holds log p (flip-averaged if asked) plus per-class biases."""

    def __init__(self, model, bias, flip):
        super().__init__()
        self.model, self.flip = model, flip
        self.register_buffer("bias", torch.tensor(bias, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, x):
        p = self.model(x)["out"].float().softmax(1)
        if self.flip:
            p = 0.5 * (p + self.model(x.flip(-1))["out"].float().softmax(1).flip(-1))
        return {"out": p.clamp_min(1e-8).log() + self.bias}


def save_prediction_figure(model, dataset, idx, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb, gt = dataset[idx]
    with torch.no_grad():
        pred = model(rgb.unsqueeze(0).to(device))["out"].argmax(1)[0].cpu().numpy()
    rgb_np = (rgb.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    panels = [("RGB", rgb_np), ("GT", colorize_mask(gt.numpy())), ("Pred", colorize_mask(pred)),
              ("Overlay", (0.6 * rgb_np + 0.4 * colorize_mask(pred)).astype(np.uint8))]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--model", choices=list(MODELS), default="fcn_resnet50_v1d")
    p.add_argument("--tricks", default="")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--figures", default=None, help="directory for prediction figures")
    p.add_argument("--num-figures", type=int, default=3)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (as in training and autoseg.calibrate)")
    p.add_argument("--bias-json", default=None, help="decision adjustment from autoseg.calibrate (see docstring)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = make_loaders(args.data_root, args.batch_size, aug="none", rare_weight=0)
    model = build_model(args.model, tricks=args.tricks).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    if args.bias_json:
        adj = json.load(open(args.bias_json))
        model = Adjusted(model, [adj["bias"].get(n, 0.0) for n in CLASS_NAMES], adj.get("flip", False)).to(device)

    _, pixel_acc, ious, miou, _ = evaluate(model, val_loader, device, amp=args.amp)
    print(f"Pixel Accuracy: {pixel_acc * 100:.2f}%")
    print(f"mIoU: {miou * 100:.2f}%")
    for i, (name, iou) in enumerate(zip(CLASS_NAMES, ious)):
        print(f"Class {i:2d} ({name:12s}): IoU = {'N/A' if np.isnan(iou) else f'{iou * 100:5.2f}%'}")

    if args.figures:
        os.makedirs(args.figures, exist_ok=True)
        model.eval()
        ds = val_loader.dataset
        for idx in np.linspace(0, len(ds) - 1, args.num_figures, dtype=int):
            save_prediction_figure(model, ds, int(idx), device, os.path.join(args.figures, f"pred_{idx:04d}.png"))


if __name__ == "__main__":
    main()
