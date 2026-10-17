"""Check, before training, whether the synthetic glare (autoseg.data.add_glare) reproduces the backlit failure:
trained models on the five non-backlit Validation drives, clean vs with glare (fixed seed per frame).

    python experiments/postcourse/glare_check.py --data-root /path/2D_Segmentation \
        --member FCN=fcn_resnet50_v1d:fin_s42_last.pt --member UNet=unet:u_fin_s42_last.pt
"""
import argparse
import os
import random
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from autoseg.data import CLASS_NAMES, NUM_CLASSES, add_glare, frame_key, load_label  # noqa: E402
from autoseg.models import build_model  # noqa: E402

BACKLIT = ("F5", "F7")


def iou(conf):
    c = conf.cpu().numpy().astype(float)
    tp = np.diag(c)
    denom = c.sum(0) + c.sum(1) - tp
    return np.where(denom > 0, tp / np.maximum(denom, 1), np.nan) * 100, c


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--member", action="append", required=True, help="NAME=MODEL:CKPT")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = os.path.join(args.data_root, "Validation")
    nets = {}
    for m in args.member:
        name, rest = m.split("=", 1)
        arch, ckpt = rest.split(":", 1)
        nets[name] = build_model(arch).to(dev).eval()
        nets[name].load_state_dict(torch.load(ckpt, map_location=dev))
    names = [n for n in sorted(os.listdir(os.path.join(root, "RGB")), key=frame_key) if frame_key(n)[0] not in BACKLIT]
    conf = {(n, v): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev) for n in nets for v in ("clean", "glare")}
    for k, f in enumerate(names):
        rgb = Image.open(os.path.join(root, "RGB", f)).convert("RGB")
        label = load_label(os.path.join(root, "Label", f))
        y = torch.from_numpy(label.astype(np.int64)).to(dev)
        for variant, im in (("clean", rgb), ("glare", add_glare(rgb, random.Random(k), label))):
            x = TF.to_tensor(im)[None].to(dev)
            for n, net in nets.items():
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    pred = net(x)["out"].float().argmax(1)[0]
                conf[(n, variant)] += torch.bincount(y.view(-1) * NUM_CLASSES + pred.view(-1),
                                                     minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    road, build = CLASS_NAMES.index("Roads"), CLASS_NAMES.index("Buildings")
    for n in nets:
        clean, _ = iou(conf[(n, "clean")])
        glare, c = iou(conf[(n, "glare")])
        print(f"{n}: mIoU clean {np.nanmean(clean):.2f} -> glare {np.nanmean(glare):.2f} | RoadLines "
              f"{glare[CLASS_NAMES.index('RoadLines')]:.1f} | road -> Buildings {100 * c[road, build] / c[road].sum():.1f}%")


if __name__ == "__main__":
    main()
