"""How much does each model rely on the rest of the frame? Each tile of a g x g grid is predicted on its own
and the tiles are stitched; mIoU on the backlit drives F5 + F7 and on the other five.

    python experiments/postcourse/context_check.py --data-root /path/2D_Segmentation --grids 1,2 \
        --member FCN=fcn_resnet50_v1d:fin_s42_last.pt --member UNet=unet:u_fin_s42_last.pt
"""
import argparse
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from autoseg.data import NUM_CLASSES, frame_key, load_label  # noqa: E402
from autoseg.models import build_model  # noqa: E402

BACKLIT = ("F5", "F7")


def miou(conf):
    c = conf.cpu().numpy().astype(float)
    tp = np.diag(c)
    denom = c.sum(0) + c.sum(1) - tp
    return float(np.nanmean(np.where(denom > 0, tp / np.maximum(denom, 1), np.nan))) * 100


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--member", action="append", required=True, help="NAME=MODEL:CKPT")
    p.add_argument("--grids", default="1,2,4", help="tiles per side, 1 = full frame")
    args = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grids = [int(g) for g in args.grids.split(",")]
    root = os.path.join(args.data_root, "Validation")
    nets = {}
    for m in args.member:
        name, rest = m.split("=", 1)
        arch, ckpt = rest.split(":", 1)
        nets[name] = build_model(arch).to(dev).eval()
        nets[name].load_state_dict(torch.load(ckpt, map_location=dev))
    conf = {(n, g, b): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=dev)
            for n in nets for g in grids for b in (True, False)}
    for f in sorted(os.listdir(os.path.join(root, "RGB")), key=frame_key):
        x = TF.to_tensor(Image.open(os.path.join(root, "RGB", f)).convert("RGB"))[None].to(dev)
        y = torch.from_numpy(load_label(os.path.join(root, "Label", f)).astype(np.int64)).to(dev)
        backlit = frame_key(f)[0] in BACKLIT
        h, w = y.shape
        for n, net in nets.items():
            for g in grids:
                pred = torch.empty(h, w, dtype=torch.long, device=dev)
                th, tw = h // g, w // g
                for i in range(g):
                    for j in range(g):
                        tile = (slice(i * th, (i + 1) * th), slice(j * tw, (j + 1) * tw))
                        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                            pred[tile] = net(x[(..., *tile)])["out"].float().argmax(1)[0]
                conf[(n, g, backlit)] += torch.bincount(y.view(-1) * NUM_CLASSES + pred.view(-1),
                                                        minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    for n in nets:
        for g in grids:
            print(f"{n} {g}x{g}: F5+F7 {miou(conf[(n, g, True)]):.2f} | other five {miou(conf[(n, g, False)]):.2f}")


if __name__ == "__main__":
    main()
