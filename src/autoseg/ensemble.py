"""Evaluation-only ensembles: equal-weight mean of the members' softmax on the Validation split.

Members are loaded once; each --combo is scored with and without flip TTA, overall and per sequence.
No weights are fitted, so nothing is tuned on Validation.

    python -m autoseg.ensemble --data-root ... --cache-dir ... --amp \
        --member F42=fcn_resnet50_v1d:fin_s42_last.pt --member U42=unet:u_fin_s42_last.pt \
        --combo F42 --combo U42 --combo F42+U42 --out ensemble.json
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import NUM_CLASSES, SegDataset
from .engine import autocast
from .models import build_model


def mean_probs(probs, names):
    return torch.stack([probs[n] for n in names]).mean(0)


def iou_table(conf):
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    return float(np.nanmean(np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--member", action="append", required=True, help="NAME=MODEL:CKPT")
    p.add_argument("--combo", action="append", required=True, help="member names joined by +")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SegDataset(os.path.join(args.data_root, "Validation", "RGB"), os.path.join(args.data_root, "Validation", "Label"),
                    train=False, cache_dir=os.path.join(args.cache_dir, "val") if args.cache_dir else None)
    models = {}
    for m in args.member:
        name, rest = m.split("=", 1)
        arch, ckpt = rest.split(":", 1)
        models[name] = build_model(arch).to(device).eval()
        models[name].load_state_dict(torch.load(ckpt, map_location=device))
    combos = [c.split("+") for c in args.combo]
    assert all(n in models for c in combos for n in c), "every combo member needs a --member"

    def softmax(model, x):
        with autocast(device, args.amp):
            return model(x)["out"].float().softmax(1)[0]

    conf = {}  # (combo, variant, sequence) -> confusion
    for i in ds.frame_order:
        x, y = ds[i]
        x, y, seq = x[None].to(device), y.to(device), ds.sequence[i]
        plain = {n: softmax(m, x) for n, m in models.items()}
        flip = {n: 0.5 * (plain[n] + softmax(m, x.flip(-1)).flip(-1)) for n, m in models.items()}
        for c in combos:
            for v, probs in (("plain", plain), ("flip", flip)):
                pred = mean_probs(probs, c).argmax(0)
                key = ("+".join(c), v, seq)
                if key not in conf:
                    conf[key] = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                conf[key] += torch.bincount(y.view(-1) * NUM_CLASSES + pred.view(-1),
                                            minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)

    report = {}
    for c in dict.fromkeys(k[0] for k in conf):
        for v in ("plain", "flip"):
            per_seq = {k[2]: conf[k].cpu().numpy() for k in conf if k[:2] == (c, v)}
            total = sum(per_seq.values())
            report.setdefault(c, {})[v] = {"mIoU": iou_table(total),
                                          "per_sequence_confusion": {s: m.tolist() for s, m in per_seq.items()}}
            print(f"ENSEMBLE {c:20s} {v:5s} mIoU {iou_table(total) * 100:6.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f)


if __name__ == "__main__":
    main()
