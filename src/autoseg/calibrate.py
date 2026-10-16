"""Post-hoc decision adjustment for mIoU (no retraining), checked with leave-one-sequence-out.

argmax maximises pixel accuracy, which large classes dominate; mIoU weighs every class
equally, so the mIoU-optimal decision leans towards small classes. The model was already
trained with class weights and Dice, so the right amount of extra lean (possibly negative)
is measured here. Decision rule:  argmax_c [ log p_c + b_c ]
  logit adjustment  b_c = -tau * log(prior_c), prior = training pixel share  (one value, tau)
  per-class bias    b_c free for every class, None fixed at 0 (coordinate ascent)
Every validation sequence is scored with the adjustment fitted on the other six sequences;
the pooled confusion gives the cross-validated mIoU. The search runs on log-probabilities
sampled on a stride-4 pixel grid; the chosen adjustments are then applied at full resolution.

    python -m autoseg.calibrate --data-root ... --cache-dir ... --ckpt model_last.pt --amp --out calib.json
    (also writes calib_bias.json for evaluate --bias-json)
"""
import argparse
import json
import os

import numpy as np
import torch

from .data import CLASS_NAMES, NUM_CLASSES, SegDataset, class_pixel_ratio
from .engine import autocast
from .models import MODELS, build_model

STRIDE = 4
TAUS = np.round(np.arange(-0.5, 1.0001, 0.125), 3)
KEY_CONFUSIONS = (("Other", "Buildings"), ("Walls", "Buildings"), ("Fences", "Sidewalks"),
                  ("Fences", "Vegetation"), ("RoadLines", "Roads"))


def confusion(logp, labels, bias):
    """logp (N, C, H, W), labels (N, H, W), bias (C,) -> (C, C) confusion of argmax(logp + bias)."""
    pred = (logp + bias.view(1, -1, 1, 1).to(logp.dtype)).argmax(1)
    return torch.bincount(labels.reshape(-1) * NUM_CLASSES + pred.reshape(-1),
                          minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)


def miou(conf):
    conf = conf.double()
    tp = conf.diag()
    denom = conf.sum(0) + conf.sum(1) - tp
    ious = torch.where(denom > 0, tp / denom.clamp(min=1), torch.full_like(tp, float("nan")))
    return float(torch.nanmean(ious)), ious


def fit_tau(logp, labels, log_prior):
    scores = [miou(confusion(logp, labels, -t * log_prior))[0] for t in TAUS]
    return float(TAUS[int(np.argmax(scores))])


def fit_bias(logp, labels, fixed=0):
    """Coordinate ascent on per-class biases: a coarse sweep, then a fine one around the result."""
    bias = torch.zeros(NUM_CLASSES, device=logp.device)
    best = miou(confusion(logp, labels, bias))[0]
    # +-2 (+-2.3 after the fine sweep). A +-4 search was also tried: Buildings and Poles moved past -2.3,
    # the in-sample mIoU rose, but the leave-one-sequence-out mIoU did not (fits the validation set more)
    coarse, fine = torch.arange(-2.0, 2.0001, 0.2), torch.arange(-0.3, 0.3001, 0.05)
    for grid, relative in ((coarse, False), (fine, True)):
        for c in range(NUM_CLASSES):
            if c == fixed:
                continue
            for v in (bias[c].item() + grid if relative else grid):
                trial = bias.clone()
                trial[c] = v
                s = miou(confusion(logp, labels, trial))[0]
                if s > best + 1e-9:
                    best, bias = s, trial
    return bias


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", choices=list(MODELS), default="fcn_resnet50_v1d")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = lambda s: (os.path.join(args.data_root, s, "RGB"), os.path.join(args.data_root, s, "Label"))
    cache = lambda s: os.path.join(args.cache_dir, s) if args.cache_dir else None
    ds = SegDataset(*split("Validation"), train=False, cache_dir=cache("val"))
    cached = os.path.join(args.cache_dir, "train", "label.npy") if args.cache_dir else None
    prior = class_pixel_ratio(split("Training")[1], cached)
    log_prior = torch.tensor(np.log(np.maximum(prior, 1e-8)), dtype=torch.float32, device=device)
    model = build_model(args.model).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    def log_probs(x, flip):
        with autocast(device, args.amp):
            p = model(x)["out"].float().softmax(1)
            if flip:
                p = 0.5 * (p + model(x.flip(-1))["out"].float().softmax(1).flip(-1))
        return p.clamp_min(1e-8).log()[0]

    order = list(ds.frame_order)
    seq_of = [ds.sequence[i] for i in order]
    seqs = list(dict.fromkeys(seq_of))
    variants = ("plain", "flip")

    # pass 1: stride-4 samples of log-probabilities for the search
    n, (h, w) = len(order), ds[order[0]][1].shape
    sub = {v: torch.empty(n, NUM_CLASSES, (h + STRIDE - 1) // STRIDE, (w + STRIDE - 1) // STRIDE,
                          dtype=torch.float16, device=device) for v in variants}
    lab = torch.empty(n, *sub["plain"].shape[-2:], dtype=torch.long, device=device)
    for k, i in enumerate(order):
        x, y = ds[i]
        x = x[None].to(device)
        for v in variants:
            sub[v][k] = log_probs(x, v == "flip")[:, ::STRIDE, ::STRIDE].half()
        lab[k] = y[::STRIDE, ::STRIDE].to(device)

    seq_idx = {s: torch.tensor([k for k, q in enumerate(seq_of) if q == s], device=device) for s in seqs}
    report = {"prior": dict(zip(CLASS_NAMES, prior.tolist())), "variants": {}}
    params = {}
    for v in variants:
        lp = sub[v]
        zero = torch.zeros(NUM_CLASSES, device=device)
        fits = {"all": {"tau": fit_tau(lp, lab, log_prior), "bias": fit_bias(lp, lab)}}
        for s in seqs:
            keep = torch.cat([seq_idx[o] for o in seqs if o != s])
            fits[s] = {"tau": fit_tau(lp[keep], lab[keep], log_prior), "bias": fit_bias(lp[keep], lab[keep])}
        params[v] = fits
        report["variants"][v] = {
            "subsample_mIoU": miou(confusion(lp, lab, zero))[0],
            "fitted": {  # kept apart from the per-method results below (same names otherwise collide)
                "tau_all": fits["all"]["tau"],
                "tau_per_fold": {s: fits[s]["tau"] for s in seqs},
                "bias_all": dict(zip(CLASS_NAMES, fits["all"]["bias"].tolist())),
                "bias_per_fold": {s: dict(zip(CLASS_NAMES, fits[s]["bias"].tolist())) for s in seqs},
            },
        }
        print(f"CALIB {v}: tau(all) {fits['all']['tau']:+.3f} | tau per fold "
              + " ".join(f"{s}:{fits[s]['tau']:+.2f}" for s in seqs), flush=True)

    # pass 2: full resolution; each sequence uses the adjustment fitted without it
    methods = ("none", "tau_cv", "bias_cv", "tau_all", "bias_all")
    conf = {(v, m): torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
            for v in variants for m in methods}
    for k, i in enumerate(order):
        x, y = ds[i]
        x, y, s = x[None].to(device), y.to(device), seq_of[k]
        for v in variants:
            lp = log_probs(x, v == "flip")[None]
            f = params[v]
            b = {"none": torch.zeros(NUM_CLASSES, device=device),
                 "tau_cv": -f[s]["tau"] * log_prior, "bias_cv": f[s]["bias"],
                 "tau_all": -f["all"]["tau"] * log_prior, "bias_all": f["all"]["bias"]}
            for m in methods:
                conf[(v, m)] += confusion(lp, y[None], b[m])

    for v in variants:
        for m in methods:
            c = conf[(v, m)]
            mi, ious = miou(c)
            rows = c.double() / c.sum(1, keepdim=True).clamp(min=1)
            idx = {n_: j for j, n_ in enumerate(CLASS_NAMES)}
            report["variants"][v][m] = {
                "mIoU": mi, "acc": float(c.diag().sum() / c.sum()),
                "iou": {n_: (None if torch.isnan(t) else float(t)) for n_, t in zip(CLASS_NAMES, ious)},
                "confusions": {f"{a}->{b_}": float(rows[idx[a], idx[b_]]) for a, b_ in KEY_CONFUSIONS}}
            print(f"CALIB {v:5s} {m:8s} mIoU {mi * 100:6.2f}", flush=True)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)
        # flip TTA + per-class bias fitted on all of Validation, in the format of evaluate --bias-json
        with open(os.path.splitext(args.out)[0] + "_bias.json", "w") as fh:
            json.dump({"flip": True, "bias": dict(zip(CLASS_NAMES, params["flip"]["all"]["bias"].tolist()))}, fh, indent=1)


if __name__ == "__main__":
    main()
