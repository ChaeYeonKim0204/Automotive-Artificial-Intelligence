"""Evaluation-only analysis of a checkpoint on the Validation split (no retraining).

Single-frame variants
  plain          one forward pass
  flip           mean softmax of the image and its horizontal flip
  ms             mean softmax over scales 0.75 / 1 / 1.25, each with its flip
Temporal variants (use neighbouring frames of the same sequence; reported as video results)
  mean_causal    mean softmax of t-1 and t
  mean           mean softmax of t-1, t, t+1
  flow_causal    t-1 softmax warped into frame t with DIS optical flow and averaged with t
                 where the flow passes a forward-backward consistency check
  flow           the same with t-1 and t+1
  ms_flow_causal flow_causal on the multi-scale softmax
Also reported: flicker (share of pixels whose predicted label differs from the previous
frame while the ground truth does not), raw and in flow-aligned coordinates;
per-sequence mIoU; the most frequent confusions.

    python -m autoseg.analyze --data-root ... --cache-dir ... --ckpt model_last.pt --amp --out report.json
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data import CLASS_NAMES, NUM_CLASSES, SegDataset
from .engine import autocast
from .models import MODELS, build_model

VARIANTS = ("plain", "flip", "ms", "mean_causal", "mean", "flow_causal", "flow", "ms_flow_causal")
SCALES = (0.75, 1.25)


def iou_table(conf):
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    ious = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return float(np.nanmean(ious)), float(tp.sum() / max(conf.sum(), 1)), ious


def add_conf(conf, gt, pred):
    conf += torch.bincount(gt.reshape(-1) * NUM_CLASSES + pred.reshape(-1),
                           minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)


class Flow:
    """DIS optical flow. ``calc(a, b)`` returns f with a(x) ~ b(x + f(x))."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    def calc(self, a, b):
        return self.dis.calc(a, b, None)


def to_gray(x):
    rgb = (x[-3:].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)  # last 3 channels = current frame
    return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))


def sampling_grid(flow, device):
    """Grid that samples a neighbour at x + flow(x), plus the in-bounds mask."""
    f = torch.from_numpy(flow).to(device)
    h, w = f.shape[:2]
    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32), indexing="ij")
    sx, sy = xs + f[..., 0], ys + f[..., 1]
    inside = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    grid = torch.stack([2 * sx / (w - 1) - 1, 2 * sy / (h - 1) - 1], -1)[None]
    return grid, inside


def warp(t, grid, mode="bilinear"):
    """t: (C, H, W) on the neighbour's pixel grid -> (C, H, W) on the current frame."""
    return F.grid_sample(t[None].float(), grid, mode=mode, padding_mode="border", align_corners=True)[0]


def consistent(f_cn, f_nc, grid, inside, alpha=0.01, beta=0.5):
    """Forward-backward check (Sundaram et al., 2010): f_cn(x) + f_nc(x + f_cn(x)) ~ 0."""
    fc = torch.from_numpy(f_cn).permute(2, 0, 1).to(grid.device)
    fn = warp(torch.from_numpy(f_nc).permute(2, 0, 1).to(grid.device), grid)
    err = ((fc + fn) ** 2).sum(0)
    return inside & (err < alpha * ((fc ** 2).sum(0) + (fn ** 2).sum(0)) + beta)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", choices=list(MODELS), default="fcn_resnet50_v1d")
    p.add_argument("--tricks", default="")
    p.add_argument("--two-frame", action="store_true")
    p.add_argument("--two-frame-same", action="store_true")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", default=None, help="optional JSON output path")
    p.add_argument("--per-frame", default=None, help="optional .npz with the confusion matrix of every frame and variant")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    two = "prev" if args.two_frame else ("same" if args.two_frame_same else False)
    ds = SegDataset(os.path.join(args.data_root, "Validation", "RGB"), os.path.join(args.data_root, "Validation", "Label"),
                    train=False, cache_dir=os.path.join(args.cache_dir, "val") if args.cache_dir else None, two_frame=two)
    model = build_model(args.model, tricks=args.tricks, in_channels=6 if two else 3).to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    flow = Flow()

    def softmax(x):
        with autocast(device, args.amp):
            return model(x)["out"].float().softmax(1)

    def frame_probs(x):
        size = x.shape[-2:]
        p_plain = softmax(x)
        p_flip = softmax(x.flip(-1)).flip(-1)
        ms = [p_plain, p_flip]
        for s in SCALES:
            xs = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
            for flip in (False, True):
                q = softmax(xs.flip(-1) if flip else xs)
                q = q.flip(-1) if flip else q
                ms.append(F.interpolate(q, size=size, mode="bilinear", align_corners=False))
        return {"plain": p_plain[0], "flip": (0.5 * (p_plain + p_flip))[0], "ms": torch.stack(ms).mean(0)[0]}

    conf = {v: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for v in VARIANTS}
    seq_conf = {}
    flick = {k: [0, 0] for k in ("plain_raw", "plain_aligned", "flow_causal_aligned", "gt_aligned")}

    frame_names, per_frame = [], {v: [] for v in VARIANTS}
    seqs = {}
    for i in ds.frame_order:
        seqs.setdefault(ds.sequence[i], []).append(i)
    for seq, frames in seqs.items():
        seq_conf[seq] = {v: torch.zeros_like(conf[v]) for v in VARIANTS}
        win, last = [], None  # win: last 3 frame entries; last: (entry, preds) of the previous finalised frame
        for k in range(len(frames) + 1):
            if k < len(frames):
                x, y = ds[frames[k]]
                e = {"gt": y.to(device), "gray": to_gray(x), "idx": frames[k], **frame_probs(x[None].to(device))}
                if win:
                    win[-1]["to_next"] = flow.calc(win[-1]["gray"], e["gray"])
                    e["to_prev"] = flow.calc(e["gray"], win[-1]["gray"])
                win = (win + [e])[-3:]
            if k == 0:
                continue
            c = len(win) - 2 if k < len(frames) else len(win) - 1
            cur = win[c]
            prev = win[c - 1] if c >= 1 and "to_prev" in cur else None
            nxt = win[c + 1] if c + 1 < len(win) else None

            def aligned(n, key_cn, key_nc):
                """Sampling grid from the current frame into neighbour n, and its reliability mask."""
                grid, inside = sampling_grid(cur[key_cn], device)
                return grid, consistent(cur[key_cn], n[key_nc], grid, inside).float()

            preds = {v: cur[v].argmax(0) for v in ("plain", "flip", "ms")}
            neigh = [w for w in (prev, nxt) if w is not None]
            preds["mean"] = torch.stack([cur["plain"]] + [w["plain"] for w in neigh]).mean(0).argmax(0)
            preds["mean_causal"] = (0.5 * (cur["plain"] + prev["plain"]) if prev else cur["plain"]).argmax(0)
            fused = {"flow_causal": [cur["plain"], 1.0], "flow": [cur["plain"], 1.0], "ms_flow_causal": [cur["ms"], 1.0]}
            if prev is not None:
                grid_p, m_p = aligned(prev, "to_prev", "to_next")
                wp, wpm = warp(prev["plain"], grid_p), warp(prev["ms"], grid_p)
                for key, w in (("flow_causal", wp), ("flow", wp), ("ms_flow_causal", wpm)):
                    fused[key][0] = fused[key][0] + m_p * w
                    fused[key][1] = fused[key][1] + m_p
            if nxt is not None:
                grid_n, m_n = aligned(nxt, "to_next", "to_prev")
                fused["flow"][0] = fused["flow"][0] + m_n * warp(nxt["plain"], grid_n)
                fused["flow"][1] = fused["flow"][1] + m_n
            for key, (s, wsum) in fused.items():
                preds[key] = (s / wsum).argmax(0)
            for v in VARIANTS:
                add_conf(conf[v], cur["gt"], preds[v])
                add_conf(seq_conf[seq][v], cur["gt"], preds[v])
            if args.per_frame:
                frame_names.append(ds.rgb_files[cur["idx"]])
                for v in VARIANTS:
                    c = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device)
                    add_conf(c, cur["gt"], preds[v])
                    per_frame[v].append(c.cpu().numpy())

            if last is not None and prev is not None:
                prev_entry, prev_preds = last
                same_raw = cur["gt"] == prev_entry["gt"]
                flick["plain_raw"][0] += ((preds["plain"] != prev_preds["plain"]) & same_raw).sum().item()
                flick["plain_raw"][1] += same_raw.sum().item()
                gt_w = warp(prev_entry["gt"][None].float(), grid_p, mode="nearest")[0].long()
                ok = m_p.bool()
                flick["gt_aligned"][0] += ((gt_w != cur["gt"]) & ok).sum().item()
                flick["gt_aligned"][1] += ok.sum().item()
                same = ok & (gt_w == cur["gt"])
                for name, v in (("plain_aligned", "plain"), ("flow_causal_aligned", "flow_causal")):
                    pw = warp(prev_preds[v][None].float(), grid_p, mode="nearest")[0].long()
                    flick[name][0] += ((pw != preds[v]) & same).sum().item()
                    flick[name][1] += same.sum().item()
            last = (cur, preds)

    if args.per_frame:
        np.savez_compressed(args.per_frame, names=np.array(frame_names), **{v: np.stack(c) for v, c in per_frame.items()})
    report = {"variants": {}, "flicker": {}, "per_sequence": {}}
    for v in VARIANTS:
        miou, acc, ious = iou_table(conf[v].cpu().numpy())
        report["variants"][v] = {"mIoU": miou, "acc": acc,
                                 "iou": {n: (None if np.isnan(i) else float(i)) for n, i in zip(CLASS_NAMES, ious)}}
        print(f"ANALYZE {v:15s} mIoU {miou * 100:6.2f}  acc {acc * 100:6.2f}")
    for k, (a, b) in flick.items():
        report["flicker"][k] = a / max(b, 1)
        print(f"FLICKER {k:20s} {a / max(b, 1) * 100:6.2f}%")
    for seq, cs in seq_conf.items():
        report["per_sequence"][seq] = {v: iou_table(cs[v].cpu().numpy())[0] for v in ("plain", "ms", "flow_causal")}
    print("PER-SEQ plain mIoU:", " ".join(f"{s} {r['plain'] * 100:.1f}" for s, r in report["per_sequence"].items()))
    c = conf["plain"].cpu().numpy().astype(float)
    rows = c / np.maximum(c.sum(1, keepdims=True), 1)
    off = sorted(((rows[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(NUM_CLASSES)
                  for j in range(NUM_CLASSES) if i != j), reverse=True)[:8]
    report["top_confusions"] = [{"true": t, "pred": q, "share": float(s)} for s, t, q in off]
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
