"""Fine-tune a temporal head (ConvGRU) or its matched control on short clips.

Starts from a trained per-frame checkpoint; the backbone stays frozen.
Both modes see the same clips, augmentation, optimiser and schedule; only the
recurrent branch differs. Evaluation on Validation reports
  window: the model sees the same T-frame window as in training (t-T+1 .. t)
  stream: the hidden state is carried through the whole sequence

    python -m autoseg.train_temporal --data-root ... --cache-dir ... --base-ckpt best_last.pt --mode gru
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .data import IGNORE, NUM_CLASSES, SegDataset, class_pixel_ratio, class_weights_from_ratio
from .engine import autocast
from .models import build_model
from .models.temporal import TemporalFCN
from .train import lr_factor


class Clips(Dataset):
    def __init__(self, ds, length):
        self.ds, self.length = ds, length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.get_clip(idx, self.length)


def iou_table(conf):
    conf = conf.cpu().numpy()
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    ious = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    return float(np.nanmean(ious)), float(tp.sum() / conf.sum())


@torch.no_grad()
def evaluate_temporal(model, ds, device, length, amp):
    model.eval()
    conf = {k: torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64, device=device) for k in ("window", "stream")}
    seqs = {}
    for i in ds.frame_order:
        seqs.setdefault(ds.sequence[i], []).append(i)
    for frames in seqs.values():
        recent, h_stream = [], None
        for idx in frames:
            x, y = ds[idx]
            x, y = x[None].to(device), y.to(device)
            with autocast(device, amp):
                z = model.encode(x)
                recent = (recent + [z])[-length:]
                window = [recent[0]] * (length - len(recent)) + recent  # sequence start repeats frame 0
                h = None
                for zz in window:
                    f, h = model.fuse(zz, h)
                pred_w = model.decode(f, y.shape[-2:]).argmax(1)[0]
                fs, h_stream = model.fuse(z, h_stream)
                pred_s = model.decode(fs, y.shape[-2:]).argmax(1)[0]
            for k, pred in (("window", pred_w), ("stream", pred_s)):
                conf[k] += torch.bincount(y.view(-1) * NUM_CLASSES + pred.view(-1),
                                          minlength=NUM_CLASSES ** 2).view(NUM_CLASSES, NUM_CLASSES)
    return {k: iou_table(v) for k, v in conf.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", default="fcn_resnet50_v1d", choices=["fcn_resnet50_v1d"])
    p.add_argument("--base-ckpt", required=True)
    p.add_argument("--mode", choices=["gru", "none"], required=True)
    p.add_argument("--clip", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--gru-lr-mult", type=float, default=10.0, help="lr multiplier for the recurrent branch")
    p.add_argument("--warmup-epochs", type=float, default=0.5)
    p.add_argument("--aug", default="enhanced")
    p.add_argument("--crop", type=int, default=0)
    p.add_argument("--scale-aug", default=None)
    p.add_argument("--two-frame", action="store_true")
    p.add_argument("--two-frame-same", action="store_true")
    p.add_argument("--rare-weight", type=float, default=3.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    two = "prev" if args.two_frame else ("same" if args.two_frame_same else False)
    scale = tuple(map(float, args.scale_aug.split(","))) if args.scale_aug else None
    split = lambda s: (os.path.join(args.data_root, s, "RGB"), os.path.join(args.data_root, s, "Label"))
    cache = lambda s: os.path.join(args.cache_dir, s) if args.cache_dir else None
    train_ds = SegDataset(*split("Training"), train=True, aug=args.aug, crop=args.crop, cache_dir=cache("train"),
                          scale=scale, two_frame=two)
    val_ds = SegDataset(*split("Validation"), train=False, cache_dir=cache("val"), two_frame=two)
    weights = [args.rare_weight if r else 1.0 for r in train_ds.has_rare]
    loader = DataLoader(Clips(train_ds, args.clip), batch_size=args.batch_size, num_workers=args.num_workers,
                        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True),
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    cached = os.path.join(args.cache_dir, "train", "label.npy") if args.cache_dir else None
    cw = torch.tensor(class_weights_from_ratio(class_pixel_ratio(split("Training")[1], cached)),
                      dtype=torch.float32, device=device)

    base = build_model(args.model, in_channels=6 if two else 3)
    base.load_state_dict(torch.load(args.base_ckpt, map_location="cpu"))
    model = TemporalFCN(base, mode=args.mode).to(device)
    # head (shared by both modes) at --lr; the zero-initialised recurrent branch at --gru-lr-mult x lr,
    # its output projection without weight decay (decay would pull it back to the zero start)
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    head = [p for n, p in named if n.startswith(("head_pre.", "head_out."))]
    gru = [p for n, p in named if n.startswith(("proj.", "cell."))]
    out = [p for n, p in named if n.startswith("out.")]
    assert len(head) + len(gru) + len(out) == len(named)
    groups = [{"params": head, "lr": args.lr, "weight_decay": args.weight_decay}]
    if gru:
        groups += [{"params": gru, "lr": args.lr * args.gru_lr_mult, "weight_decay": args.weight_decay},
                   {"params": out, "lr": args.lr * args.gru_lr_mult, "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=args.lr, fused=device.type == "cuda")
    decay, no_decay = head + gru, out
    total, warm = args.epochs * len(loader), args.warmup_epochs * len(loader)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: lr_factor(it, total, warm, "cosine"))
    n_train = sum(p.numel() for p in decay + no_decay)
    print(f"mode {args.mode}: trainable params {n_train / 1e6:.2f}M")

    before = evaluate_temporal(model, val_ds, device, args.clip, args.amp)
    print(f"before fine-tune: window mIoU {before['window'][0]:.4f} stream mIoU {before['stream'][0]:.4f}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast(device, args.amp):
                out = model(x)  # (B, T, K, H, W)
                k = out.shape[2]
                loss = F.cross_entropy(out.reshape(-1, k, *out.shape[-2:]), y.reshape(-1, *y.shape[-2:]),
                                       weight=cw, ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item()
        msg = f"Epoch {epoch:02d} | trn loss {total_loss / len(loader):.4f}"
        if args.mode == "gru":
            msg += f" | out-proj |w| {model.out.weight.abs().mean().item():.2e}"
        print(msg, flush=True)
    res = evaluate_temporal(model, val_ds, device, args.clip, args.amp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"TEMPORAL mode {args.mode} window_mIoU {res['window'][0]:.4f} window_acc {res['window'][1]:.4f} "
          f"stream_mIoU {res['stream'][0]:.4f} stream_acc {res['stream'][1]:.4f} "
          f"before_window_mIoU {before['window'][0]:.4f}")


if __name__ == "__main__":
    main()
