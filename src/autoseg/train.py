"""Train a scratch FCN on the CARLA 13-class dataset.

Final-model recipe (experiments/chaeyeon/11_..._FINAL.ipynb):
    python -m autoseg.train --data-root "2D Segmentation" --model fcn_resnet50_v1d \
        --aug enhanced --scheduler --epochs 20 --out models/fcn_resnet50_v1d.pt
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim

from .data import CLASS_NAMES, class_pixel_ratio, class_weights_from_ratio, make_loaders
from .engine import evaluate, train_one_epoch
from .models import MODELS, build_model


def parse_crop(text):
    if "x" in text:
        h, w = text.lower().split("x")
        return int(h), int(w)
    return int(text)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="folder holding Training/ and Validation/")
    p.add_argument("--model", choices=list(MODELS), default="fcn_resnet50_v1d")
    p.add_argument("--tricks", default="", help="for fcn_resnet50_tricks: subset of b,c,d (ResNet-B/C/D)")
    p.add_argument("--aug", choices=["none", "basic", "enhanced"], default="enhanced")
    p.add_argument("--crop", type=parse_crop, default=512,
                   help="train crop: 512 (square), 576x768 (HxW), or 0 for the full frame")
    p.add_argument("--rare-weight", type=float, default=3.0, help="oversampling weight for images with a rare class")
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--scheduler", action="store_true", help="ReduceLROnPlateau on val pixel acc (factor 0.5, patience 1)")
    p.add_argument("--patience", type=int, default=5, help="early-stop patience on val pixel acc")
    p.add_argument("--select", choices=["acc", "miou"], default="acc", help="metric for checkpoint selection")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="models/model.pt")
    p.add_argument("--tb-dir", default=None, help="TensorBoard log dir (one run per subfolder)")
    p.add_argument("--run-name", default=None, help="TensorBoard run name (default: checkpoint stem)")
    # speed options (all off by default = the course pipeline)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--persistent-workers", action="store_true", help="keep DataLoader workers alive across epochs")
    p.add_argument("--cache-dir", default=None, help="decode PNGs once into memory-mapped .npy files here")
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--cudnn-benchmark", action="store_true")
    p.add_argument("--amp", action="store_true", help="bf16 autocast")
    p.add_argument("--val-batch-size", type=int, default=None, help="default: same as --batch-size")
    p.add_argument("--scale-aug", default=None, help="random rescale range before the crop, e.g. 0.75,1.5")
    p.add_argument("--dice-weight", type=float, default=0.0, help="add w * soft Dice to the main loss")
    p.add_argument("--two-frame", action="store_true", help="input = previous frame + current frame (6 channels)")
    p.add_argument("--two-frame-same", action="store_true",
                   help="6-channel control: current frame twice (no temporal information)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="by default the full training state is saved every epoch to <out>_resume.pt and a run "
                        "restarted with the same --out continues from it (deleted when the run finishes)")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (needs triton, Linux)")
    # optimisation recipe (defaults = the course recipe: Adam, no decay, no schedule)
    p.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9, help="SGD only")
    p.add_argument("--lr-schedule", choices=["none", "plateau", "cosine", "poly"], default=None,
                   help="cosine/poly step every iteration; plateau = --scheduler (default none)")
    p.add_argument("--warmup-epochs", type=float, default=0.0, help="linear warmup for cosine/poly")
    p.add_argument("--poly-power", type=float, default=0.9)
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay of weights and BN buffers (e.g. 0.999); validation then uses the EMA model")
    args = p.parse_args()
    if args.lr_schedule is None:
        args.lr_schedule = "plateau" if args.scheduler else "none"
    return args


def lr_factor(it, total, warmup, kind, power=0.9):
    """Multiplier on the base lr at iteration ``it`` (0-based) of ``total``."""
    if it < warmup:
        return (it + 1) / warmup
    p = min(1.0, (it - warmup) / max(1, total - warmup))
    if kind == "cosine":
        return 0.5 * (1.0 + math.cos(math.pi * p))
    if kind == "poly":
        return (1.0 - p) ** power
    return 1.0


class EMA:
    """Exponential moving average of weights and BN buffers, evaluated instead of the raw model.

    Decay warms up as min(decay, (1 + n) / (10 + n)) so early epochs are not
    dominated by the random initialisation (same rule as timm's ModelEmaV2 warmup).
    """

    def __init__(self, model, decay):
        import copy
        self.model, self.decay, self.n = copy.deepcopy(model).eval(), decay, 0
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        self.n += 1
        ema_p, cur_p = list(self.model.parameters()), list(model.parameters())
        torch._foreach_lerp_(ema_p, cur_p, 1.0 - d)
        for e, c in zip(self.model.buffers(), model.buffers()):
            if e.dtype.is_floating_point:
                e.lerp_(c, 1.0 - d)
            else:
                e.copy_(c)


def make_optimizer(name, params, lr, weight_decay, momentum, fused):
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, fused=fused)
    cls = optim.AdamW if name == "adamw" else optim.Adam
    return cls(params, lr=lr, weight_decay=weight_decay, fused=fused)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    train_loader, val_loader = make_loaders(args.data_root, args.batch_size, args.aug, args.crop,
                                            args.rare_weight, args.num_workers, args.pin_memory,
                                            args.persistent_workers, args.cache_dir, args.val_batch_size,
                                            scale=tuple(map(float, args.scale_aug.split(","))) if args.scale_aug else None,
                                            two_frame="prev" if args.two_frame else ("same" if args.two_frame_same else False))
    class_weights = None
    if not args.no_class_weights:
        cached = os.path.join(args.cache_dir, "train", "label.npy") if args.cache_dir else None
        ratio = class_pixel_ratio(os.path.join(args.data_root, "Training", "Label"), cached)
        class_weights = torch.tensor(class_weights_from_ratio(ratio), dtype=torch.float32, device=device)
        print("class weights:", np.round(class_weights.cpu().numpy(), 4))

    model = build_model(args.model, tricks=args.tricks,
                        in_channels=6 if (args.two_frame or args.two_frame_same) else 3).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    speed = dict(amp=args.amp, channels_last=args.channels_last)
    optimizer = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay,
                               args.momentum, fused=device.type == "cuda")
    plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1) \
        if args.lr_schedule == "plateau" else None
    per_iter = None
    if args.lr_schedule in ("cosine", "poly"):
        total, warm = args.epochs * len(train_loader), args.warmup_epochs * len(train_loader)
        per_iter = optim.lr_scheduler.LambdaLR(
            optimizer, lambda it: lr_factor(it, total, warm, args.lr_schedule, args.poly_power))
    ema = EMA(model, args.ema) if args.ema else None
    eval_model = ema.model if ema else model
    train_model = torch.compile(model) if args.compile else model

    def on_step():
        if per_iter:
            per_iter.step()
        if ema:
            ema.update(model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = None
    if args.tb_dir:
        from torch.utils.tensorboard import SummaryWriter
        run_name = args.run_name or os.path.splitext(os.path.basename(args.out))[0]
        writer = SummaryWriter(os.path.join(args.tb_dir, run_name))
        writer.add_text("args", " ".join(f"{k}={v}" for k, v in sorted(vars(args).items())))
    best, tolerance, best_ep, best_row = -1.0, 0, 0, None
    start, stopped, epoch, val_acc, miou = 1, False, 0, 0.0, 0.0
    resume_path = os.path.splitext(args.out)[0] + "_resume.pt"
    if args.resume and os.path.exists(resume_path):
        st = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        if per_iter:
            per_iter.load_state_dict(st["per_iter"])
        if plateau:
            plateau.load_state_dict(st["plateau"])
        if ema:
            ema.model.load_state_dict(st["ema"])
            ema.n = st["ema_n"]
        best, tolerance, best_ep, best_row = st["best"], st["tolerance"], st["best_ep"], st["best_row"]
        epoch, val_acc, miou, stopped = st["epoch"], st["last"][0], st["last"][1], st["stopped"]
        random.setstate(st["rng"]["python"])
        np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"].cpu())
        if device.type == "cuda" and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(st["rng"]["cuda"].cpu())
        start = epoch + 1
        print(f"resumed from {resume_path} after epoch {epoch}", flush=True)
    for epoch in range(start, args.epochs + 1):
        if stopped:
            break
        t0 = time.time()
        trn_loss, trn_acc = train_one_epoch(train_model, train_loader, optimizer, device, class_weights,
                                            on_step=on_step, dice_weight=args.dice_weight, **speed)
        t_train = time.time() - t0
        val_loss, val_acc, ious, miou, _ = evaluate(eval_model, val_loader, device, class_weights, **speed)
        if plateau:
            plateau.step(val_acc)
        score = val_acc if args.select == "acc" else miou
        print(f"Epoch {epoch:02d} | {time.time() - t0:.0f}s | trn loss {trn_loss:.4f} acc {trn_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} mIoU {miou:.4f} | train {t_train:.0f}s")
        if writer:
            writer.add_scalars("loss", {"train": trn_loss, "val": val_loss}, epoch)
            writer.add_scalars("pixel_acc", {"train": trn_acc, "val": val_acc}, epoch)
            writer.add_scalar("val/mIoU", miou, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("epoch_time_s", time.time() - t0, epoch)
            for name, iou in zip(CLASS_NAMES, ious):
                if not np.isnan(iou):
                    writer.add_scalar(f"val_iou/{name}", iou, epoch)
        if score > best:
            best, tolerance, best_ep, best_row = score, 0, epoch, (val_acc, miou)
            torch.save(eval_model.state_dict(), args.out)
            print(f"  saved -> {args.out}")
        else:
            tolerance += 1
            if tolerance >= args.patience:
                print(f"  early stop (patience {args.patience})")
                stopped = True
        if args.resume:
            state = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "per_iter": per_iter.state_dict() if per_iter else None,
                     "plateau": plateau.state_dict() if plateau else None,
                     "ema": ema.model.state_dict() if ema else None, "ema_n": ema.n if ema else 0,
                     "best": best, "tolerance": tolerance, "best_ep": best_ep, "best_row": best_row,
                     "last": (val_acc, miou), "stopped": stopped,
                     "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                             "torch": torch.get_rng_state(),
                             "cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None}}
            torch.save(state, resume_path + ".tmp")
            os.replace(resume_path + ".tmp", resume_path)  # atomic: a crash never leaves a half-written file
        if stopped:
            break
    epoch = max(epoch, start - 1)
    torch.save(eval_model.state_dict(), os.path.splitext(args.out)[0] + "_last.pt")
    # one machine-readable line per run (used by the experiment queue)
    print(f"FINAL last_ep {epoch} last_acc {val_acc:.4f} last_mIoU {miou:.4f} "
          f"sel_ep {best_ep} sel_acc {best_row[0]:.4f} sel_mIoU {best_row[1]:.4f} "
          f"peak_mem_GB {torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0:.1f}")
    if writer:
        writer.close()
    if args.resume and os.path.exists(resume_path):
        os.remove(resume_path)


if __name__ == "__main__":
    main()
