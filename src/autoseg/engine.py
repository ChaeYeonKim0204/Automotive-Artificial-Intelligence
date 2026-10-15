"""Training step, evaluation and IoU metrics."""
import contextlib

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE, NUM_CLASSES

AUX_WEIGHT = 0.4


def dice_loss(logits, target, num_classes=NUM_CLASSES):
    """1 - soft Dice, averaged over the classes present in the batch (IGNORE pixels skipped)."""
    valid = (target != IGNORE).unsqueeze(1).float()
    prob = logits.float().softmax(1) * valid
    onehot = F.one_hot(torch.where(target == IGNORE, 0, target), num_classes).permute(0, 3, 1, 2).float() * valid
    dims = (0, 2, 3)
    inter, total = (prob * onehot).sum(dims), prob.sum(dims) + onehot.sum(dims)
    present = onehot.sum(dims) > 0
    if not present.any():  # every pixel ignored
        return logits.sum() * 0.0
    return 1.0 - ((2 * inter + 1.0) / (total + 1.0))[present].mean()


def seg_loss(out_dict, target, class_weights=None, dice_weight=0.0):
    if not (target != IGNORE).any():  # all-ignore batch: cross-entropy would be NaN
        return out_dict["out"].sum() * 0.0
    loss = F.cross_entropy(out_dict["out"], target, weight=class_weights, ignore_index=IGNORE)
    if "aux" in out_dict:  # FCN models have an auxiliary head; U-Net does not
        loss = loss + AUX_WEIGHT * F.cross_entropy(out_dict["aux"], target, weight=class_weights, ignore_index=IGNORE)
    if dice_weight:
        loss = loss + dice_weight * dice_loss(out_dict["out"], target)
    return loss


def autocast(device, amp):
    """bf16 autocast when ``amp`` is set; bf16 keeps fp32 range, so no GradScaler is needed."""
    if amp and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _to(x, y, device, channels_last):
    x = x.to(device, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    return x, y.to(device, non_blocking=True)


def train_one_epoch(model, loader, optimizer, device, class_weights=None, amp=False, channels_last=False,
                    on_step=None, dice_weight=0.0):
    """``on_step`` runs after every optimizer step (per-iteration lr schedule, EMA update)."""
    model.train()
    total_loss, correct, pixels = 0.0, 0, 0
    for x, y in loader:
        x, y = _to(x, y, device, channels_last)
        optimizer.zero_grad()
        with autocast(device, amp):
            out = model(x)
            loss = seg_loss(out, y, class_weights, dice_weight)
        loss.backward()
        optimizer.step()
        if on_step:
            on_step()
        total_loss += loss.item()
        with torch.no_grad():
            valid = y != IGNORE
            correct += ((out["out"].argmax(1) == y) & valid).sum().item()
            pixels += valid.sum().item()
    return total_loss / len(loader), correct / pixels


@torch.no_grad()
def evaluate(model, loader, device, class_weights=None, num_classes=NUM_CLASSES, amp=False, channels_last=False):
    """Returns (avg loss, pixel accuracy, per-class IoU list, mIoU, confusion matrix)."""
    model.eval()
    total_loss = 0.0
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    for x, y in loader:
        x, y = _to(x, y, device, channels_last)
        with autocast(device, amp):
            out = model(x)
            total_loss += seg_loss(out, y, class_weights).item()
        pred = out["out"].argmax(1)
        keep = y.view(-1) != IGNORE
        idx = y.view(-1)[keep] * num_classes + pred.view(-1)[keep]
        conf += torch.bincount(idx, minlength=num_classes ** 2).view(num_classes, num_classes)
    conf = conf.cpu().numpy()
    pixel_acc = np.trace(conf) / conf.sum()
    tp = np.diag(conf)
    denom = conf.sum(0) + conf.sum(1) - tp
    ious = np.where(denom > 0, tp / np.maximum(denom, 1), np.nan)
    miou = np.nanmean(ious)
    return total_loss / len(loader), pixel_acc, ious.tolist(), float(miou), conf
