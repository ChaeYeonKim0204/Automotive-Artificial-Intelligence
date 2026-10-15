"""Dataset, label handling, class statistics and augmentation.

Data layout (course-provided, not redistributed):
    <root>/Training/RGB/*.png, <root>/Training/Label/*.png
    <root>/Validation/RGB/*.png, <root>/Validation/Label/*.png
Labels are 800x600 PNGs whose first channel holds the class id (0..12).
"""
import os
import random
import re

# PIL before torchvision: torchvision 0.24 loads its own image DLLs first and
# Pillow 12 then fails with "DLL load failed while importing _imaging" on Windows.
from PIL import Image, ImageFilter  # noqa: I001
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

NUM_CLASSES = 13
CLASS_NAMES = [
    "None", "Buildings", "Fences", "Other", "Pedestrians", "Poles", "RoadLines",
    "Roads", "Sidewalks", "Vegetation", "Vehicles", "Walls", "TrafficSigns",
]
PALETTE = np.array([
    [0, 0, 0], [70, 70, 70], [190, 153, 153], [150, 120, 90], [220, 20, 60],
    [153, 153, 153], [255, 255, 0], [128, 64, 128], [244, 35, 232], [107, 142, 35],
    [0, 0, 142], [102, 102, 156], [220, 220, 0],
], dtype=np.uint8)
# Classes treated as "rare" for oversampling and stronger augmentation.
# Only Pedestrians (4) is rare per image (18% of images); RoadLines (6) and
# TrafficSigns (12) appear in 97% / 95% of images, so almost every image
# counts as rare and the x3 sampler barely changes the distribution.
# Kept as the final experiment used it (see docs/postmortem.md).
RARE_CLASSES = {4, 6, 12}
IGNORE = 255  # label value for padding added by scale augmentation; skipped by loss and metrics


def frame_key(name):
    """("02_01", 7) for 02_01_007.png and ("F66", 12) for F66-12.png.

    F-sequence numbers are not zero-padded, so sorted() file order is not frame order.
    """
    m = re.match(r"^(.*?)[_-](\d+)\.png$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def load_label(path):
    raw = np.array(Image.open(path))
    return raw[:, :, 0] if raw.ndim == 3 else raw


def colorize_mask(mask):
    return PALETTE[mask]


def class_pixel_ratio(label_dir, cached=None):
    """Pixel share of every class over all label files (fractions summing to 1).

    ``cached``: optional label.npy from the decode cache (same values, no PNG decoding).
    """
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    masks = np.load(cached, mmap_mode="r") if cached and os.path.exists(cached) else \
        (load_label(os.path.join(label_dir, n)) for n in sorted(os.listdir(label_dir)))
    for mask in masks:
        counts += np.bincount(np.asarray(mask).ravel(), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return counts / counts.sum()


def class_weights_from_ratio(ratio):
    """1 / log(1.02 + p), normalised to mean 1 (ENet-style weighting)."""
    w = 1.0 / np.log(1.02 + ratio)
    return w / w.mean()


class SegDataset(Dataset):
    """RGB + label pairs. ``aug`` is one of none | basic | enhanced.

    basic:    random horizontal flip + 512x512 random crop
    enhanced: flip, brightness/contrast jitter, Gaussian blur (rare images
              only) and crop, applied more often to images that contain a
              rare class (aug_factor 1.0 vs 0.4)
    Validation uses the full 800x600 frame with no augmentation.
    ``crop``: int for a square crop, (h, w) for a rectangle, 0 for the full frame.
    ``scale``: (lo, hi) random rescale before the crop; padding is labelled IGNORE.
    ``two_frame``: "prev" stacks the previous frame of the same sequence on the
    current one (6 channels; the first frame of a sequence repeats itself);
    "same" stacks the current frame twice (6-channel control without temporal
    information); False/None gives the usual 3-channel input.
    """

    def __init__(self, rgb_dir, label_dir, train=True, aug="enhanced", crop=512, cache_dir=None,
                 scale=None, two_frame=False):
        self.rgb_dir, self.label_dir = rgb_dir, label_dir
        if two_frame is True:
            two_frame = "prev"
        assert two_frame in (False, None, "prev", "same"), two_frame
        self.train, self.aug, self.crop, self.scale, self.two_frame = train, aug, crop, scale, two_frame
        self.rgb_files = sorted(os.listdir(rgb_dir))
        self.label_files = sorted(os.listdir(label_dir))
        assert self.rgb_files == self.label_files, "RGB and Label file names must pair 1:1"
        keys = [frame_key(n) for n in self.rgb_files]
        self.frame_order = sorted(range(len(keys)), key=lambda i: keys[i])
        self.prev = list(range(len(keys)))
        for a, b in zip(self.frame_order, self.frame_order[1:]):
            if keys[a][0] == keys[b][0]:
                self.prev[b] = a
        self.sequence = [k[0] for k in keys]
        self.cache_dir, self._rgb, self._lab = cache_dir, None, None
        if cache_dir:
            self._build_cache()
        self.has_rare = [False] * len(self.label_files)
        if train:
            labels = np.load(os.path.join(cache_dir, "label.npy"), mmap_mode="r") if cache_dir else None
            for i, name in enumerate(self.label_files):
                mask = labels[i] if cache_dir else load_label(os.path.join(label_dir, name))
                self.has_rare[i] = bool(np.isin(mask, list(RARE_CLASSES)).any())

    def _build_cache(self):
        """Decode every PNG once into two uint8 .npy files (memory-mapped later).

        Worker processes then read raw arrays instead of decoding PNGs every epoch.
        The memmaps are opened lazily in __getitem__ so they are not pickled into
        Windows spawn workers.
        """
        rgb_path = os.path.join(self.cache_dir, "rgb.npy")
        lab_path = os.path.join(self.cache_dir, "label.npy")
        if os.path.exists(rgb_path) and os.path.exists(lab_path):
            return
        os.makedirs(self.cache_dir, exist_ok=True)
        first = np.array(Image.open(os.path.join(self.rgb_dir, self.rgb_files[0])).convert("RGB"))
        n, (h, w, _) = len(self.rgb_files), first.shape
        rgb = np.lib.format.open_memmap(rgb_path + ".tmp", "w+", np.uint8, (n, h, w, 3))
        lab = np.lib.format.open_memmap(lab_path + ".tmp", "w+", np.uint8, (n, h, w))
        for i, (rf, lf) in enumerate(zip(self.rgb_files, self.label_files)):
            rgb[i] = np.array(Image.open(os.path.join(self.rgb_dir, rf)).convert("RGB"))
            lab[i] = load_label(os.path.join(self.label_dir, lf))
        rgb.flush(), lab.flush()
        del rgb, lab
        os.replace(rgb_path + ".tmp", rgb_path)
        os.replace(lab_path + ".tmp", lab_path)

    def __len__(self):
        return len(self.rgb_files)

    def _load_rgb(self, idx):
        if self.cache_dir:
            return Image.fromarray(np.asarray(self._rgb[idx]))
        return Image.open(os.path.join(self.rgb_dir, self.rgb_files[idx])).convert("RGB")

    def _open(self):
        if self.cache_dir and self._rgb is None:
            self._rgb = np.load(os.path.join(self.cache_dir, "rgb.npy"), mmap_mode="r")
            self._lab = np.load(os.path.join(self.cache_dir, "label.npy"), mmap_mode="r")

    def _load_label(self, idx):
        if self.cache_dir:
            return np.array(self._lab[idx])
        return load_label(os.path.join(self.label_dir, self.label_files[idx]))

    def _frame_inputs(self, idx):
        """RGB image(s) that form the network input for frame ``idx``."""
        rgbs = [self._load_rgb(idx)]
        if self.two_frame == "prev":
            rgbs.insert(0, self._load_rgb(self.prev[idx]))
        elif self.two_frame == "same":
            rgbs.insert(0, rgbs[0].copy())
        return rgbs

    def __getitem__(self, idx):
        self._open()
        rgbs, label = self._frame_inputs(idx), self._load_label(idx)
        if self.train and self.aug != "none":
            rgbs, (label,) = self._augment(rgbs, [label], self.has_rare[idx])
        return torch.cat([TF.to_tensor(r) for r in rgbs]), torch.tensor(label, dtype=torch.long)

    def clip_indices(self, idx, length):
        """``length`` consecutive frames ending at ``idx`` (sequence starts repeat the first frame)."""
        out = [idx]
        for _ in range(length - 1):
            out.insert(0, self.prev[out[0]])
        return out

    def get_clip(self, idx, length):
        """(T, C, H, W) inputs and (T, H, W) labels; one shared augmentation for the whole clip."""
        self._open()
        frames = self.clip_indices(idx, length)
        per = [self._frame_inputs(i) for i in frames]
        n_in = len(per[0])
        rgbs = [r for group in per for r in group]
        labels = [self._load_label(i) for i in frames]
        if self.train and self.aug != "none":
            rgbs, labels = self._augment(rgbs, labels, self.has_rare[idx])
        x = torch.stack([torch.cat([TF.to_tensor(r) for r in rgbs[k * n_in:(k + 1) * n_in]])
                         for k in range(length)])
        y = torch.stack([torch.tensor(l, dtype=torch.long) for l in labels])
        return x, y

    def _augment(self, rgbs, labels, is_rare):
        """One random transform shared by every image in ``rgbs`` and every mask in ``labels``."""
        f = 1.0 if (self.aug == "basic" or is_rare) else 0.4
        if random.random() < 0.5 * f:
            rgbs = [TF.hflip(r) for r in rgbs]
            labels = [np.fliplr(l).copy() for l in labels]
        if self.aug == "enhanced":
            if random.random() < 0.3 * f:
                b, c = 0.7 + 0.6 * random.random(), 0.7 + 0.6 * random.random()
                rgbs = [TF.adjust_contrast(TF.adjust_brightness(r, b), c) for r in rgbs]
            if is_rare and random.random() < 0.15 * f:
                rgbs = [r.filter(ImageFilter.GaussianBlur(radius=1.5)) for r in rgbs]
        h, w = labels[0].shape
        ch, cw = (self.crop, self.crop) if isinstance(self.crop, int) else self.crop
        if self.scale:
            s = random.uniform(*self.scale)
            nh, nw = round(h * s), round(w * s)
            rgbs = [r.resize((nw, nh), Image.BILINEAR) for r in rgbs]
            labels = [np.array(Image.fromarray(l).resize((nw, nh), Image.NEAREST)) for l in labels]
            if not (ch and cw):
                ch, cw = h, w  # full-frame training: crop back to the original size
            ph, pw = max(ch, nh), max(cw, nw)
            if (ph, pw) != (nh, nw):
                # zoomed out: paste at a random offset (no fixed black border), padding is IGNORE
                oy, ox = random.randint(0, ph - nh), random.randint(0, pw - nw)
                padded = []
                for r in rgbs:
                    canvas = Image.new("RGB", (pw, ph))
                    canvas.paste(r, (ox, oy))
                    padded.append(canvas)
                rgbs = padded
                new_labels = []
                for l in labels:
                    lab = np.full((ph, pw), IGNORE, dtype=l.dtype)
                    lab[oy:oy + nh, ox:ox + nw] = l
                    new_labels.append(lab)
                labels = new_labels
            h, w = labels[0].shape
        if ch and cw and h >= ch and w >= cw:
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            rgbs = [TF.crop(r, top, left, ch, cw) for r in rgbs]
            labels = [l[top:top + ch, left:left + cw] for l in labels]
        return rgbs, labels


def make_loaders(root, batch_size=4, aug="enhanced", crop=512, rare_weight=3.0, num_workers=0,
                 pin_memory=False, persistent_workers=False, cache_dir=None, val_batch_size=None,
                 scale=None, two_frame=False):
    """``cache_dir``: decode PNGs once into <cache_dir>/{train,val}/*.npy (see SegDataset)."""
    cache = (lambda split: os.path.join(cache_dir, split)) if cache_dir else (lambda split: None)
    train_ds = SegDataset(os.path.join(root, "Training", "RGB"), os.path.join(root, "Training", "Label"),
                          train=True, aug=aug, crop=crop, cache_dir=cache("train"), scale=scale,
                          two_frame=two_frame)
    val_ds = SegDataset(os.path.join(root, "Validation", "RGB"), os.path.join(root, "Validation", "Label"),
                        train=False, cache_dir=cache("val"), two_frame=two_frame)
    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=persistent_workers and num_workers > 0)
    if rare_weight and rare_weight != 1.0:
        weights = [rare_weight if r else 1.0 for r in train_ds.has_rare]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_loader = DataLoader(train_ds, sampler=sampler, **kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **{**kw, "batch_size": val_batch_size or batch_size})
    return train_loader, val_loader
