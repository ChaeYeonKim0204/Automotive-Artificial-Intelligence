"""Smallest check that fails if the models or the metric break.

    cd src && python -m autoseg.smoke_test
"""
import numpy as np
import torch

from .engine import evaluate
from .models import build_model


def main():
    x = torch.rand(1, 3, 96, 128)
    for name in ("fcn_resnet50", "fcn_resnet50_v1d"):
        out = build_model(name).eval()(x)
        assert out["out"].shape == (1, 13, 96, 128), (name, out["out"].shape)
        assert out["aux"].shape == (1, 13, 96, 128), (name, out["aux"].shape)
    for tricks in ("", "b", "c", "d", "b,c", "b,d", "c,d", "b,c,d"):
        out = build_model("fcn_resnet50_tricks", tricks=tricks).eval()(x)
        assert out["out"].shape == (1, 13, 96, 128), (tricks, out["out"].shape)
    # tricks "" and "c,d" must match the two course models parameter-for-parameter
    import torch as _t
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(build_model("fcn_resnet50_tricks", tricks="")) == n(build_model("fcn_resnet50"))
    assert n(build_model("fcn_resnet50_tricks", tricks="c,d")) == n(build_model("fcn_resnet50_v1d"))

    # confusion-matrix IoU against a hand-computed case
    class Fixed(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((1, 13, 2, 2), -10.0)
            logits[0, 0, 0, :] = 10.0   # predict class 0 on the top row
            logits[0, 1, 1, :] = 10.0   # predict class 1 on the bottom row
            return {"out": logits, "aux": logits}
    y = torch.tensor([[[0, 1], [1, 1]]])          # truth: one class-0 pixel, three class-1
    loader = [(torch.zeros(1, 3, 2, 2), y)]
    _, acc, ious, miou, _ = evaluate(Fixed(), loader, torch.device("cpu"))
    assert abs(acc - 0.75) < 1e-6, acc               # 3 of 4 pixels right
    assert abs(ious[0] - 0.5) < 1e-6, ious[0]        # class 0: TP 1, FP 1, FN 0
    assert abs(ious[1] - 2 / 3) < 1e-6, ious[1]      # class 1: TP 2, FP 0, FN 1
    assert np.isnan(ious[2]), ious[2]                 # absent class is NaN, not 0
    assert abs(miou - (0.5 + 2 / 3) / 2) < 1e-6, miou

    # .npy cache must return exactly what PNG decoding returns
    import os, tempfile
    from PIL import Image
    from .data import SegDataset
    with tempfile.TemporaryDirectory() as d:
        rng = np.random.default_rng(0)
        for sub in ("RGB", "Label"):
            os.makedirs(os.path.join(d, sub))
        for i in range(3):
            Image.fromarray(rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)).save(os.path.join(d, "RGB", f"{i}.png"))
            lab = np.zeros((6, 8, 3), np.uint8); lab[..., 0] = rng.integers(0, 13, (6, 8))
            Image.fromarray(lab).save(os.path.join(d, "Label", f"{i}.png"))
        args = (os.path.join(d, "RGB"), os.path.join(d, "Label"))
        png = SegDataset(*args, train=True, aug="none")
        npy = SegDataset(*args, train=True, aug="none", cache_dir=os.path.join(d, "cache"))
        assert png.has_rare == npy.has_rare
        for i in range(3):
            (a, la), (b, lb) = png[i], npy[i]
            assert torch.equal(a, b) and torch.equal(la, lb), i
        from .data import class_pixel_ratio
        assert np.array_equal(class_pixel_ratio(args[1]),
                              class_pixel_ratio(args[1], os.path.join(d, "cache", "label.npy")))
        del npy  # release memmaps before the temp dir is removed (Windows)

    # lr schedule: linear warmup to 1, then decay to ~0 at the last iteration
    from .train import lr_factor
    for kind in ("cosine", "poly"):
        f = [lr_factor(i, 100, 10, kind) for i in range(100)]
        assert abs(f[9] - 1.0) < 1e-9 and f[0] < 0.2, (kind, f[:10])
        assert all(a >= b for a, b in zip(f[10:], f[11:])), kind
        assert f[-1] < 0.05, (kind, f[-1])
    assert lr_factor(50, 100, 0, "none") == 1.0

    # EMA: first update is a near-copy (decay warmup), later updates move by (1 - decay)
    from .train import EMA
    m = torch.nn.Linear(2, 1)
    ema = EMA(m, 0.99)
    with torch.no_grad():
        m.weight.fill_(1.0)
    ema.update(m)                                    # d = 0.1 -> 0.9 of the way to 1.0
    w0 = ema.model.weight.detach().clone()
    assert torch.all((w0 - 1.0).abs() < 0.1 * 2), w0
    with torch.no_grad():
        m.weight.fill_(2.0)
    for _ in range(2000):
        ema.update(m)
    assert torch.allclose(ema.model.weight, torch.full_like(w0, 2.0), atol=1e-3), ema.model.weight

    # rectangular and full-frame crops
    from .train import parse_crop
    assert parse_crop("512") == 512 and parse_crop("576x768") == (576, 768) and parse_crop("0") == 0
    with tempfile.TemporaryDirectory() as d:
        for sub in ("RGB", "Label"):
            os.makedirs(os.path.join(d, sub))
        Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(os.path.join(d, "RGB", "0.png"))
        Image.fromarray(np.zeros((60, 80, 3), np.uint8)).save(os.path.join(d, "Label", "0.png"))
        for crop, shape in ((32, (32, 32)), ((40, 64), (40, 64)), (0, (60, 80))):
            x, y = SegDataset(os.path.join(d, "RGB"), os.path.join(d, "Label"), aug="basic", crop=crop)[0]
            assert tuple(x.shape[1:]) == shape and tuple(y.shape) == shape, (crop, x.shape, y.shape)

    # frame order: F-sequence numbers are not zero-padded
    from .data import frame_key, IGNORE
    assert frame_key("F66-12.png") == ("F66", 12) and frame_key("02_01_007.png") == ("02_01", 7)
    # scale augmentation keeps the output size, pads with IGNORE when zoomed out, two-frame gives 6 channels
    with tempfile.TemporaryDirectory() as d:
        for sub in ("RGB", "Label"):
            os.makedirs(os.path.join(d, sub))
        for i in (1, 2, 10):
            Image.fromarray(np.full((60, 80, 3), i, np.uint8)).save(os.path.join(d, "RGB", f"F1-{i}.png"))
            Image.fromarray(np.ones((60, 80, 3), np.uint8)).save(os.path.join(d, "Label", f"F1-{i}.png"))
        ds = SegDataset(os.path.join(d, "RGB"), os.path.join(d, "Label"), aug="basic", crop=0, scale=(0.5, 0.5))
        x, y = ds[0]
        assert tuple(x.shape) == (3, 60, 80) and (y == IGNORE).any() and (y == 1).any(), (x.shape, y.unique())
        ds2 = SegDataset(os.path.join(d, "RGB"), os.path.join(d, "Label"), train=False, two_frame=True)
        order = [ds2.rgb_files[i] for i in ds2.frame_order]
        assert order == ["F1-1.png", "F1-2.png", "F1-10.png"], order
        i10 = ds2.rgb_files.index("F1-10.png")
        x, _ = ds2[i10]
        assert x.shape[0] == 6 and abs(x[0].mean().item() * 255 - 2) < 0.5, x[0].mean()  # previous frame = F1-2
    # loss ignores IGNORE pixels; Dice is 0 for a perfect prediction
    from .engine import seg_loss, dice_loss  # noqa: F811
    t = torch.tensor([[[0, 1], [IGNORE, 1]]])
    logits = torch.full((1, 13, 2, 2), -20.0)
    logits[0, 0, 0, 0] = logits[0, 1, 0, 1] = logits[0, 1, 1, 1] = 20.0
    assert dice_loss(logits, t).item() < 1e-3
    assert seg_loss({"out": logits, "aux": logits}, t, dice_weight=0.5).item() < 1e-3
    # 6-channel control repeats the current frame; clips share one augmentation across frames
    with tempfile.TemporaryDirectory() as d:
        for sub in ("RGB", "Label"):
            os.makedirs(os.path.join(d, sub))
        for i in range(1, 5):
            Image.fromarray(np.full((60, 80, 3), 10 * i, np.uint8)).save(os.path.join(d, "RGB", f"S-{i}.png"))
            lab = np.zeros((60, 80, 3), np.uint8); lab[:, :40, 0] = i
            Image.fromarray(lab).save(os.path.join(d, "Label", f"S-{i}.png"))
        rgb_dir, lab_dir = os.path.join(d, "RGB"), os.path.join(d, "Label")
        same = SegDataset(rgb_dir, lab_dir, train=False, two_frame="same")
        x, _ = same[0]
        assert x.shape[0] == 6 and torch.equal(x[:3], x[3:])
        clip_ds = SegDataset(rgb_dir, lab_dir, aug="basic", crop=0, scale=(0.5, 0.5))
        idx4 = clip_ds.rgb_files.index("S-4.png")
        assert [clip_ds.rgb_files[i] for i in clip_ds.clip_indices(idx4, 3)] == ["S-2.png", "S-3.png", "S-4.png"]
        offsets = set()
        for _ in range(20):
            xc, yc = clip_ds.get_clip(idx4, 3)
            assert tuple(xc.shape) == (3, 3, 60, 80) and tuple(yc.shape) == (3, 60, 80)
            ign = [(yc[t] == IGNORE) for t in range(3)]
            assert all(torch.equal(ign[0], m) for m in ign), "padding must be shared by every frame in a clip"
            rows = torch.nonzero(~ign[0])[:, 0]
            offsets.add(int(rows.min()))
        assert len(offsets) > 1, "zoom-out paste offset should be random"
    # loss on an all-ignore batch is 0, not NaN
    z = seg_loss({"out": torch.zeros(1, 13, 2, 2, requires_grad=True), "aux": torch.zeros(1, 13, 2, 2)},
                 torch.full((1, 2, 2), IGNORE), dice_weight=0.5)
    assert z.item() == 0.0
    # 6-channel models build and run
    for name in ("fcn_resnet50", "fcn_resnet50_v1d"):
        out = build_model(name, in_channels=6).eval()(torch.rand(1, 6, 64, 64))
        assert out["out"].shape == (1, 13, 64, 64)
    print("smoke test OK")


if __name__ == "__main__":
    main()
