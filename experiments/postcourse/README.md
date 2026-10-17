# Post-course experiments (2026-10)

Commands behind `results/postcourse_runs.csv`, `results/postcourse_class_iou.csv` and `results/postcourse_backlit.csv`. Run from `src/` unless noted.
Data: the course `2D Segmentation` folder (`Training/` and `Validation/`, each with `RGB/` and `Label/`).

All runs share the course pipeline: enhanced augmentation, class-weighted cross-entropy, ×3 oversampling of images with a rare class.
There is no separate test set, so Validation was not used for training; it was used to select checkpoints and hyperparameters and to fit the decision bias. All reported numbers are on Validation. The synthetic glare was designed by comparing with the backlit Validation drives F5 / F7.

## Common flags (speed only, no effect on the recipe)

```bash
DATA="--data-root /path/2D_Segmentation --cache-dir /path/cache"
FAST="--num-workers 4 --pin-memory --persistent-workers --channels-last --cudnn-benchmark --amp"
```

`--cache-dir` decodes the PNGs once into memory-mapped `.npy` files. `--amp` is bf16 autocast. 115 → 41 s/epoch on an RTX 4070 Ti.

## Final recipe

```bash
python -m autoseg.train $DATA $FAST --model fcn_resnet50_v1d --aug enhanced \
    --optimizer adamw --lr 5e-4 --weight-decay 1e-2 --lr-schedule cosine --warmup-epochs 1 \
    --crop 0 --dice-weight 0.5 --epochs 100 --batch-size 4 --select miou --patience 1000 \
    --seed 1 --out ../models/fin_s1.pt
```

`--crop 0` trains on the full 800×600 frame. The run saves `fin_s1.pt` (best Validation mIoU epoch) and `fin_s1_last.pt` (epoch 100). Seeds 42, 1, 2: about 1 h 50 min each.

Course recipe reproduction (stage 0): `--optimizer adam --lr 2e-4 --lr-schedule plateau --epochs 20 --select acc --patience 5` (512 crop default).

## Synthetic backlight

Add `--glare 0.3` to the final recipe: 30 % of the training images get `autoseg.data.add_glare` (low sun ahead, wet road from the label). The glare draws from its own random stream, so the other augmentation follows the run without glare.

```bash
# from the repository root: does the synthetic glare reproduce the backlit failure on the non-backlit drives?
python experiments/postcourse/glare_check.py --data-root /path/2D_Segmentation --member FCN=fcn_resnet50_v1d:models/fin_s42_last.pt
# how much does a model rely on the rest of the frame? (each tile of a g x g grid predicted on its own)
python experiments/postcourse/context_check.py --data-root /path/2D_Segmentation --grids 1,2 --member FCN=fcn_resnet50_v1d:models/fin_s42_last.pt
```

## U-Net comparison

Same two recipes with `--model unet` (classic U-Net, 31.0 M parameters, no auxiliary head).

## Evaluation-only tools

```bash
# flip / multi-scale TTA, temporal averaging with optical flow, flicker, per-sequence mIoU, confusions
python -m autoseg.analyze $DATA --model fcn_resnet50_v1d --amp --ckpt ../models/fin_s1_last.pt --out analysis.json
#   --per-frame frames.npz adds the confusion matrix of every frame and variant (used for the backlit F5 / F7 split)

# per-class decision bias, checked with leave-one-sequence-out (7 Validation sequences); also writes calib_bias.json
python -m autoseg.calibrate $DATA --model fcn_resnet50_v1d --amp --ckpt ../models/fin_s1_last.pt --out calib.json

# equal-weight ensemble of trained models (no weights fitted)
python -m autoseg.ensemble $DATA --amp --member F1=fcn_resnet50_v1d:../models/fin_s1_last.pt \
    --member U1=unet:../models/u_fin_s1_last.pt --combo F1 --combo U1 --combo F1+U1 --out ensemble.json

# apply the fitted bias (62.64 for seed 1) and write prediction figures
python -m autoseg.evaluate --data-root /path/2D_Segmentation --model fcn_resnet50_v1d --amp \
    --ckpt ../models/fin_s1_last.pt --bias-json calib_bias.json --figures ../results/figures --num-figures 3
```

Bias file format: `{"flip": true, "bias": {"Buildings": -2.3, ...}}`; prediction = argmax over classes of log p + bias, with p averaged over the image and its horizontal flip.

## Temporal head

```bash
python -m autoseg.train_temporal $DATA --amp --mode gru  --seed 1 --base-ckpt ../models/fin_s1_last.pt --out t_gru.pt
python -m autoseg.train_temporal $DATA --amp --mode none --seed 1 --base-ckpt ../models/fin_s1_last.pt --out t_none.pt
```

ConvGRU on the frozen backbone (3-frame clips, 10 epochs) against the same head without the recurrent branch.

## `postcourse_runs.csv` columns

- `stage`: order of the search (0 course recipe reproduction, 1–4 recipe search at 30 epochs with 512 crop, 5 crop, 6–8 full frame, 9 U-Net, 10 synthetic backlight).
- `last_miou`: last epoch. `last5_miou`: mean of the last 5 epochs (decision metric from stage 2 on). `best_miou` / `best_epoch`: best Validation epoch. `acc_selected_miou`: checkpoint chosen by pixel accuracy (course rule).
- `env`: Windows runs (stage 1) kept only the best epoch.
- Model `C+D`: ResNet-50 with the ResNet-C stem and ResNet-D downsampling (`fcn_resnet50_v1d`, the course final model). `B`, `C`, `D` and combinations: `fcn_resnet50_tricks`.
