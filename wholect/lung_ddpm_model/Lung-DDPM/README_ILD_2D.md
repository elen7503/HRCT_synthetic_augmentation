# ILD 2-D Conditional Diffusion

This directory contains a small-data, semantic-layout-conditioned diffusion
pipeline for thoracic CT. It is designed for **classification augmentation and
controlled texture experiments**, not for deriving true lesion segmentation
masks from rectangular ILD ROIs.

The refactored pipeline mixes:

1. CT + lung-mask slices with no class-labelled ROI, used to learn general lung
   anatomy and texture under a null/unknown disease condition.
2. CT + lung-mask + class-labelled ROI slices, used to learn spatial and class
   conditioning.

An absent or empty ROI is always interpreted as **unknown**, never as confirmed
healthy tissue or lesion-free background.

## What changed after V3

The original V3 experiment trained on only 664 ROI-positive slices. Its sampler
dropped every empty-ROI slice, and its validation preview mainly measured how
well an SDEdit-style sampler restored a real reference CT. Falling training loss
therefore did not prove that the network used disease labels.

The refactor adds:

- image-led discovery of all aligned CT + lung-mask slices;
- optional ROI files and explicit `null_unknown` condition semantics;
- mixed sampling with a configurable ROI-positive fraction;
- rare-class balancing inside the ROI-positive group;
- reconstruction previews with non-overlapping titles;
- correct-condition versus no-disease, label-swap and ROI-move controls;
- multi-seed diversity diagnostics;
- dataset manifests that record whether each sample is `disease_roi` or
  `null_unknown`;
- unit tests for missing ROI, empty ROI and mixed sampling.

Because the training population changed, a mixed model **must start from
scratch**. Do not resume a V3 checkpoint into a mixed experiment.

## Dataset semantics

Expected layout:

```text
ILD_dataset/
└── patient_<id>/
    ├── images/
    │   └── slice_<n>.npy
    ├── lung_masks/
    │   └── slice_<n>.npy
    └── roi_masks/                 # optional per slice
        └── slice_<n>.npy
```

The current disease channels are:

| Channel | Value | Name |
|---:|---:|---|
| 0 | binary | Lung mask |
| 1 | 1 | Healthy-control ROI |
| 2 | 2 | Emphysema ROI |
| 3 | 3 | Ground-glass ROI |
| 4 | 4 | Fibrosis ROI |
| 5 | 5 | Micronodule ROI |
| 6 | 6 | Consolidation ROI |

These ROIs indicate labelled tissue regions. They are not guaranteed to be
pixel-accurate lesion contours. Generated outputs must not be presented as
segmentation predictions.

### Current patient-level split

The split is read from:

```text
../nnUNet_raw/Dataset502_ILD/conversion_report.json
```

The current discovery result is:

| Split | Patients | All CT+lung slices | ROI-positive | Null/unknown |
|---|---:|---:|---:|---:|
| Train | 79 | 2,194 | 664 | 1,530 |
| Validation | 17 | 468 | 99 | 369 |
| Test | 17 | 413 | 173 | 240 |

Training, validation and test patients remain disjoint. Validation preview cases
are selected only from ROI-positive validation slices so every disease channel
can be inspected.

## Model and objective

The model is a 2-D conditional U-Net trained to predict diffusion noise. Its
input is:

```text
noisy CT (1 channel)
+ lung layout (1 channel)
+ six disease-layout channels
```

Loss weights are spatially assigned:

```text
outside lung  → outside-loss-weight
inside lung   → lung-loss-weight
disease ROI   → roi-loss-weight × per-class weight
```

Null-condition slices still contribute lung and outside-lung denoising loss, but
their six disease channels are zero. This teaches an anatomy/texture prior
without incorrectly labelling unannotated lung pixels as healthy.

`--roi-sample-fraction` controls the expected training mixture. For example,
`0.5` assigns half of sampler probability mass to ROI-positive slices and half
to null-condition slices. Rare classes are balanced only within the ROI group.

## Environment

```bash
cd /media/NAS_R02/USER_PATH/liwei/github_depo/ILD-Segmentation-nnUNet/Lung-DDPM
conda activate nnUnet_ILD
```

The required Python packages are listed in `requirements.txt`. CUDA is required
for training and sampling.

## Recommended mixed training command

Use a new experiment directory, for example `ILD-DDPM-2D-Mixed-V4`:

```bash
CUDA_VISIBLE_DEVICES=1 python -u train_ild_2d.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --output-dir checkpoints/ILD-DDPM-2D-Mixed-V4 \
  --image-size 256 \
  --model-channels 64 \
  --num-res-blocks 2 \
  --timesteps 250 \
  --batch-size 8 \
  --steps 60000 \
  --roi-sample-fraction 0.5 \
  --learning-rate 2e-5 \
  --ema-decay 0.999 \
  --outside-loss-weight 0.1 \
  --lung-loss-weight 1 \
  --roi-loss-weight 8 \
  --class-loss-weights 2 4 1.25 1 1.5 1.75 \
  --condition-dropout 0.1 \
  --max-training-strength 0.65 \
  --guidance-scale 1.5 \
  --blend-mode lung \
  --inpainting-strength 0.35 \
  --roi-crop-probability 0.5 \
  --roi-crop-size 384 \
  --rotation-degrees 7 \
  --scale-range 0.95 1.05 \
  --intensity-jitter 0.03 \
  --save-every 2000 \
  --log-every 100 \
  --preview-count 6 \
  --control-eval-every 10000 \
  --control-seeds 3 \
  --workers 4 \
  --seed 42
```

Important options:

| Option | Meaning |
|---|---|
| `--roi-sample-fraction` | Expected ROI-positive fraction after sampling |
| `--condition-dropout` | Drops disease channels for classifier-free guidance; lung remains |
| `--max-training-strength` | Restricts training noise levels for small-data image-to-image use |
| `--blend-mode lung` | Generate inside lung and preserve reference CT outside lung |
| `--inpainting-strength` | SDEdit strength; lower values preserve more reference information |
| `--control-eval-every` | Frequency of condition-ablation diagnostics; `0` disables |
| `--foreground-only` | Legacy ablation that recreates ROI-only data selection |

`training_loss.csv` records both mean loss and the actual sampled ROI-condition
fraction. A large mismatch from `--roi-sample-fraction` should be investigated.

## Checkpoint outputs

Each experiment contains:

```text
checkpoints/ILD-DDPM-2D-Mixed-V4/
├── ild_ddpm_2d_step_XXXXXXX.pth
├── training_config.json
├── training_loss.csv
├── train_manifest.json
├── validation_manifest.json
├── previews/
├── preview_metrics/
└── condition_control/
```

The manifest SHA-256 is stored in checkpoints. Resume is refused when the
current training files differ from the checkpoint manifest.

## Resume a mixed experiment

Use exactly the same architecture, data and sampling arguments, then add:

```bash
--resume checkpoints/ILD-DDPM-2D-Mixed-V4/ild_ddpm_2d_step_0020000.pth
```

Changing the dataset mixture requires a new experiment rather than resume.

## Reconstruction preview

The regular three-column preview shows:

```text
reference + condition | EMA reconstruction + condition | absolute difference ×4
```

ROI PSNR, SSIM and MAE are paired reconstruction diagnostics. They do **not**
measure realism, diversity, disease correctness or whether the model uses the
condition. Tiny ROIs can make SSIM especially unstable.

## Condition-control evaluation

Every `--control-eval-every` steps, the pipeline uses the same reference and
stochastic trajectory for four conditions:

1. correct disease condition;
2. all disease channels removed;
3. disease labels cyclically swapped while retaining the ROI shape;
4. disease ROI moved to another lung location.

It also samples the correct condition with multiple random seeds.

The preview columns are:

```text
Reference | Correct | No disease | Label swap | ROI moved | |Correct-null| ×4
```

Each row is annotated with its original class label IDs. The four generated
columns use an identical stochastic seed, so their differences are attributable
to the changed condition rather than unrelated sampling noise. The correct,
label-swap and ROI-move panels display the layout actually supplied to that
sample, using the fixed class colors.

The JSON reports:

- correct versus no-disease MAE inside ROI and lung;
- correct versus label-swap MAE;
- correct versus ROI-move MAE;
- fraction of pixels changing by more than five intensity units;
- mean pairwise lung MAE across random seeds.

Interpretation:

- near-zero correct/null, correct/swap and correct/move differences indicate
  that the model is ignoring disease conditions;
- nonzero differences show sensitivity, not medical correctness;
- nonzero multi-seed differences show diversity, not realism;
- medical validity still requires expert review or a validated downstream
  disease classifier/segmenter.

## Evaluate an existing checkpoint without training

For a new mixed checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1 python train_ild_2d.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --output-dir checkpoints/Mixed-V4-evaluation \
  --image-size 256 \
  --model-channels 64 \
  --num-res-blocks 2 \
  --timesteps 250 \
  --preview-count 6 \
  --control-eval-every 1 \
  --control-seeds 4 \
  --guidance-scale 1.5 \
  --blend-mode lung \
  --inpainting-strength 0.35 \
  --preview-only \
  --resume checkpoints/ILD-DDPM-2D-Mixed-V4/ild_ddpm_2d_step_0018000.pth
```

For a legacy V3 ROI-only checkpoint, add `--foreground-only`. This is for
evaluation only; do not continue V3 training as a mixed experiment.

## Aggregate checkpoint reconstruction diagnostics

```bash
python analyze_ild_ddpm_experiment.py checkpoints/ILD-DDPM-2D-Mixed-V4
```

Outputs are written under `analysis/`:

```text
checkpoint_summary.csv
per_case_metrics.csv
per_label_metrics.csv
checkpoint_metric_curves.png
analysis_report.json
```

Choose candidate checkpoints from both reconstruction and condition-control
results. A checkpoint with the best PSNR but negligible condition effect is not
a useful conditional generator.

## Generate class-balanced synthetic slices

```bash
CUDA_VISIBLE_DEVICES=1 python sample_ild_2d_balanced.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --checkpoint checkpoints/ILD-DDPM-2D-Mixed-V4/ild_ddpm_2d_step_0018000.pth \
  --output-dir data/ILD-synthetic-Mixed-V4 \
  --target-per-class 250 \
  --guidance-scale 1.5 \
  --blend-mode lung \
  --inpainting-strength 0.35
```

Sampling reuses real layouts. Synthetic images must remain grouped with their
source patient when building downstream splits; otherwise patient information
can leak across train and validation/test sets.

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
python -m py_compile ild_2d.py ild_evaluation.py train_ild_2d.py
```

## Known limitations

- The dataset contains only 664 ROI-positive training slices.
- ROI boxes/regions are not pixel-accurate disease masks.
- A null condition means unannotated, not healthy.
- SDEdit with a real reference can encourage reconstruction instead of strong
  semantic editing.
- PSNR and SSIM cannot establish medical validity.
- Condition-control tests measure sensitivity, not whether changed texture is
  the requested pathology.
- 2-D training ignores through-plane continuity.

The recommended next scientific improvement is to add independently annotated
pixel-level lesion data or a validated pathology classifier and use it to score
condition fidelity on held-out patients.
