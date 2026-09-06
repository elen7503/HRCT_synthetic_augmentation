# Lung-DDPM for ILD CT synthesis

This repository contains two separate conditional diffusion pipelines for ILD
CT synthesis:

| Pipeline | Training unit | Model input | Main entry point |
|---|---|---|---|
| 2-D | one axial `256 x 256` slice | noisy CT + lung mask + 6 ROI class channels | `train_ild_2d.py` |
| 3-D | one `128 x 128 x 128` volume | noisy CT + lung mask + 17 ROI class channels | `train_ild_3d.py` |

Both pipelines are supported, but they are **different models**. A 2-D
checkpoint cannot be loaded by the 3-D code, or vice versa.

The ROI is used as a semantic layout condition. It is not a pixel-accurate
lesion mask, so this project is intended for controlled synthesis and data
augmentation, not for producing ground-truth segmentations.

## Installation

```bash
cd /media/NAS_R02/USER_PATH/liwei/github_depo/ILD-Segmentation-nnUNet/Lung-DDPM
conda activate nnUnet_ILD
pip install -r requirements.txt
```

A CUDA GPU is required for training and sampling.

## Data and patient split

The 2-D loader expects:

```text
ILD_dataset/
└── patient_<id>/
    ├── images/slice_<n>.npy
    ├── lung_masks/slice_<n>.npy
    └── roi_masks/slice_<n>.npy       # optional for the mixed pipeline
```

The 3-D loader uses the prepared NIfTI volumes referenced by
`../nnUNet_raw/Dataset503_ILD3DROI/dataset.json`. Always use the supplied split
report so slices/volumes from one patient cannot leak between training and
validation.

## 2-D training

### Recommended new mixed experiment

The current implementation can learn general lung anatomy from all aligned CT
and lung-mask slices, while ROI-positive slices additionally provide disease
conditions. An empty/missing ROI means **unknown**, not healthy.

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_ild_2d.py \
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
  --preview-count 6 \
  --control-eval-every 10000 \
  --control-seeds 3 \
  --workers 4 \
  --seed 42
```

Do not resume the old V3 checkpoint into this mixed experiment: V3 used a
different training population.

### Reproducing the existing 2-D V3 experiment

The trained model is stored in:

```text
checkpoints/ILD-DDPM-2D-V3/
```

V3 used only 664 ROI-positive training slices and six disease classes. Its
essential settings are recorded in `training_config.json`. To reproduce its
data selection, the command must include `--foreground-only`:

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_ild_2d.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --output-dir checkpoints/ILD-DDPM-2D-V3-Reproduction \
  --image-size 256 --model-channels 64 --num-res-blocks 2 --timesteps 250 \
  --batch-size 8 --steps 60000 --foreground-only \
  --learning-rate 2e-5 --ema-decay 0.999 \
  --outside-loss-weight 0.1 --lung-loss-weight 1 --roi-loss-weight 8 \
  --class-loss-weights 2 4 1.25 1 1.5 1.75 \
  --condition-dropout 0.1 --max-training-strength 0.65 \
  --guidance-scale 1.5 --blend-mode lung --inpainting-strength 0.35 \
  --roi-crop-probability 0.5 --roi-crop-size 384 \
  --rotation-degrees 7 --scale-range 0.95 1.05 --intensity-jitter 0.03 \
  --save-every 2000 --preview-count 6 --workers 4 --seed 42
```

## Loading V3 for 2-D inference

### 1. Evaluate one checkpoint on fixed validation cases

The example below loads the EMA weights at step 18,000 and performs inference
without further training. Step 18,000 is an evaluation candidate, not a claim
that it is universally the best checkpoint; replace it after comparing the
saved validation/control results.

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_ild_2d.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --output-dir checkpoints/ILD-DDPM-2D-V3-Evaluation \
  --image-size 256 --model-channels 64 --num-res-blocks 2 --timesteps 250 \
  --batch-size 8 --foreground-only --preview-count 6 \
  --guidance-scale 1.5 --blend-mode lung --inpainting-strength 0.35 \
  --control-eval-every 1 --control-seeds 3 \
  --preview-only \
  --resume checkpoints/ILD-DDPM-2D-V3/ild_ddpm_2d_step_0018000.pth
```

Outputs are written to `previews/`, `preview_metrics/`, and
`condition_control/`. The preview columns are reference CT, EMA reconstruction,
and amplified absolute difference. The control evaluation additionally checks
removed labels, swapped labels, moved ROIs, and multiple random seeds.

`--inpainting-strength 0.35` is SDEdit-style reference-conditioned inference:
it intentionally preserves substantial reference information. It must not be
reported as unconditional generation. A value near `1.0` starts near pure
noise, but V3 was trained only up to strength `0.65`, so its quality may degrade.

### 2. Generate a class-balanced synthetic 2-D dataset

This sampler reads the architecture and default sampling settings directly from
the checkpoint and loads its EMA model:

```bash
CUDA_VISIBLE_DEVICES=0 python -u sample_ild_2d_balanced.py \
  --data-root ../ILD_dataset \
  --split-report ../nnUNet_raw/Dataset502_ILD/conversion_report.json \
  --checkpoint checkpoints/ILD-DDPM-2D-V3/ild_ddpm_2d_step_0018000.pth \
  --output-dir generated/ILD-DDPM-2D-V3-balanced \
  --target-per-class 250 \
  --guidance-scale 1.5 \
  --blend-mode lung \
  --inpainting-strength 0.35
```

Each generated case contains an image, lung mask and copied semantic ROI, plus
`generation_report.json`. Use `--overwrite` only when intentionally replacing a
non-empty output directory.

### 3. Compare checkpoints

```bash
python analyze_ild_ddpm_experiment.py \
  checkpoints/ILD-DDPM-2D-V3 \
  --output-dir checkpoints/ILD-DDPM-2D-V3/analysis
```

Checkpoint selection should consider reconstruction metrics, condition-control
responses, visual anatomy, disease plausibility and diversity. PSNR/SSIM alone
only measure paired reconstruction similarity.

## 3-D data preparation

The preparation step resamples CT, lung and ROI to a common physical spacing,
then lung-centred pads/crops them to `128^3`. CT uses linear interpolation;
masks use nearest-neighbour interpolation.

```bash
python prepare_ild_3d_128.py \
  --data-root ../ILD_dataset \
  --size 128 \
  --z-spacing 2.5
```

This does not invent additional anatomy. Resampling only places existing
physical anatomy onto a consistent voxel grid.

## 3-D training and inference previews

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_ild_3d.py \
  --dataset-dir ../nnUNet_raw/Dataset503_ILD3DROI \
  --output-dir checkpoints/ILD-DDPM-3D-V1 \
  --volume-size 128 \
  --model-channels 32 \
  --num-res-blocks 1 \
  --timesteps 250 \
  --batch-size 1 \
  --gradient-accumulation 2 \
  --steps 100000 \
  --learning-rate 1e-5 \
  --ema-decay 0.995 \
  --outside-loss-weight 0.1 \
  --lung-loss-weight 1 \
  --roi-loss-weight 8 \
  --condition-dropout 0.1 \
  --save-every 2000 \
  --preview-every 10000 \
  --preview-strength 1.0 \
  --guidance-scale 1.5 \
  --workers 2 \
  --seed 42
```

Every `--preview-every` steps, the code runs inference with EMA weights and
saves:

```text
checkpoints/ILD-DDPM-3D-V1/
├── previews_3d/inference_step_XXXXXXX.png
├── previews_3d/inference_step_XXXXXXX.json
└── inference_nifti/<case>_step_XXXXXXX.nii.gz
```

The PNG shows axial, coronal and sagittal views of reference, generated CT,
condition and absolute difference. The NIfTI contains the full generated volume
and can be opened in 3D Slicer or ITK-SNAP.

To resume 3-D training, repeat the original command and append:

```bash
--resume checkpoints/ILD-DDPM-3D-V1/ild_ddpm_3d_step_0020000.pth
```

The current 3-D configuration retains ROI labels 1-17. It is independent of
the six-class nnU-Net experiment.

## Sharing and reproducibility

For another user to reproduce an experiment, share:

- this source directory and `requirements.txt`;
- the selected `.pth` checkpoint;
- its `training_config.json`;
- `train_manifest.json` and `validation_manifest.json`;
- the patient-level split report or instructions for recreating it;
- the exact inference command and random seed.

The checkpoint loader verifies the training manifest for safe resume/preview.
The receiving user therefore needs the same patient split and aligned input
files for the validation commands above. Do not redistribute clinical data
unless its license and data-use agreement permit it.

More implementation details are available in [README_ILD_2D.md](README_ILD_2D.md)
and [README_ILD_3D.md](README_ILD_3D.md).

## Upstream project and citation

This ILD adaptation is based on **Lung-DDPM: Semantic Layout-guided Diffusion
Models for Thoracic CT Image Synthesis**. The upstream generic 3-D entry points
remain available as `train.py` and `sample.py`; the `*_ild_2d.py` and
`*_ild_3d.py` files are the dataset-specific workflows documented above.

If this code contributes to a publication, cite the original work:

```bibtex
@misc{jiang2025lungddpmsemanticlayoutguideddiffusion,
  title        = {Lung-DDPM: Semantic Layout-guided Diffusion Models for Thoracic CT Image Synthesis},
  author       = {Yifan Jiang and Yannick Lemaréchal and Josée Bafaro and Jessica Abi-Rjeile and Philippe Joubert and Philippe Després and Venkata Manem},
  year         = {2025},
  eprint       = {2502.15204},
  archivePrefix= {arXiv},
  primaryClass = {eess.IV},
  url          = {https://arxiv.org/abs/2502.15204}
}
```
