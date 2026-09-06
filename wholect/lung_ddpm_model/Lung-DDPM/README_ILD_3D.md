# 3-D ILD Lung-DDPM

This branch adapts the paper's volumetric Lung-DDPM design to the local ILD
dataset. It trains a 3-D U-Net noise predictor on complete `128³` volumes and
uses anatomically aware sampling: diffusion generates inside the lung while the
reference CT is retained outside the lung.

## Data and conditions

Each source acquisition contains CT, lung and ROI NIfTI files. Preparation
resamples them to a lung-centered `128³` grid. CT uses linear interpolation;
lung and ROI use nearest-neighbor interpolation.

The model has 19 input channels:

```text
1 noisy CT + 1 lung mask + 17 one-hot ROI labels
```

The output is one channel of predicted 3-D noise. ROI values 1–17 are retained.
The ROI is a semantic condition, not a pixel-accurate lesion segmentation.

## Prepare all 128³ volumes

```bash
cd /media/NAS_R02/USER_PATH/liwei/github_depo/ILD-Segmentation-nnUNet/Lung-DDPM
conda activate nnUnet_ILD

python prepare_ild_3d_128.py \
  --data-root ../ILD_dataset \
  --size 128 \
  --z-spacing 2.5
```

Prepared files are placed under each patient's `lung_ddpm_3d_128/` directory.

## First 3-D training experiment

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
  --lung-loss-weight 1.0 \
  --roi-loss-weight 8.0 \
  --condition-dropout 0.1 \
  --save-every 2000 \
  --preview-every 10000 \
  --preview-strength 1.0 \
  --guidance-scale 1.5 \
  --workers 2 \
  --seed 42
```

`--preview-strength 1.0` performs inference from pure noise and is the honest
generation test closest to the paper. It is slower than SDEdit reconstruction.
For a secondary reconstruction diagnostic, use a value such as `0.35`, but do
not compare that result with pure generation as if they were equivalent.

## Periodic inference outputs

Every `--preview-every` optimizer steps, the fixed validation case and random
seed are used to save:

```text
checkpoints/ILD-DDPM-3D-V1/
├── previews_3d/
│   ├── inference_step_0010000.png
│   └── inference_step_0010000.json
└── inference_nifti/
    └── <case>_step_0010000.nii.gz
```

The PNG contains axial, coronal and sagittal views of:

```text
Reference CT | EMA generated CT | Condition | Absolute difference
```

The NIfTI contains the entire generated 3-D CT in HU and preserves the prepared
volume affine. Open it with the reference, lung and ROI in 3D Slicer or ITK-SNAP.

## Resume

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_ild_3d.py \
  [the same arguments as the original run] \
  --resume checkpoints/ILD-DDPM-3D-V1/ild_ddpm_3d_step_0020000.pth
```

Do not resume 2-D checkpoints into the 3-D model.

## Limitations

- There are 120 volumes from 113 patients, far fewer than the paper's internal cohort.
- 62.9% of source slices have no ROI, although every volume has at least one ROI slice.
- ROI labels occupy about 8.5% of lung voxels.
- Several labels occur in only one or two patients and cannot support reliable generalization.
- The original 10 mm Z sampling is interpolated, not recovered as real anatomy.
- Periodic paired MAE is a reconstruction diagnostic, not proof of realism or disease fidelity.
