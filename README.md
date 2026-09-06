# HRCT Synthetic Augmentation

MSc thesis project (Imperial College London): does diffusion-model
synthetic data augmentation improve multi-class classification of ILD
(interstitial lung disease) tissue patterns in HRCT? Two independent
branches investigate this — patch-level classification with a
self-trained DDPM, and whole-CT classification with an externally
provided semantic-layout-guided diffusion model.

## Repo layout

```
patches/          Branch A: patch-level classification + own DDPMs
wholect/          Branch B: whole-CT classification + external diffusion model
diffusion_model/  DDPM code/architecture used by Branch A
classifier_lib/   Shared classifier library (models.py, data_helpers.py)
scripts/          PBS job launchers (.sh) for both branches
ILD_DB/           Raw source data (see "Restoring data" below)
```

## Setup

1. Clone the repo.
2. Create the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate ild
   ```
   (PyTorch 2.5.1+cu121 — later versions are known to cause an
   `ImportError` in this codebase; use this exact version.)
3. Set `PROJECT_ROOT` to the repo's absolute path — every script and job
   launcher reads this instead of a hardcoded path:
   ```bash
   export PROJECT_ROOT=/absolute/path/to/this/repo
   ```
   Add this to your `.bashrc` (or equivalent) if you'll be running jobs
   across multiple sessions.
4. PBS job scripts require `PROJECT_ROOT` to already be exported before
   `qsub` — they will fail immediately with a clear error message if it
   isn't set (`: "${PROJECT_ROOT:?...}"` guard at the top of each `.sh`).

## Restoring data (required before running anything)

This repo's git history contains code only. All datasets, checkpoints,
and generated images (~6.4 GB total) are stored separately on OneDrive,
as 6 `.tar.gz` archives. Download all 6 to `~/for_onedrive_upload/` (or
adjust the source path below), then restore each with `PROJECT_ROOT`
already exported:

```bash
# 1. Raw source DICOM data
mkdir -p "$PROJECT_ROOT/ILD_DB"
tar -xzf ~/for_onedrive_upload/ILD_DB_data.tar.gz -C "$PROJECT_ROOT/ILD_DB"

# 2. External model's own dataset package
mkdir -p "$PROJECT_ROOT/wholect/lung_ddpm_model"
tar -xzf ~/for_onedrive_upload/wholect_lung_ddpm_dataset.tar.gz -C "$PROJECT_ROOT/wholect/lung_ddpm_model"

# 3. Processed .npy datasets (both branches) -- paths inside the archive
#    are already complete from repo root, so extract straight to PROJECT_ROOT
tar -xzf ~/for_onedrive_upload/processed_npy_datasets.tar.gz -C "$PROJECT_ROOT"

# 4. Model checkpoints (patch DDPMs, attention U-Net, external model 2D/3D,
#    nnU-Net) -- also extracts to multiple destinations automatically
tar -xzf ~/for_onedrive_upload/checkpoints.tar.gz -C "$PROJECT_ROOT"

# 5. nnU-Net raw training images
tar -xzf ~/for_onedrive_upload/nnunet_raw_images.tar.gz -C "$PROJECT_ROOT/wholect/lung_ddpm_model"

# 6. Generated synthetic samples + validation previews
tar -xzf ~/for_onedrive_upload/generated_and_previews.tar.gz -C "$PROJECT_ROOT"
```

**Extract all 6, in any order.** Steps 3 and 6 both contribute files to
`wholect/lung_ddpm_model/Lung-DDPM/generated/` (step 3 has the `.npy`
arrays, step 6 has the `.png` thumbnails and `generation_report.json`)
— that directory is only complete once both have been extracted. Some
files are duplicated across archives (by design, so each archive is
self-contained); re-extracting is harmless, it just overwrites with
identical content.

## Branch A — patch-level classification (`patches/`)

32×32 HRCT patches, 5 tissue classes, LOPO cross-validation. Own DDPMs
(one per class), generator quality validated via a full checkpoint
sweep (recall/FID in classifier feature space) rather than trusting the
lowest-training-loss checkpoint.

Key scripts (`patches/scripts/`):
- `create_classification_data.py` — extracts real patches from
  `ILD_DB/ILD_DB_talismanTestSuite/` into `patches/ILD_DB_npy/`
- `classification_lopo.py` — baseline (real-only) LOPO classification
- `classification_lopo_augmented_v3_5k.py` — final reference augmented
  comparison (5000 synthetic patches/class, quality-selected checkpoints)
- `classification_lopo_augmented_attention.py` — same, using the
  self-attention U-Net variant
- `sweep_evaluate*.py` — checkpoint-quality sweep (FID/precision/recall)

PBS launchers: `scripts/job_lopo_*.sh`, `job_ema_finetune_*.sh`,
`job_train_attention_*.sh`.

See the numeric results and full experimental history in the
dissertation writeup for details across all dataset versions (v1-v5,
attention).

## Branch B — whole-CT classification (`wholect/`)

Whole 512×512 axial CT slices, 5 classes, dominant-class-per-slice
labeling. Uses an externally provided semantic-layout-guided,
class-conditioned 2D diffusion model (`wholect/lung_ddpm_model/`,
reference-conditioned/inpainting-based sampling, not unconditional
generation) for augmentation.

Key scripts (`wholect/scripts/`):
- `build_wholect_dataset.py` — extracts whole-slice dataset from
  `ILD_DB/ILD_DB_volumeROIs/` (applies `RescaleSlope`/`RescaleIntercept`
  correctly via `apply_modality_lut` — raw `pydicom` pixel values are
  NOT Hounsfield Units without this)
- `build_wholect_augmented.py` — merges real + external-model synthetic
  samples, tagging synthetic samples by their real source-patient ID to
  prevent leakage in patient-level cross-validation
- `build_wholect_fixed_split.py` — stratified 80/20 train/test split,
  restricted so the test set was never seen by the external model's own
  training
- `classification_wholect_uint8_lopo.py` — slice-level LOPO
- `classification_wholect_patientagg_lopo.py` — patient-level, majority
  vote across slice predictions (the branch's strongest result)
- `classification_wholect_pool.py` — patient-level, mean/max feature
  pooling on the fixed split
- `evaluate_wholect_synthetic_quality.py` — classifier-feature-space
  FID/precision/recall for synthetic sample quality

PBS launchers: `scripts/job_wholect_*.sh`.

**Note:** `classification_wholect_lopo.py` (no `_uint8` suffix) is a
deprecated, pre-HU-fix version kept only because several other scripts
still import `WholeCTClassifier` from it. Do not use it directly for
classification — use `classification_wholect_uint8_lopo.py`.

## License / data provenance

`ILD_DB/` (raw HRCT data) originates from the University Hospitals of
Geneva's public ILD database and requires a signed end-user copyright
agreement for access — see `ILD_DB/README` after restoring. The
external diffusion model in `wholect/lung_ddpm_model/` was provided by
a project collaborator and is not original to this repo.
