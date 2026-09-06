# ILD Deep Learning Pipelines - Supplementary Materials

This directory contains the complete pipeline for Interstitial Lung Disease (ILD) analysis, from raw data processing to model training.

## 🔄 Overall Workflow

The pipeline is divided into two main tracks: **Classification** and **Segmentation**.

### Track 1: Lung Disease Classification
1. **Data Prep**: Run `create_classification_data.py`.
   - **Input**: Raw `.tif` patches in `ILD_DB/ILD_DB_talismanTestSuite/`.
   - **Output**: `ILD_DB_npy/images.npy` and `labels.npy` (32x32 patches).
2. **Training**: Run `classification_entry.py`.
   - Reads the `.npy` files and trains a CNN classifier (5 classes).

### Track 2: Lung Segmentation
### Data Processing
1. **DICOM to Image** (Required for segmentation)
   - Use `convert_dicom.ipynb` to process the raw `ILD_DB` DICOM files.
   - **⚠️ CRITICAL UPDATE**: You *must* re-run this notebook! It has been updated to save masks as lossless `.png` files without altering their exact label values. The old `.jpg` masks are corrupted by compression and scaling artifacts.
   - Place the output into the `ILD_DB_fig/` directory.
   - **Output**: JPG slices and masks in `ILD_DB_fig/`.
2. **Training**: Run `segmentation_entry.py`.
   - Pairs CT slices with masks from `ILD_DB_fig/` and trains a UNet-based model.

---

## 📂 Segmentation Datasets Explained

Inside `ILD_DB_fig/`, you will find three main sub-directories:

- **`ILD_DB_lungMasks`**: Contains binary masks (Background vs. Lung). Used for basic lung field extraction.
- **`ILD_DB_txtROIs`**: ROI masks presumably derived from specific regional annotations.
- **`ILD_DB_volumeROIs`**: The primary dataset for disease segmentation. Contains 13 classes (Background + 12 disease types like Fibrosis, Emphysema, etc.).

---

## 🚀 Running the Entry Scripts

Both scripts are optimized for the current environment and include unified logging.

### 📊 `classification_entry.py`
Trains a model to classify 32x32 CT patches.
- **Key Variables**:
  - `BATCH_SIZE`: Number of patches per batch (default: 16).
  - `NUM_EPOCHS`: Total training cycles (default: 50).
  - `DEVICE`: Set to `cpu` for stability on current PyTorch 1.13 + H100 environment.
  - `train_split` / `val_split`: Controls data division (default: 0.8 / 0.1, leaving 0.1 for test).

### 🔍 `segmentation_entry.py`
Trains a model for pixel-level lung segmentation.
- **Key Variables**:
  - `DATASET_NAME`: **Crucial.** Switch between `"ILD_DB_volumeROIs"`, `"ILD_DB_lungMasks"`, or `"ILD_DB_txtROIs"`.
  - `FILTER_EMPTY_MASKS`: Set to `True` (default) to skip slices where the mask is entirely black. This makes training much more efficient by focusing on annotated areas.
  - `IMG_SIZE`: Target resolution (default: 256 for faster training).
  - `NUM_CLASSES`: Automatically adjusts based on `DATASET_NAME`.
  - **Robust Loading**: The script now automatically detects binary masks (0/255) and remaps them to (0/1) for `lungMasks` tasks, while preserving disease indices for `volumeROIs`.

---

## 📈 Logging & Results
All runs generate a timestamped folder in the `experiments/` directory:
- `train.log`: Full console output including per-class metrics.
- `checkpoints/`: Model weights for every epoch and the `best_model.pth`.
- **Metrics**: 
  - Classification reports Precision/Recall/F1 for each disease class.
  - Segmentation reports **Dice Scores** for each mask category.
