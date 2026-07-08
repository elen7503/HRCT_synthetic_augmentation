import os
import re
import numpy as np
import pydicom
import SimpleITK as sitk
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────

# Patients with no CT files — skip entirely
SKIP_PATIENTS = {
    '8','53','57','142','154','171','172',
    '173','174','175','177','179','182','183','184','185'
}

# HU window for lung tissue
HU_MIN = -1350.0
HU_MAX =  150.0

# Target in-plane spacing after resampling (mm)
# Z spacing stays at 1.0mm — no change needed
TARGET_SPACING_XY = 0.7

# Label remapping: from original 17-class to 5-class scheme
# 0 = background / unannotated (excluded from loss)
# 1 = ground_glass   (29 patients — most common)
# 2 = fibrosis       (27 patients)
# 3 = micronodules   (13 patients)
# 4 = consolidation  (13 patients)
# 5 = other_pathology (all remaining rare classes merged)
LABEL_REMAP = {
    0:  0,   # unannotated → background
    1:  5,   # healthy → other (only 7 patients, too few)
    2:  5,   # emphysema → other
    3:  1,   # ground_glass → class 1
    4:  2,   # fibrosis → class 2
    5:  3,   # micronodules → class 3
    6:  4,   # consolidation → class 4
    7:  5,   # bronchial_wall_thickening → other
    8:  5,   # reticulation → other
    9:  5,   # macronodules → other
    10: 5,   # cysts → other
    11: 5,   # peripheral_micronodules → other
    12: 5,   # bronchiectasis → other
    13: 5,   # air_trapping → other
    14: 5,   # early_fibrosis → other
    15: 5,   # increased_attenuation → other
    16: 5,   # tuberculosis → other
    17: 5,   # pcp → other
}

# 3D patch size: [depth, height, width]
# Depth=16 because patients only have 14–52 slices (mean 25)
# Using 64 depth would be too large for most patients
PATCH_SIZE    = (16, 128, 128)
PATCH_STRIDE  = (8,  64,  64)   # 50% overlap in all dimensions


# ─────────────────────────────────────────────────────────────────
# HELPER: slice index from filename
# ─────────────────────────────────────────────────────────────────
def _slice_index(filename):
    numbers = re.findall(r'\d+', filename)
    return int(numbers[-1])


# ─────────────────────────────────────────────────────────────────
# STEP 1: Load raw HRCT + tissue mask + lung mask for one patient
# ─────────────────────────────────────────────────────────────────
def load_patient_raw(pid, vol_roi_base, lung_mask_base):
    """
    Load the three raw volumes for one patient.

    Returns:
        hrct       : [D, H, W] float32  — raw HU values
        tissue_mask: [D, H, W] uint8    — original labels 0–17
        lung_mask  : [D, H, W] uint8    — 0=outside lung, 255=lung
        spacing    : (float, float)     — (pixel_spacing_mm, slice_thickness_mm)
    """
    vol_dir   = os.path.join(vol_roi_base,   pid)
    lmask_dir = os.path.join(lung_mask_base, pid)

    # ── HRCT slices ───────────────────────────────────────────────
    ct_files = sorted(
        [f for f in os.listdir(vol_dir) if f.startswith('CT-') and f.endswith('.dcm')],
        key=_slice_index
    )

    # Read spacing from first slice
    ds0 = pydicom.dcmread(os.path.join(vol_dir, ct_files[0]))
    ps  = float(ds0.PixelSpacing[0])
    st  = float(ds0.SliceThickness)

    ct_slices = []
    for f in ct_files:
        ds  = pydicom.dcmread(os.path.join(vol_dir, f))
        slope     = float(getattr(ds, 'RescaleSlope',     1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        ct_slices.append(ds.pixel_array.astype(np.float32) * slope + intercept)
    hrct = np.stack(ct_slices, axis=0)

    # ── Tissue mask (roi_mask/) ───────────────────────────────────
    mask_dir   = os.path.join(vol_dir, 'roi_mask')
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith('.dcm')],
        key=_slice_index
    )
    tissue_slices = []
    for f in mask_files:
        ds = pydicom.dcmread(os.path.join(mask_dir, f))
        tissue_slices.append(ds.pixel_array.astype(np.uint8))
    tissue_mask = np.stack(tissue_slices, axis=0)

    # ── Lung mask (lung_mask/) ────────────────────────────────────
    lm_dir    = os.path.join(lmask_dir, 'lung_mask')
    lm_files  = sorted(
        [f for f in os.listdir(lm_dir) if f.endswith('.dcm')],
        key=_slice_index
    )
    lung_slices = []
    for f in lm_files:
        ds = pydicom.dcmread(os.path.join(lm_dir, f))
        lung_slices.append(ds.pixel_array.astype(np.uint8))
    lung_mask = np.stack(lung_slices, axis=0)

    return hrct, tissue_mask, lung_mask, (ps, st)


# ─────────────────────────────────────────────────────────────────
# STEP 2: Resample HRCT and masks to target XY spacing
#
# Why SimpleITK here and not scipy?
# SimpleITK handles physical spacing correctly and applies the right
# interpolation per volume type automatically:
#   - HRCT: linear interpolation (smooth intensity values)
#   - masks: nearest neighbour (must preserve integer labels exactly,
#            no blending allowed between label 3 and label 4)
# ─────────────────────────────────────────────────────────────────
def resample_volume(volume, original_spacing_xy, is_mask=False):
    """
    Resample a 3D volume from its original XY spacing to TARGET_SPACING_XY.
    Z spacing is always 1.0mm and stays unchanged.

    Args:
        volume             : [D, H, W] numpy array
        original_spacing_xy: float, current mm per pixel in X and Y
        is_mask            : bool — use nearest neighbour if True,
                             linear interpolation if False

    Returns:
        resampled volume as numpy array
    """
    # SimpleITK expects spacing as (x, y, z) — note the axis order
    # Our numpy array is [D, H, W] = [z, y, x] in physical space
    sitk_image = sitk.GetImageFromArray(volume.astype(np.float32))
    sitk_image.SetSpacing((
        original_spacing_xy,   # x spacing
        original_spacing_xy,   # y spacing
        1.0                    # z spacing (always 1mm)
    ))

    original_size    = sitk_image.GetSize()       # (W, H, D) in SimpleITK order
    original_spacing = sitk_image.GetSpacing()

    # Compute new size to preserve physical extent
    # new_size = original_size * (original_spacing / target_spacing)
    new_size = [
        int(round(original_size[0] * original_spacing[0] / TARGET_SPACING_XY)),  # W
        int(round(original_size[1] * original_spacing[1] / TARGET_SPACING_XY)),  # H
        original_size[2]   # D stays the same
    ]

    new_spacing = (TARGET_SPACING_XY, TARGET_SPACING_XY, 1.0)

    interpolator = (sitk.sitkNearestNeighbor if is_mask
                    else sitk.sitkLinear)

    resampled = sitk.Resample(
        sitk_image,
        new_size,
        sitk.Transform(),          # identity transform (no rotation)
        interpolator,
        sitk_image.GetOrigin(),
        new_spacing,
        sitk_image.GetDirection(),
        0.0,                       # default value for voxels outside boundary
        sitk_image.GetPixelID()
    )

    return sitk.GetArrayFromImage(resampled)   # back to numpy [D, H, W]


# ─────────────────────────────────────────────────────────────────
# STEP 3: Apply lung mask, clip HU, normalize to [0,1]
# ─────────────────────────────────────────────────────────────────
def preprocess_hrct(hrct, lung_mask):
    """
    Apply lung mask, clip to lung HU window, normalize to [0,1].

    Args:
        hrct      : [D, H, W] float32, HU values
        lung_mask : [D, H, W] uint8,   0=outside, 255=lung

    Returns:
        [D, H, W] float32, values in [0, 1], non-lung voxels = 0
    """
    # Step 3a: zero out everything outside the lung
    # lung_mask > 0 gives a boolean array — True where lung
    lung_binary = (lung_mask > 0).astype(np.float32)
    hrct_masked = hrct * lung_binary

    # Step 3b: clip to lung window
    hrct_clipped = np.clip(hrct_masked, HU_MIN, HU_MAX)

    # Step 3c: normalize to [0, 1]
    # We use the fixed window min/max (not per-patient min/max)
    # so that intensity values are comparable across patients
    hrct_normalized = (hrct_clipped - HU_MIN) / (HU_MAX - HU_MIN)

    return hrct_normalized.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# STEP 4: Remap tissue labels from 17-class to 5-class
# ─────────────────────────────────────────────────────────────────
def remap_labels(tissue_mask):
    """
    Convert original 0–17 labels to simplified 0–5 scheme.

    Args:
        tissue_mask: [D, H, W] uint8, values 0–17

    Returns:
        [D, H, W] uint8, values 0–5
    """
    remapped = np.zeros_like(tissue_mask, dtype=np.uint8)
    for original_label, new_label in LABEL_REMAP.items():
        remapped[tissue_mask == original_label] = new_label
    return remapped


# ─────────────────────────────────────────────────────────────────
# STEP 5: Extract 3D patches from a preprocessed volume pair
#
# We only keep patches that contain at least one annotated voxel
# (tissue_mask > 0). This avoids storing hundreds of empty patches
# from unannotated background regions.
# ─────────────────────────────────────────────────────────────────
def extract_patches(hrct_vol, mask_vol, min_annotated_voxels=50):
    """
    Slide a window over the 3D volume and extract patches.

    Args:
        hrct_vol             : [D, H, W] float32, normalized
        mask_vol             : [D, H, W] uint8, remapped labels 0–5
        min_annotated_voxels : int — skip patches with fewer annotated voxels

    Returns:
        list of (hrct_patch, mask_patch) tuples
        each patch is shape PATCH_SIZE
    """
    D, H, W = hrct_vol.shape
    pd, ph, pw = PATCH_SIZE
    sd, sh, sw = PATCH_STRIDE

    patches = []

    for d in range(0, max(1, D - pd + 1), sd):
        for h in range(0, max(1, H - ph + 1), sh):
            for w in range(0, max(1, W - pw + 1), sw):

                # Clamp to volume boundaries
                d_end = min(d + pd, D)
                h_end = min(h + ph, H)
                w_end = min(w + pw, W)
                d_start = d_end - pd
                h_start = h_end - ph
                w_start = w_end - pw

                hrct_patch = hrct_vol[d_start:d_end, h_start:h_end, w_start:w_end]
                mask_patch = mask_vol[d_start:d_end, h_start:h_end, w_start:w_end]

                # Only keep patches with meaningful annotation
                n_annotated = np.sum(mask_patch > 0)
                if n_annotated >= min_annotated_voxels:
                    patches.append((hrct_patch.copy(), mask_patch.copy()))

    return patches


# ─────────────────────────────────────────────────────────────────
# MAIN: Process all patients and save patches to disk
# ─────────────────────────────────────────────────────────────────
def preprocess_all_patients(vol_roi_base, lung_mask_base, output_dir):
    """
    Run the full preprocessing pipeline for all usable patients.
    Saves one .npz file per patch: hrct_patch + mask_patch + metadata.

    Output structure:
        output_dir/
            patient_3_patch_0.npz
            patient_3_patch_1.npz
            patient_7_patch_0.npz
            ...
    """
    os.makedirs(output_dir, exist_ok=True)

    patients = sorted(
        [p for p in os.listdir(vol_roi_base)
         if os.path.isdir(os.path.join(vol_roi_base, p))
         and p not in SKIP_PATIENTS
         and p != 'HRCT_pilot'],
        key=lambda x: int(x)
    )

    total_patches = 0
    patient_summary = []

    for pid in patients:
        print(f'\nProcessing patient {pid}...')

        try:
            # ── Step 1: Load raw volumes ──────────────────────────
            hrct, tissue_mask, lung_mask, (ps, st) = load_patient_raw(
                pid, vol_roi_base, lung_mask_base
            )
            print(f'  Loaded  — HRCT: {hrct.shape}  spacing: {ps:.4f}x{st:.1f}mm')

            # ── Step 2: Resample to 0.7mm XY spacing ─────────────
            hrct         = resample_volume(hrct,         ps, is_mask=False)
            tissue_mask  = resample_volume(tissue_mask,  ps, is_mask=True).astype(np.uint8)
            lung_mask    = resample_volume(lung_mask,    ps, is_mask=True).astype(np.uint8)
            print(f'  Resampled — new shape: {hrct.shape}')

            # ── Step 3: Apply lung mask, clip, normalize ──────────
            hrct = preprocess_hrct(hrct, lung_mask)
            print(f'  Normalized — HU range now: [{hrct.min():.3f}, {hrct.max():.3f}]')

            # ── Step 4: Remap labels ──────────────────────────────
            tissue_mask = remap_labels(tissue_mask)
            unique = np.unique(tissue_mask[tissue_mask > 0]).tolist()
            print(f'  Labels remapped — classes present: {unique}')

            # ── Step 5: Extract patches ───────────────────────────
            patches = extract_patches(hrct, tissue_mask)
            print(f'  Patches extracted: {len(patches)}')

            # ── Save each patch as its own .npz ───────────────────
            for i, (hp, mp) in enumerate(patches):
                fname = f'patient_{pid}_patch_{i}.npz'
                np.savez_compressed(
                    os.path.join(output_dir, fname),
                    hrct=hp,          # float32 [16, 128, 128]
                    mask=mp,          # uint8   [16, 128, 128]
                    patient_id=pid,
                    patch_idx=i
                )

            total_patches += len(patches)
            patient_summary.append({
                'pid': pid, 'n_patches': len(patches), 'classes': unique
            })

        except Exception as e:
            print(f'  ERROR — patient {pid}: {e}')
            continue

    # ── Final summary ─────────────────────────────────────────────
    print(f'\n{"="*50}')
    print(f'PREPROCESSING COMPLETE')
    print(f'Patients processed : {len(patient_summary)}')
    print(f'Total patches saved: {total_patches}')
    print(f'Output directory   : {output_dir}')
    print(f'{"="*50}')

    # Save summary as a simple text log
    log_path = os.path.join(output_dir, 'preprocessing_log.txt')
    with open(log_path, 'w') as f:
        f.write(f'Patients processed: {len(patient_summary)}\n')
        f.write(f'Total patches: {total_patches}\n\n')
        for s in patient_summary:
            f.write(f"patient_{s['pid']}: {s['n_patches']} patches, "
                    f"classes={s['classes']}\n")
    print(f'Log saved → {log_path}')


if __name__ == '__main__':

    VOL_ROI_BASE   = '/Users/Elena/Desktop/main/msc/final_project/ILD_DB/ILD_DB_volumeROIs'
    LUNG_MASK_BASE = '/Users/Elena/Desktop/main/msc/final_project/ILD_DB/ILD_DB_lungMasks'
    OUTPUT_DIR     = '/Users/Elena/Desktop/main/msc/final_project/patches'
    preprocess_all_patients(VOL_ROI_BASE, LUNG_MASK_BASE, OUTPUT_DIR)