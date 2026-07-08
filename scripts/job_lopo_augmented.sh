#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N lopo_augmented
#PBS -o /rds/general/user/eh1121/home/Final_Project/logs/lopo_augmented.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/logs/lopo_augmented.err

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild

cd /rds/general/user/eh1121/home/Final_Project/ILD-Segmentation-And-Classification-DL

echo "=========================================="
echo "Job: LOPO Augmented (real + synthetic)"
echo "Start: $(date)"
echo "=========================================="

python Supplementary_materials/classification_lopo_augmented.py
    --imgs_path /rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_images.npy \
    --lbls_path /rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_labels.npy \
    --pids_path /rds/general/user/eh1121/home/Final_Project/ILD_DB_npy_augmented/all_patient_ids.npy

echo "=========================================="
echo "End: $(date)"
echo "=========================================="
