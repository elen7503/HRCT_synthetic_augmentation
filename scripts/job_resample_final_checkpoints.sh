#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N resample_final_checkpoints
#PBS -o $PBS_O_WORKDIR/logs/resample_final_checkpoints.out
#PBS -e $PBS_O_WORKDIR/logs/resample_final_checkpoints.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model

echo "=========================================="
echo "Resampling all classes from quality-selected checkpoints"
echo "Start: $(date)"
echo "=========================================="

# healthy: epoch_0300, target 470 synthetic
python sample.py --class_idx 0 --weightfile outputs/healthy/epoch_0300.pt \
    --n_samples 470 --output_dir outputs/synthetic_v2

# emphysema: epoch_2000, target 4823 synthetic
python sample.py --class_idx 1 --weightfile outputs/emphysema/epoch_2000.pt \
    --n_samples 4823 --output_dir outputs/synthetic_v2

# ground_glass: epoch_0200, target 3774 synthetic
python sample.py --class_idx 2 --weightfile outputs/ground_glass/epoch_0200.pt \
    --n_samples 3774 --output_dir outputs/synthetic_v2

# fibrosis: epoch_0100, target 2961 synthetic
python sample.py --class_idx 3 --weightfile outputs/fibrosis/epoch_0100.pt \
    --n_samples 2961 --output_dir outputs/synthetic_v2

# micronodules: no synthetic needed (real=6821 already exceeds ~6000 target)

echo "=========================================="
echo "End: $(date)"
echo "=========================================="