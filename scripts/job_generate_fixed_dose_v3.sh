#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N generate_fixed_dose_v3
#PBS -o $PBS_O_WORKDIR/logs/generate_fixed_dose_v3.out
#PBS -e $PBS_O_WORKDIR/logs/generate_fixed_dose_v3.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model

echo "=========================================="
echo "Generating FIXED dose (3000 synthetic patches) per class"
echo "from quality-selected checkpoints, isolating quantity from quality"
echo "Start: $(date)"
echo "=========================================="

FIXED_DOSE=3000

# healthy: epoch_0300
python sample.py --class_idx 0 --weightfile outputs/healthy/epoch_0300.pt \
    --n_samples $FIXED_DOSE --output_dir outputs/synthetic_v3

# emphysema: epoch_2000
python sample.py --class_idx 1 --weightfile outputs/emphysema/epoch_2000.pt \
    --n_samples $FIXED_DOSE --output_dir outputs/synthetic_v3

# ground_glass: epoch_0200
python sample.py --class_idx 2 --weightfile outputs/ground_glass/epoch_0200.pt \
    --n_samples $FIXED_DOSE --output_dir outputs/synthetic_v3

# fibrosis: epoch_0100
python sample.py --class_idx 3 --weightfile outputs/fibrosis/epoch_0100.pt \
    --n_samples $FIXED_DOSE --output_dir outputs/synthetic_v3

# micronodules: epoch_0100 -- NEW, never generated before (v2 didn't need it for balancing)
python sample.py --class_idx 4 --weightfile outputs/micronodules/epoch_0100.pt \
    --n_samples $FIXED_DOSE --output_dir outputs/synthetic_v3

echo "=========================================="
echo "End: $(date)"
echo "=========================================="