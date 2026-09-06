#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N generate_pool_v5
#PBS -o $PBS_O_WORKDIR/logs/generate_pool_v5.out
#PBS -e $PBS_O_WORKDIR/logs/generate_pool_v5.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model

echo "=========================================="
echo "Generating LARGER pool (8000/class) for quality filtering (v5)"
echo "Start: $(date)"
echo "=========================================="

POOL_SIZE=8000

python sample.py --class_idx 0 --weightfile outputs/healthy/epoch_0300.pt \
    --n_samples $POOL_SIZE --output_dir outputs/synthetic_v5_pool

python sample.py --class_idx 1 --weightfile outputs/emphysema/epoch_2000.pt \
    --n_samples $POOL_SIZE --output_dir outputs/synthetic_v5_pool

python sample.py --class_idx 2 --weightfile outputs/ground_glass/epoch_0200.pt \
    --n_samples $POOL_SIZE --output_dir outputs/synthetic_v5_pool

python sample.py --class_idx 3 --weightfile outputs/fibrosis/epoch_0100.pt \
    --n_samples $POOL_SIZE --output_dir outputs/synthetic_v5_pool

python sample.py --class_idx 4 --weightfile outputs/micronodules/epoch_0100.pt \
    --n_samples $POOL_SIZE --output_dir outputs/synthetic_v5_pool

echo "=========================================="
echo "End: $(date)"
echo "=========================================="