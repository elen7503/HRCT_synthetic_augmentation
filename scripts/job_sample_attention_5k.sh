#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=02:00:00
#PBS -N sample_attention_5k
#PBS -o $PBS_O_WORKDIR/logs/sample_attention_5k.out
#PBS -e $PBS_O_WORKDIR/logs/sample_attention_5k.err
: "${PROJECT_ROOT:?PROJECT_ROOT no está definida — exporta esta variable antes de lanzar el job}"
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd ${PROJECT_ROOT}/diffusion_model

python sample.py --class_idx 0 --weightfile outputs_attention/healthy/epoch_0100.pt \
    --n_samples 5000 --output_dir outputs_attention/synthetic_5k

python sample.py --class_idx 1 --weightfile outputs_attention/emphysema/epoch_0050.pt \
    --n_samples 5000 --output_dir outputs_attention/synthetic_5k

python sample.py --class_idx 2 --weightfile outputs_attention/ground_glass/epoch_0150.pt \
    --n_samples 5000 --output_dir outputs_attention/synthetic_5k

python sample.py --class_idx 3 --weightfile outputs_attention/fibrosis/epoch_0150.pt \
    --n_samples 5000 --output_dir outputs_attention/synthetic_5k

python sample.py --class_idx 4 --weightfile outputs_attention/micronodules/epoch_0100.pt \
    --n_samples 5000 --output_dir outputs_attention/synthetic_5k
