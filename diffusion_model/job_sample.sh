#!/bin/bash
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -N ddpm_sample
#PBS -o /rds/general/user/eh1121/home/Final_Project/diffusion_model/logs/sample_${CLASS_IDX}.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/diffusion_model/logs/sample_${CLASS_IDX}.err

module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild

cd /rds/general/user/eh1121/home/Final_Project/diffusion_model

CLASS_IDX=${CLASS_IDX:-1}
declare -A N_SAMPLES
N_SAMPLES[0]=470
N_SAMPLES[1]=4823
N_SAMPLES[2]=3774
N_SAMPLES[3]=2961
N_SAMPLES[4]=0

N=${N_SAMPLES[$CLASS_IDX]}

echo "Sampling $N patches for class $CLASS_IDX"

python sample.py \
    --class_idx  $CLASS_IDX \
    --weightfile /rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/$(python3 -c "print({0:'healthy',1:'emphysema',2:'ground_glass',3:'fibrosis',4:'micronodules'}[$CLASS_IDX])")/best.pt \
    --n_samples  $N \
    --batch_size 128 \
    --timesteps  1000 \
    --output_dir /rds/general/user/eh1121/home/Final_Project/diffusion_model/outputs/synthetic

echo "Done."
