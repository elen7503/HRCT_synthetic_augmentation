#!/bin/bash
# Run this ONCE to create and submit all 5 attention-UNet training jobs.
# Each is independent -- they run on the compute nodes via qsub, not
# interactively, so this is safe to leave running while you write.

cd /rds/general/user/eh1121/home/Final_Project

declare -A CLASS_IDX=( [healthy]=0 [emphysema]=1 [ground_glass]=2 [fibrosis]=3 [micronodules]=4 )

for name in "${!CLASS_IDX[@]}"; do
  idx=${CLASS_IDX[$name]}
  cat > scripts/job_train_attention_${name}.sh << JOBEOF
#!/bin/bash
#PBS -l select=1:ncpus=4:mem=32gb:ngpus=1
#PBS -l walltime=12:00:00
#PBS -N train_attention_${name}
#PBS -o /rds/general/user/eh1121/home/Final_Project/logs/train_attention_${name}.out
#PBS -e /rds/general/user/eh1121/home/Final_Project/logs/train_attention_${name}.err
module load CUDA/12.1.1
source /rds/general/user/eh1121/home/miniforge3/etc/profile.d/conda.sh
conda activate ild
cd /rds/general/user/eh1121/home/Final_Project/diffusion_model
echo "Training ${name} (attention U-Net) | Start: \$(date)"
python train.py --class_idx ${idx} \\
    --epochs 1000 \\
    --save_every 50 \\
    --output_dir outputs_attention
echo "End: \$(date)"
JOBEOF
  echo "Submitting ${name}..."
  qsub scripts/job_train_attention_${name}.sh
done

echo ""
echo "All 5 jobs submitted. Check anytime with: qstat -u eh1121"
echo "This is safe to leave running while you write -- no need to babysit it."