#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --output=spg_result_%j.txt
#SBATCH --error=spg_VCOCOOD_%j.txt
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=mmlab_example
#SBATCH --account=PCS0269

# module load python/3.9
# module load cuda/11.6.2

# source ~/ascebd_env/bin/activate
source activate pocket
cd ~/spatially-conditioned-graphs

CUDA_VISIBLE_DEVICES=0 python cache.py --dataset vcoco --data-root vcoco \
    --detection-dir vcoco/detections/test \
    --cache-dir vcoco_cache --partition test \
    --model-path /users/PCS0256/lijing/spatially-conditioned-graphs/checkpoints/vcoco/transh_50/ckpt_03732_12.pt