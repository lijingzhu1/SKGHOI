#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=4
#SBATCH --output=result/spg_result_%j.txt
#SBATCH --error=error/spg_hicodetOD_%j.txt
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --job-name=mmlab_example
#SBATCH --account=PCS0269

# module load python/3.9
# module load cuda/11.6.2

# source ~/ascebd_env/bin/activate
source activate pocket
cd ~/spatially-conditioned-graphs

CUDA_VISIBLE_DEVICES=0 python test/adamixer_transH_spatital_r50_test.py --model-path checkpoints/hicodet/new_spatial/transh_50/ckpt_28236_12.pt