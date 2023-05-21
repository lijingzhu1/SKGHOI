#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=4
#SBATCH --output=vcoco_test_result_%j.txt
#SBATCH --error=vcoco_test_error_%j.txt
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --job-name=mmlab_example
#SBATCH --account=PCS0269

# module load python/3.9
# module load cuda/11.6.2

# source ~/ascebd_env/bin/activate
source activate pocket
cd ~/spatially-conditioned-graphs

python vcoco_evaluation.py