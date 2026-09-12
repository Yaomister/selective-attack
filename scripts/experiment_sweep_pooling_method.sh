#!/bin/bash
#SBATCH --job-name=mu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-2

mkdir -p logs

source /home/yao.eric/selective-attack/.venv/bin/activate

POOLS=(last_token mean image_only)  
POOL=${POOLS[$SLURM_ARRAY_TASK_ID]}

python attack/experiment.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/pooling_method_$POOL \
  --steps 200 \
  --epsilon 1 \
  --alpha 0.001 \
  --mu $MU \
  --layer_from_last -1 \
  --pooling_method $POOL