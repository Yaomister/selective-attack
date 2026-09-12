#!/bin/bash
#SBATCH --job-name=mu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-5

mkdir -p logs

source /home/yao.eric/selective-attack/.venv/bin/activate

LAYERS=(-1 -2 -4 -8 -16)   
LAYER=${LAYERS[$SLURM_ARRAY_TASK_ID]}

python attack/experiment.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/pooling_method_$POOL \
  --steps 200 \
  --epsilon 1 \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last $LAYER \
  --pooling_method last_token