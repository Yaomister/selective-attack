#!/bin/bash
#SBATCH --job-name=mu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-3

mkdir -p logs

source /home/yao.eric/selective-attack/.venv/bin/activate

MODELS=(LLaVA-1.5-7b LLaVA-NeXT InternVL Qwen-VL)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

python experiments/experiment_v3.py \
  --model_name $MODEL \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/model_$MODEL \
  --steps 200 \
  --epsilon 1 \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token