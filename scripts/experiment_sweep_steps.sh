#!/bin/bash
#SBATCH --job-name=mu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/mu_%A_%a.out
#SBATCH --array=0-4

mkdir -p logs

source /home/yao.eric/selective-attack/.venv/bin/activate

STEPS=(50 100 200 400 800)  
STEP=${STEPS[$SLURM_ARRAY_TASK_ID]}

python experiments/experiment_v3.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/STEP_$STEP \
  --steps $STEP \
  --epsilon 0.025 \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token