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

EPSILONS=(0.25 0.5 0.75 1 1.25)
EPSILON=${EPSILONS[$SLURM_ARRAY_TASK_ID]}

python attack/experiment.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results/epsilon_$EPSILON \
  --steps 200 \
  --epsilon $EPSILON \
  --alpha 0.001 \
  --mu 10 \
  --layer_from_last -1 \
  --pooling_method last_token