#!/bin/bash
#SBATCH --job-name=experiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%A.out


mkdir -p logs

source /home/yao.eric/selective-attack/.venv/bin/activate

python attack/experiment.py \
  --model_name LLaVA-1.5-7b \
  --dataset_dir ./sorted \
  --output_dir ./attack_results \
  --steps 200 \
  --epsilon 0.03 \
  --alpha 0.001 \
  --mu 10.0 \
  --layer_from_last -1 \
  --pooling_method last_token