#!/bin/bash
#SBATCH --job-name=fixtorch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fix_%A.out

source /home/yao.eric/selective-attack/.venv/bin/activate
which python
nvidia-smi
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"