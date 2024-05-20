#!/bin/bash
#SBATCH --gres=gpu:2

# module load cuda/12.1
# module load cudnn/9.1.0_cu12x
module load anaconda/2022.10
source activate LLM
python test2.py
