#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=00:40:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
module load Python
module load CUDA
srun torchrun --standalone --nproc_per_node 1 pipeline.py