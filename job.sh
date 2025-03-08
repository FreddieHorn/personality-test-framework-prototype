#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
module load Python
module load CUDA
srun torchrun --standalone --nproc_per_node 1 pipeline.py