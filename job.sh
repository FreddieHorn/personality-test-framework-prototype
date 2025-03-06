#!/bin/bash
#SBATCH --partition=A100medium
#SBATCH --time=7:00:00
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --job-name=scenario_experiment
#SBATCH --output=logs/slurm_%j.out

module load Python
module load CUDA

# Define possible values for each parameter
temperatures=(1 2 3 4 5)
scenario_methods=("default" "agentic")
interaction_methods=("default" "scriptish")
csv_dir="data/"  # Directory containing CSV files

# Iterate over all combinations
for base_csv in "$csv_dir"*.csv; do
    for temp in "${temperatures[@]}"; do
        for scenario_method in "${scenario_methods[@]}"; do
            for interaction_method in "${interaction_methods[@]}"; do
                csv_name=$(basename "$base_csv" .csv)
                echo "Running with base_csv=$csv_name, temperature=$temp, scenario_generation_method=$scenario_method, interaction_generation_method=$interaction_method"
                srun torchrun --standalone --nproc_per_node 1 pipeline.py --temperature "$temp" \
                                --scenario_generation_method "$scenario_method" \
                                --interaction_generation_method "$interaction_method" \
                                --base_csv "$base_csv"
            done
        done
    done
done

echo "All combinations executed."