import os
from datetime import datetime

from scenario_generation import scenario_generation
from interaction_generation import generate_interaction
from evaluation import evaluation

if __name__ == "__main__":
    tempratures = [5]
    interaction_modes = ["default"]
    characters_csvs = ["baseline_steve.csv"]

    print("models declared")
    for temperature in tempratures: # I know that this triple for loop is diabolical and bash script probably would be better
        for inter_mode in interaction_modes:
            for csv in characters_csvs:
                print(f"PROCESSING {csv} || interaction_mode: {inter_mode} || temperature {temperature}")
                date = datetime.now().strftime("%Y-%m-%d_%H-%M")
                folder_name = f"session_{date}/steve_{inter_mode}_{temperature}"
                os.makedirs(f"output/{folder_name}", exist_ok=True)
                step_0_csv_path = csv
                step_1_csv_path = f'output/{folder_name}/scenarios.csv'
                step_2_csv_path = f'output/{folder_name}/interactions.csv'
                step_3_csv_path = f'output/{folder_name}/evaluated_interactions.csv'
                scenario_generation(step_0_csv_path, step_1_csv_path, temperature = temperature)
                generate_interaction(step_1_csv_path, step_2_csv_path, mode=inter_mode)
                evaluation(step_2_csv_path, step_3_csv_path)
