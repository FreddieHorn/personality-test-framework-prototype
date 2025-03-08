import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scenario_generation import scenario_generation
from interaction_generation import generate_interaction
from evaluation import evaluation

if __name__ == "__main__":
    tempratures = [1,2,3,4,5]
    interaction_modes = ["default", "script"]
    characters_csvs = ["baseline_all.csv"]
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Is CUDA available: {torch.cuda.is_available()}\n")
    print("Model&Tokenizer declaration...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("models declared")
    for temperature in tempratures: # I know that this triple for loop is diabolical and bash script probably would be better
        for inter_mode in interaction_modes:
            for csv in characters_csvs:
                print(f"PROCESSING {csv} || interaction_mode: {inter_mode} || temperature {temperature}")
                date = datetime.now().strftime("%Y-%m-%d_%H-%M")
                folder_name = f"session_{date}/{csv.split("_")[1].split(".")[0]}_{inter_mode}_{temperature}"
                os.makedirs(f"output/{folder_name}", exist_ok=True)
                step_0_csv_path = csv
                step_1_csv_path = f'output/{folder_name}/scenarios.csv'
                step_2_csv_path = f'output/{folder_name}/interactions.csv'
                step_3_csv_path = f'output/{folder_name}/evaluated_interactions.csv'
                scenario_generation(step_0_csv_path, step_1_csv_path, model, tokenizer, temperature = temperature)
                generate_interaction(step_1_csv_path, step_2_csv_path, model, tokenizer, mode=inter_mode)
                evaluation(step_2_csv_path, step_3_csv_path, model, tokenizer)
