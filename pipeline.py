import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from scenario_generation import scenario_generation
from interaction_generation import generate_interaction
from evaluation import evaluation

if __name__ == "__main__":
    
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    # Get the current date and time formatted as YYYY-MM-DD_HH-MM
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(folder_name, exist_ok=True)

    # CONFIG - move to somewhere pls
    step_0_csv_path = 'baseline.csv'
    step_1_csv_path = f'output/{folder_name}/scenarios.csv'
    step_2_csv_path = f'output/{folder_name}/interactions.csv'
    step_3_csv_path = f'output/{folder_name}/evaluated_interactions.csv'
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=False,
    #         bnb_4bit_compute_dtype=torch.float16
    #     )
    print(f"Is CUDA available: {torch.cuda.is_available()}\n")
    print("Model&Tokenizer declaration...")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("models declared")
    scenario_generation(step_0_csv_path, step_1_csv_path, model, tokenizer)
    generate_interaction(step_1_csv_path, step_2_csv_path, model, tokenizer)
    evaluation(step_2_csv_path, step_3_csv_path, model, tokenizer)