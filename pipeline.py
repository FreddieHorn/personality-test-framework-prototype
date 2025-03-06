import os
import argparse
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scenario_generation import scenario_generation
from interaction_generation import generate_interaction
from evaluation import evaluation

def main():
    parser = argparse.ArgumentParser(description="Run scenario generation pipeline with configurable parameters.")
    parser.add_argument("--temperature", type=int, default=1, help="Temperature for scenario generation (default: 1)")
    parser.add_argument("--scenario_generation_method", type=str, default="default", help="Scenario generation method (default: 'default')")
    parser.add_argument("--interaction_generation_method", type=str, default="default", help="Interaction generation method (default: 'default')")
    parser.add_argument("--base_csv", type=str, required=True, help="Path to the base CSV file")
    
    args = parser.parse_args()
    
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    folder_name = datetime.now().strftime("%Y-%m-%d_%H-%M")
    os.makedirs(f"output/{folder_name}_temp_{args.temperature}_scenario_{args.scenario_generation_method}_interaction_{args.interaction_generation_method}", exist_ok=True)
    
    step_1_csv_path = f'output/{folder_name}/scenarios.csv'
    step_2_csv_path = f'output/{folder_name}/interactions.csv'
    step_3_csv_path = f'output/{folder_name}/evaluated_interactions.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Is CUDA available: {torch.cuda.is_available()}\n")
    print("Model & Tokenizer declaration...")
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    print("Models declared")
    
    scenario_generation(args.base_csv, step_1_csv_path, model, tokenizer, 
                        temperature=args.temperature, generation_method=args.scenario_generation_method)
    generate_interaction(step_1_csv_path, step_2_csv_path, model, tokenizer, 
                         generation_method=args.interaction_generation_method)
    evaluation(step_2_csv_path, step_3_csv_path, model, tokenizer)
    
if __name__ == "__main__":
    main()

