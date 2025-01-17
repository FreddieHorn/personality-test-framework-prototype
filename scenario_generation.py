import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from json_schemas import JSON_SCHEMA_SCENARIO
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from jsonformer import Jsonformer
from prompts import scenario_creation_prompt

def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
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

    # Load the input data
    input_csv = 'pesonality_test_Sheet.csv'
    output_csv = 'scenarios.csv'
    data = pd.read_csv(input_csv)

    # Process rows
    results = []
    for _, row in data.iterrows():
        result = scenario_creation_prompt(
            setting=row["Setting"],
            topic=row["Topic"],
            model = model,
            tokenizer = tokenizer
        )
        results.append(result)
        print(result) 
     # Save results
    data["scenario"] = [result.get("scenario", "") for result in results]
    data["shared_goal"] = [result.get("shared_goal", "") for result in results]
    
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    main()