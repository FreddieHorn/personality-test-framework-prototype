import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from prompts import generate_interaction_prompt
from json_schemas import JSON_SCHEMA_SCENARIO
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from jsonformer import Jsonformer

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
    input_csv = 'scenario.csv' # output of the previous step
    output_csv = 'pipeline_output.csv' # output of the whole pipeline (for now)
    data = pd.read_csv(input_csv)

    # Process rows
    results = []
    for _, row in data.iterrows():
        result = llama_3_prompt(
            agent1=row["Character1"],
            agent2=row["Character2"],
            goal=row["shared_goal"],
            scenario=row["scenario"],
            personality1=row["Personality1"],
            personality2=row["Personality2"],
            setting = row["Setting"],
            topic=row["Topic"],
            model = model,
            tokenizer = tokenizer
        )
        results.append(result)
        print(result) 
     # Save results
    data["interaction"] = [result.get("interaction", "") for result in results]
    
    data.to_csv(input_csv, index=False)
    print(f"Results saved to {input_csv}")


if __name__ == "__main__":
    main()