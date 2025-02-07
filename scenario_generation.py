import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from prompts import scenario_creation_prompt

def scenario_generation(input_csv: str, output_csv: str,  model, tokenizer, temperature = 1):
    # Load the input data
    data = pd.read_csv(input_csv)

    # Process rows
    results = []
    for _, row in data.iterrows():
        result = scenario_creation_prompt(
            setting=row["Setting"],
            topic=row["Topic"],
            agent_1_name = row["Character1"],
            agent_2_name = row["Character2"],
            temperature = temperature,
            model = model,
            tokenizer = tokenizer
        )
        results.append(result)
        print(result) 
     # Save results
    data["scenario"] = [result.get("scenario", "") for result in results]
    data["shared_goal"] = [result.get("shared_goal", "") for result in results]
    data["first_agent_goal"] = [result.get("first_agent_goal", "") for result in results]
    data["second_agent_goal"] = [result.get("second_agent_goal", "") for result in results]

    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")