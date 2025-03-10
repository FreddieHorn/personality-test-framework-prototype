import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from prompts import scenario_creation_prompt, concept_agent_prompt, narrative_agent_prompt, logical_consistency_agent_prompt, conflict_agent_prompt, goal_agent_prompt

def scenario_generation(input_csv: str, output_csv: str,  model, tokenizer, mode = 'default', temperature = 1):
    # Load the input data
    data = pd.read_csv(input_csv)

    # Process rows
    results = []

    if mode == 'default':
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
    elif mode == 'agentic':
        for _, row in data.iterrows():
            step_1_scenario = concept_agent_prompt(agent1_name = row["Character1"], 
                                                agent2_name= row["Character2"], 
                                                setting=row["Setting"],
                                                topic=row["Topic"],
                                                model=model, tokenizer=tokenizer)
            print(f"Concept agent prompt: \n {step_1_scenario["scenario"]}")
            step_2_scenario = narrative_agent_prompt(step_1_scenario["scenario"], model=model, tokenizer=tokenizer)
            print(f"Narrative agent prompt: \n {step_2_scenario["scenario"]}")
            step_3_scenario = logical_consistency_agent_prompt(step_2_scenario["scenario"], model = model, tokenizer = tokenizer)
            print(f"Logical consistency agent prompt: \n {step_3_scenario["scenario"]}")
            result_scenario = conflict_agent_prompt(step_3_scenario["scenario"], model=model, tokenizer = tokenizer, temperature = temperature)
            print(f"Conflict (Result) agent prompt: \n {step_3_scenario["scenario"]}")
            goal_creation = goal_agent_prompt(result_scenario["scenario"], model=model, tokenizer = tokenizer)
            print(f"Goals: \n {goal_creation}")
            result_scenario.update(goal_creation)
            results.append(result_scenario)
            print(result_scenario) 
     # Save results
    data["scenario"] = [result.get("scenario", "") for result in results]
    data["shared_goal"] = [result.get("shared_goal", "") for result in results]
    data["first_agent_goal"] = [result.get("first_agent_goal", "") for result in results]
    data["second_agent_goal"] = [result.get("second_agent_goal", "") for result in results]

    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")