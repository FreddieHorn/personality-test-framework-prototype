import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
from prompts import choose_goal_prompt, extrapolate_goals_prompt, generate_roles_prompt
from logging import getLogger
from openai import OpenAI
from utils import sample_2_goals

log = getLogger(__name__)


def setup_goals(input_csv: str, output_csv: str, client: OpenAI):
    # Load the input data
    data = pd.read_csv(input_csv)

    # Process rows
    results = []
    log.info(f"Sampling goals, choosing goal categories and extrapolating goals for {len(data)} rows")
    for _, row in data.iterrows():
        base_goals = sample_2_goals("Human_Goals_List_Clean.csv")
        print(base_goals)
        goal_category = choose_goal_prompt(base_goals=base_goals, client=client)
        print(goal_category)
        extrapolated_goals = extrapolate_goals_prompt(base_goals=base_goals, goal_category=goal_category["chosen_goal_category"], client=client)
        print(extrapolated_goals)
        roles = generate_roles_prompt(agent1_name=row["Character1"], agent2_name=row["Character2"], goal_category=goal_category["chosen_goal_category"], client=client)
        print(roles)
        results.append({
            "base_goals": base_goals,
            "goal_category": goal_category["chosen_goal_category"],
            "first_agent_goal": extrapolated_goals["first_agent_extrapolated_goal"],
            "second_agent_goal": extrapolated_goals["second_agent_extrapolated_goal"],
            "shared_goal": extrapolated_goals["shared_goal"],
            "agent1_role": roles["agent1_role"],
            "agent2_role": roles["agent2_role"]
        })
    # Save results
    data["base_goals"] = [result["base_goals"] for result in results]
    data["goal_category"] = [result["goal_category"] for result in results]
    data["first_agent_goal"] = [result["first_agent_goal"] for result in results]
    data["second_agent_goal"] = [result["second_agent_goal"] for result in results]
    data["shared_goal"] = [result["shared_goal"] for result in results]
    data["agent1_role"] = [result["agent1_role"] for result in results]
    data["agent2_role"] = [result["agent2_role"] for result in results]
    
    data.to_csv(output_csv, index=False)
    log.info(f"Results saved to {output_csv}")
    


    # elif mode == 'agentic':
    #     for _, row in data.iterrows():
    #         agent1_name = row["Character1"], 
    #         agent2_name= row["Character2"], 
    #         setting=row["Setting"],
    #         topic=row["Topic"],
    #         step_1_scenario = concept_agent_prompt(agent1_name, 
    #                                             agent2_name, 
    #                                             setting,
    #                                             topic,
    #                                             client=client)
    #         shared_goal = step_1_scenario["shared_goal"]
    #         first_agent_goal = step_1_scenario["first_agent_goal"]
    #         second_agent_goal = step_1_scenario["second_agent_goal"]
    #         step_2_scenario = narrative_agent_prompt(step_1_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, client=client)
    #         step_3_scenario = conflict_agent_prompt(step_2_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, client=client, temperature=temperature)
    #         result_scenario = logical_consistency_agent_prompt(step_3_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, client=client)
    #         results.append(result_scenario)