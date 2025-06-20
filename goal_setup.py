import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
from prompts import choose_goal_category_prompt, extrapolate_goals_prompt, generate_roles_prompt
from logging import getLogger
from openai import OpenAI
from utils import sample_shared_goal

log = getLogger(__name__)


def setup_goals(output_csv: str, client: OpenAI, num_records: int = 10):
    results = []
    
    log.info(f"Generating {num_records} records of goal data")
    
    for _ in range(num_records):
        # Generate data for each record
        base_shared_goal = sample_shared_goal("Human_Goals_List_Clean.csv")
        print(f"Base Shared Goal: {base_shared_goal}")
        goal_category = choose_goal_category_prompt(base_shared_goal=base_shared_goal, client=client)
        print(f"Chosen Goal Category: {goal_category['chosen_goal_category']}")
        extrapolated_goals = extrapolate_goals_prompt(
            base_shared_goal=base_shared_goal,
            goal_category=goal_category["chosen_goal_category"],
            client=client
        )
        print(f"Extrapolated Goals: {extrapolated_goals}")

        roles = generate_roles_prompt(
            goal_category=goal_category["chosen_goal_category"],
            client=client
        )
        print(f"Generated Roles: {roles}")
        # Store results
        results.append({
            "base_shared_goal": base_shared_goal,
            "goal_category": goal_category["chosen_goal_category"],
            "first_agent_goal": extrapolated_goals["first_agent_extrapolated_goal"],
            "second_agent_goal": extrapolated_goals["second_agent_extrapolated_goal"],
            "shared_goal": extrapolated_goals["shared_goal"],
            "agent1_role": roles["agent1_role"],
            "agent2_role": roles["agent2_role"]
        })

    # Convert to DataFrame and save
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv, index=False)
    log.info(f"Successfully saved {num_records} records to {output_csv}")
    


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