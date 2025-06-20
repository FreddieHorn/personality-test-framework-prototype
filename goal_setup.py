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
        base_shared_goal = sample_shared_goal("Human_Goals_List_Clean_Updated.csv")
        # print(f"Base Shared Goal: {base_shared_goal}")
        goal_category = choose_goal_category_prompt(base_shared_goal=base_shared_goal, client=client, model_name="deepseek/deepseek-chat-v3-0324")
        print(f"Chosen Goal Category: {goal_category}")
        # print(f"First Agent Goal: {goal_category['first_agent_goal']}, Second Agent Goal: {goal_category['second_agent_goal']}")

        # roles = generate_roles_prompt(
        #     goal_category=goal_category["chosen_social_goal_category"],
        #     client=client,
        #     model_name="deepseek/deepseek-chat-v3-0324:free",
        # )
        # print(f"Generated Roles: {roles}")
        # Store results
        results.append({
            "base_shared_goal": base_shared_goal,
            "social_goal_category": goal_category["chosen_social_goal_category"],
            "first_agent_goal": goal_category["first_agent_goal"],
            "second_agent_goal": goal_category["second_agent_goal"],
            "shared_goal": base_shared_goal["base_goal_shared_full_label"],
            "agent1_role": goal_category["agent1_role"],
            "agent2_role": goal_category["agent2_role"]
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