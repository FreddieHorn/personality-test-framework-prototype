import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
from prompts import choose_goal_category_prompt, extrapolate_goals_prompt, generate_roles_prompt
from logging import getLogger
from openai import OpenAI
from utils import sample_shared_goal, sample_unique_shared_goals_excluding_existing
import os

log = getLogger(__name__)


def setup_goals(output_csv: str, goal_list_csv: str, client: OpenAI, num_records: int = 10):
    results = []

    log.info(f"Generating {num_records} records of goal data")
    

    # Check if output file exists to resume progress
    if os.path.exists(output_csv):
        result_df = pd.read_csv(output_csv)
        results = result_df.to_dict('records')
        existing_shared_goals = set(result_df["shared_goal"].dropna().unique())
        log.info(f"Resuming from existing file with {len(results)} records")
            # Use sampling function that excludes existing goals
        base_shared_goals = sample_unique_shared_goals_excluding_existing(
            goal_list_csv,
            existing_shared_goals,
            num_samples=num_records
        )
    else:
        base_shared_goals = sample_shared_goal(goal_list_csv, num_samples=num_records)
        results = []

    for i, base_shared_goal in enumerate(base_shared_goals, start=len(results) + 1):
        try:
            # Generate data for each record
            goal_category = choose_goal_category_prompt(
                base_shared_goal=base_shared_goal,
                client=client,
                model_name="deepseek/deepseek-chat-v3-0324",
                provider="DeepInfra"
            )

            results.append({
                "base_shared_goal": base_shared_goal,
                "social_goal_category": goal_category["chosen_social_goal_category"],
                "explanation": goal_category["explanation"],
                "first_agent_goal": goal_category["first_agent_goal"],
                "second_agent_goal": goal_category["second_agent_goal"],
                "shared_goal": base_shared_goal["Full label"],
                "agent1_role": goal_category["agent1_role"],
                "agent2_role": goal_category["agent2_role"],
            })

            # Save after each iteration
            pd.DataFrame(results).to_csv(output_csv, index=False)
            log.info(f"Saved record {i}/{len(results)} to {output_csv}")

        except Exception as e:
            log.error(f"Error processing record {i}: {str(e)}")
            # Save progress up to this point
            pd.DataFrame(results).to_csv(output_csv, index=False)
            log.info(f"Saved progress up to record {i} before error")
            raise  # Re-raise the exception if you want to stop execution

    log.info(f"Successfully saved all {len(results)} total records to {output_csv}")


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