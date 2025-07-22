import logging
import pandas as pd
import numpy as np
import re
from logging import getLogger
from openai import OpenAI
from scenario_generation.multi_agent_scenario_generation import MultiAgentScenarioGeneration
from scenario_generation.templates import scenario_creation_prompt
import os
logging.basicConfig(level = logging.INFO)
log = getLogger(__name__)

def scenario_generation(input_csv: str, output_csv: str, client: OpenAI, model = "deepseek/deepseek-chat-v3-0324:free", difficulty = 'Medium', provider = None, agentic_generation: bool = False):
    # Load the input data
    data = pd.read_csv(input_csv)
    print(f"Model: {model}, Provider: {provider}, Difficulty: {difficulty}, Agentic Generation: {agentic_generation}")
    # Check if output file exists to resume progress
    if os.path.exists(output_csv):
        result_df = pd.read_csv(output_csv)
        results = result_df.to_dict('records')
        existing_indices = set(result_df.index)
        log.info(f"Resuming from existing file with {len(results)} records")
    else:
        # Create empty output file with headers
        pd.DataFrame(columns=data.columns.tolist() + ['scenario']).to_csv(output_csv, index=False)
        results = []
        existing_indices = set()

    log.info(f"Generating Scenarios for difficulty {difficulty}")

    if not agentic_generation:
        for idx, row in data.iterrows():
            if idx in existing_indices:
                log.info(f"Skipping already processed row {idx}")
                continue

            try:
                scenario_setting = {
                    "shared_goal": row["shared_goal"],
                    "chosen_goal_category": row["social_goal_category"],
                    "first_agent_goal": row["first_agent_goal"],
                    "second_agent_goal": row["second_agent_goal"],
                    "first_agent_role": row["agent1_role"],
                    "second_agent_role": row["agent2_role"]
                }
                
                result = scenario_creation_prompt(
                    scenario_setting=scenario_setting,
                    difficulty=difficulty,
                    client=client,
                    model_name=model,
                    provider=provider
                )
                
                # Create output record
                output_record = row.to_dict()
                output_record['scenario'] = result
                results.append(output_record)
                
                # Save after each iteration
                pd.DataFrame(results).to_csv(output_csv, index=False)
                log.info(f"Saved row {idx + 1}/{len(data)} to {output_csv}")
                log.info(f"Scenario: {result}")

            except Exception as e:
                log.error(f"Error processing row {idx}: {str(e)}")
                # Save progress up to this point
                if results:  # Only save if we have some results
                    pd.DataFrame(results).to_csv(output_csv, index=False)
                    log.info(f"Saved progress up to row {idx} before error")
                raise  # Re-raise the exception to stop execution

        log.info(f"Successfully processed all {len(data)} rows to {output_csv}")
    else:
        multi_agent_scenario_gen = MultiAgentScenarioGeneration(model_name="deepseek/deepseek-chat-v3-0324:free")
        for _, row in data.iterrows():
            scenario_setting = {
                "shared_goal": row["shared_goal"],
                "chosen_goal_category": row["social_goal_category"],
                "first_agent_goal": row["first_agent_goal"],
                "second_agent_goal": row["second_agent_goal"],
                "first_agent_role": row["agent1_role"],
                "second_agent_role": row["agent2_role"]
            }
            result_scenario = multi_agent_scenario_gen.generate_scenario(scenario_setting, "Medium")
            result_row = row.to_dict()
            result_row.update({
                "scenario": result_scenario,
            })
            pd.DataFrame([result_row]).to_csv(
                output_csv, 
                mode='a', 
                header=False, 
                index=False
            )