import os
from datetime import datetime
from openai import OpenAI
from scenario_generation import scenario_generation
from interaction_generation import generate_interaction
from evaluation import evaluation, evaluate_scenarios, evaluate_interactions
from goal_setup import setup_goals
from logging import getLogger
import logging
from dotenv import load_dotenv
logging.basicConfig(level = logging.INFO)
log = getLogger(__name__)

load_dotenv()
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if __name__ == "__main__":
    tempratures = [4,5]
    interaction_modes = ["default", "script"]
    scenario_generation_modes = ["agentic"]
    characters_csvs = ["baseline_test.csv"]
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    log.info("Model&Tokenizer declaration...")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPEN_ROUTER_API_KEY
    )
    date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    prefix = f"results_may"
    os.makedirs(f"output/{prefix}", exist_ok=True)
    log.info("Starting pipeline...")
    # setup_goals("goals_10.csv", client, num_records=10)
    # for temperature in tempratures: # I know that this triple for loop is diabolical and bash script probably would be better
    #     for scengen_mode in scenario_generation_modes:
    #         for inter_mode in interaction_modes:
    #             for csv in characters_csvs:
    #                 log.info(f"PROCESSING {csv} || scenario_mode {scengen_mode} || interaction_mode: {inter_mode} || temperature {temperature}")
    #                 date = datetime.now().strftime("%Y-%m-%d_%H-%M")
    #                 folder_name = f"{prefix}/{csv.split('_')[1].split('.')[0]}_{scengen_mode}_{inter_mode}_{temperature}"
    #                 os.makedirs(f"output/{folder_name}", exist_ok=True)
    step_0_csv_path = f"output/{prefix}/baseline_test_goals.csv"
    step_1_csv_path = f"output/{prefix}/scenarios.csv"
    #                 # step_2_csv_path = f'output/{folder_name}/interactions.csv'
    #                 # step_3_csv_path = f'output/{folder_name}/evaluated_interactions.csv'
    #                 # evaluate_scenarios_path = f'output/{folder_name}/evaluated_scenarios_v2.csv'
    #                 # evaluated_interactions_path = f'output/{folder_name}/evaluated_interactions2.csv'
    scenario_generation('goals_10.csv', 'scenarios_10.csv', client, mode='agentic', temperature = 3)
                    # generate_interaction(step_1_csv_path, step_2_csv_path, model, tokenizer, mode=inter_mode)
                    # evaluation(step_2_csv_path, step_3_csv_path, model, tokenizer)
                    # evaluate_scenarios(step_3_csv_path, evaluate_scenarios_path, model, tokenizer)
                    # evaluate_interactions(step_3_csv_path, evaluated_interactions_path, model, tokenizer)