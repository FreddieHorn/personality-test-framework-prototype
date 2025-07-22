import os
from datetime import datetime
from openai import OpenAI
from logging import getLogger
import logging
from scenario_generation.generation import scenario_generation
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
    # setup_goals("goals_50_ptt.csv", "Human_Goals_List_Clean_Updated.csv", client, num_records=5)
    difficulty = "Medium"
    scenario_generation("goals_50_ptt_FOR_MONDAY.csv", f"scenarios_{difficulty}.csv",client, model="deepseek/deepseek-chat-v3-0324",provider="DeepInfra", difficulty=difficulty)
    # generate interaction
    