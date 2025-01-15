import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from json_schemas import JSON_SCHEMA_SCENARIO
from llm_funcs import create_scenario
from environment import Agent, AgentInteraction

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )
print(f"Is CUDA available: {torch.cuda.is_available()}\n")
print("Model&Tokenizer declaration...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SETTING = "PROFFESIONAL"
TOPIC = "Office collaboration of a tight deadline"

print(f"Scenario creation with a setting: {SETTING} and a topic: {TOPIC}...")
scenario_config = create_scenario(setting=SETTING, topic=TOPIC, model=model, tokenizer=tokenizer, json_schema=JSON_SCHEMA_SCENARIO)

with open("scenario.json", "w") as file:
    json.dump(scenario_config, file, indent=4)  # 'indent' makes the file more readable

# Below code has errors heh
agents_config = {"agent_1_personality" : "Donald Trump",
                "agent_2_personality" : "Clark Kent"}
interaction_framework = AgentInteraction(agents_config = agents_config,
                                        scenario_config = scenario_config, turns=10)
print(f"Beggining the conversation between two agents...")
conversation = interaction_framework.conduct_interaction()

print(f"Writing the convo to the file")
with open(filename, "w") as file:
    for item in conversation:
        file.write(f"{item}\n")

# Evaluate agents


