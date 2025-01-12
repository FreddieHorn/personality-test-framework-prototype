import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from json_schemas import JSON_SCHEMA_SCENARIO
from llm_funcs import create_scenario
from environment import Agent, AgentInteraction

MODEL_NAME = "meta-llama/Meta-Llama-3-70B-Instruct"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16
    )
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

SETTING = "PROFFESIONAL"
TOPIC = "Office collaboration of a tight deadline"

scenario_config = create_scenario(setting=SETTING, topic=TOPIC, model=model, tokenizer=tokenizer, json_schema=JSON_SCHEMA_SCENARIO)

agents_config = {"agent_1_personality" : "Donald Trump",
                "agent_2_personality" : "Clark Kent"}
interaction_framework = AgentInteraction(agents_config = agents_config,
                                        scenario_config = scenario_config)
conversation = interaction_framework.conduct_interaction()
# agent_1 = Agent(character_name = "Donald Trump", 
#                 scenario = scenario_config["description"],
#                 goal = scenario_config["agent_1_goal"],
#                 js)


