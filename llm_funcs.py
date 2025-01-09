# Example topics where personality matters:
# Conflict resolution, Teacher-student roleplaying - trying to get a better grade, Office collaboration of a tight deadline, 
# Debating climate change solutions, handling a technical failure in a project.
import torch
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompts import initiate_scenario_prompt, rate_agents_prompt

chat_template = """### System:
{system_message}
### User:
{user_message}
"""
def create_scenario(topic, model, tokenizer, json_schema):
    messages = initiate_scenario_prompt(topic)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()
    return generated_data

def rate_agents(episode: list, scenario_questions: dict, model, tokenizer, json_schema):
    messages = rate_agents_prompt(topic)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()
    return generated_data