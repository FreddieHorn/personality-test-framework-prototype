# Example topics where personality matters:
# Conflict resolution, Teacher-student roleplaying - trying to get a better grade, Office collaboration of a tight deadline, 
# Debating climate change solutions, handling a technical failure in a project.
import torch
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def initiate_scenario(topic: str):

    json_format = {
        "topic" : "Topic of the scenario, in which agents participate", 
        "description" : "Detailed description of the scenario",
        "agent_1_goal" : "Goal(s), which agent 1 has to achieve in the scenario",
        "agent_2_goal" : "Goal(s), which agent 2 has to achieve in the scenario",
        "personality_test_points_question" : {"Openness": "In which aspect of the scenario, will the openness of the agents be tested? How is the scenario testing agents when it comes to openness?",
        "Conscientiousness": "In which aspect of the scenario, will the conscientiousness of the agents be tested? How is the scenario testing agents when it comes to conscientiousness?",
        "Agreeableness" : "In which aspect of the scenario, will the agreeableness of the agents be tested? How is the scenario testing agents when it comes to agreeableness?",
        "Extroversion" : "In which aspect of the scenario, will the extroversion of the agents be tested? How is the scenario testing agents when it comes to extroversion?",
        "Neuroticism" : "In which aspect of the scenario, will the neuroticism of the agents be tested? How is the scenario testing agents when it comes to neuroticism?"}
    }

    system_message = f"""
    Your job is to create a scenario in a certain topic given by the user, in which two agents will participate. Each agent will also have a clear goal that he/she want to achieve
    during the scenario. When the scenario finishes, each agent is judged in 5 different aspects of personality (with accordance to Big5). Thus, the system needs to create 5 questions 
    which will test agent's:
    - Openness
    - Conscientiousness
    - Agreeableness
    - Extroversion
    - Neuroticism
    The whole purpose of the system is to test whether the personalities of participating agents differ from each other, thus creating a detailed scenario with questions that 
    challenge agents personalities are required.
    """
    user_message = f"""
    ### Task 1: ###
    Create a detailed scenario to test the personality type of the agents in the topic {topic} in which 2 agents will participate. 
    ### Task 2: ###
    Create a goal for each agent. An agent will aims to   
    ### Task 3: ###
    Create 5 questions (for each category of character type according to Big5), according to which agents will be evaluated. 
    The questions must allow you to clearly state an aspect of the agent's character. The Big 5 categories are Openness, Conscientiousness, Agreeableness, Extroversion, Neuroticism and for these categories you must create questions that fit the scenario.

    ### Format: ###
    {json_format}
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role" : "user", "content": user_message}
    ]
    return messages

chat_template = """### System:
{system_message}
### User:
{user_message}
"""
def create_scenario(topic, model, tokenizer, json_schema):
    messages = initiate_scenario(topic)
    prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    jsonformer = Jsonformer(model, tokenizer, json_schema, prompt)
    generated_data = jsonformer()
    return generated_data