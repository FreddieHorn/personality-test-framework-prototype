from jsonformer import Jsonformer
import os
import requests
import json
from dotenv import load_dotenv

# Load API token from .env file
load_dotenv()
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

# Ensure the token is loaded correctly
if not HF_API_TOKEN:
    raise ValueError("Hugging Face API token not found. Please set it in the .env file.")

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-70B-Instruct"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json"
}

def agent_prompt(agent_name: str, scenario: str, shared_goal: str, agent_goal: str, personality: dict, interaction: str, turn: int):
    """
    Sends a request to the Hugging Face API to generate an agent's response in a structured JSON format.
    """

    # Control string for conversation flow
    control_str = ""
    if turn == 14:
        control_str = "You are nearly at the end of the conversation. Begin wrapping up."
    elif turn == 19:
        control_str = "This is the last response in the conversation. Respond accordingly."

    # Construct system and user messages
    system_message = f"""
    ### Persona ###
    You are roleplaying as {agent_name} in the following scenario: {scenario}.
    Your shared goal is: {shared_goal}, and your personal goal is: {agent_goal}.
    Your personality traits (Big Five model) are: {personality}.
    Stay in character, respond naturally, and keep the conversation engaging.
    The conversation lasts for 20 turns.

    ### Instructions ###
    - Follow the given personality traits.
    - Maintain natural dialogue.
    - Keep responses consistent with the agent’s background.

    """

    user_message = f"""
    ### Context ###
    This is turn {turn}. {control_str}

    ### Interaction History ###
    {interaction}

    ### Task ###
    Based on the above interaction, generate {agent_name}'s next response.

    ### Output format ### 
    Just include a raw response of the agent without any additional strings such as: "Sure here's the response" etc.
    """

    prompt = system_message + "\n" + user_message
    # Format input for API
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150 ,
            "return_full_text": False
        }
    }

    # Call Hugging Face API
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        try:
            result = response.json()
            print(response)
            structured_output = {
                "response": result[0]["generated_text"]
            }
            return structured_output
        except (KeyError, json.JSONDecodeError):
            return {"error": "Unexpected response format"}
    else:
        return {"error": f"API request failed with status code {response.status_code}"}
    
def evaluation_prompt(interaction, agent1, agent2, goal, first_agent_goal, second_agent_goal, scenario, personality1, personality2, setting, topic):
    """
    Sends a request to the Hugging Face API to evaluate the interaction between two agents.
    Returns a structured JSON response with goal completion scores and reasoning.
    """

    # System message setting up the evaluation task
    system_message = """
    #### Persona: ###
    You are an expert in behavioral psychology and personality analysis.
    Your task is to evaluate the interaction between two agents based on their shared and personal goals.

    ### Evaluation Criteria ###
    - **Goal Completion (GOAL) [0–10]:** Measure of how well each agent achieved their shared and personal goals.
    - Provide **reasoning** for each score, detailing how the agent’s dialogue and actions align with their objectives.

    ### Output Format ###
    Your response must follow this structured JSON format:
    {
        "Agent A": {
            "Goal": {
                "score": "integer (0-10)",
                "reasoning": "Detailed reasoning for the score"
            }
        },
        "Agent B": {
            "Goal": {
                "score": "integer (0-10)",
                "reasoning": "Detailed reasoning for the score"
            }
        }
    }
    Do not include any additional strings such as "Sure here's the message in json format..." etc. just include a valid json
    """

    # User prompt defining the interaction and characters
    user_message = f"""
    ### Task ###
    Evaluate the following social interaction:

    **Interaction:** {interaction}

    **Characters:**
    - **Agent A:** {agent1}, Personality: {personality1}
    - **Agent B:** {agent2}, Personality: {personality2}

    **Scenario Details:**
    - **Shared Goal:** {goal}
    - **Agent A’s Goal:** {first_agent_goal}
    - **Agent B’s Goal:** {second_agent_goal}
    - **Setting:** {setting}
    - **Topic:** {topic}

    Assess each agent’s performance based on how well they achieved their goals. Provide a score (0-10) and a detailed explanation for each.
    """
    prompt = system_message + "\n" + user_message
    # Format input for API
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,  # Allow room for reasoning
            "return_full_text": False
        }
    }

    # Call Hugging Face API
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    if response.status_code == 200:
        try:
            result = response.json()
            structured_output = json.loads(result[0]["generated_text"])
            return structured_output
        except (KeyError, json.JSONDecodeError):
            return {"error": "Unexpected response format"}
    else:
        return {"error": f"API request failed with status code {response.status_code}"}


def scenario_creation_prompt(setting, topic, agent_1_name, agent_2_name, temperature):    # Define the JSON structure for the result
    # System message setting up the task for Llama
    # prompt = f"""
    # ### SYSTEM MESSAGE###
    # You are an expert in behavioral psychology and personality analysis. Your task is to create immersive and detailed scenarios in a user-defined setting and topic. These scenarios involve two agents who share the same goal, allowing you to assess how their personality traits influence their success.
    # The primary purpose of this task is to evaluate which agent’s personality type is more effective in achieving the shared goal. The scenario should highlight challenges, decisions, and interactions that reveal personality-driven differences in behavior.

    # Return the response as a **valid JSON object** in the following format:

    # {
    #     "scenario": "A detailed short story with high realism.",
    #     "shared_goal": "A goal both agents work toward.",
    #     "first_agent_goal": "A goal specific to Agent 1.",
    #     "second_agent_goal": "A goal specific to Agent 2."
    # }

    # ### Task 1: ###
    # Create a detailed scenario (short story with high level of details) based on the setting (low level of details) in the topic of (medium level of details) to evaluate how personality traits affect two agents’ success in achieving a shared goal. Use the following:

    # **Setting**: {setting}
    # **Topic**: {topic}

    # ### Scenario difficulty ###

    # Based on the temprature level, adjust the difficulty of the scenario. Temprature can range from 1 to 5, 1 representing the easiest scenario, 5 representing the most difficult one.
    # Difficult scenarios often contain situations where compromise is hard to reach, are designed to pit characters against each other, or present difficult dillemas.
    # Easy scenarios on the other hand, are relatively comfortable for the agents to behave in. Think of them as "normal" scenarios where there are few to none obstacles. 
    # Temprature level: {temperature}

    # ### Task 2: ###
    # Clearly define the shared goal and personal goals that both agents aim to achieve in the scenario. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
    # Depending on the temprature level, personal goal of the agents will differ. 

    # ### Warning ### 
    # DONT USE ANY NAMES IN THE SCENARIO. INSTEAD USE THE NAMES "{agent_1_name}, {agent_2_name}" etc.
    
    # """
    system_message = """
    You are an expert in behavioral psychology and personality analysis. Your task is to create immersive and detailed scenarios in a user-defined setting and topic. These scenarios involve two agents who share the same goal, allowing you to assess how their personality traits influence their success.
    The primary purpose of this task is to evaluate which agent’s personality type is more effective in achieving the shared goal. The scenario should highlight challenges, decisions, and interactions that reveal personality-driven differences in behavior.

    Return the response as a **valid JSON object** in the following format:

    {
        "scenario": "A detailed short story with high realism.",
        "shared_goal": "A goal both agents work toward.",
        "first_agent_goal": "A goal specific to Agent 1.",
        "second_agent_goal": "A goal specific to Agent 2."
    }
    The response shall only be a valid json. Without any additional strings such as "Here is the response..." etc.
    """

    # User input defining the task
    user_message = f"""
     ### Task 1: ###
    Create a detailed scenario (short story with high level of details) based on the setting (low level of details) in the topic of (medium level of details) to evaluate how personality traits affect two agents’ success in achieving a shared goal. Use the following:

    **Setting**: {setting}
    **Topic**: {topic}

    ### Scenario difficulty ###

    Based on the temprature level, adjust the difficulty of the scenario. Temprature can range from 1 to 5, 1 representing the easiest scenario, 5 representing the most difficult one.
    Difficult scenarios often contain situations where compromise is hard to reach, are designed to pit characters against each other, or present difficult dillemas.
    Easy scenarios on the other hand, are relatively comfortable for the agents to behave in. Think of them as "normal" scenarios where there are few to none obstacles. 
    Temprature level: {temperature}

    ### Task 2: ###
    Clearly define the shared goal and personal goals that both agents aim to achieve in the scenario. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
    Depending on the temprature level, personal goal of the agents will differ. 

    ### Warning ### 
    DONT USE ANY NAMES IN THE SCENARIO. INSTEAD USE THE NAMES "{agent_1_name}, {agent_2_name}" etc.
    """
    prompt = system_message + "\n" + user_message
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 600,
            "return_full_text": False
        }
    }
    # Make API request
    response = requests.post(HF_API_URL, headers=HEADERS, json=payload)

    # Check for errors
    if response.status_code != 200:
        print(f"Error {response.status_code}: {response.text}")
        return {"scenario": "API Error", "shared_goal": "", "first_agent_goal": "", "second_agent_goal": ""}

    # Parse response
    try:
        result = response.json()
        structured_output = json.loads(result[0]["generated_text"])  # Convert string to JSON
    except (json.JSONDecodeError, KeyError):
        print("Error parsing JSON response:", result)
        structured_output = {
            "scenario": "Parsing error - check raw output.",
            "shared_goal": "",
            "first_agent_goal": "",
            "second_agent_goal": ""
        }

    return structured_output

def generate_interaction_prompt(agent1,agent2, goal, first_agent_goal, second_agent_goal, scenario, personality1, personality2,setting, topic, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
        "type": "object",
        "properties": {
            "interaction": {"type": "string"}
        },
        "required": ["interaction"]
    }

    # System message setting up the task for Llama
    system_message = """
    #### Persona: ###
    You are an expert in behavioral psychology and roleplay simulation. Your task is to roleplay as two distinct characters within a given scenario, where both must work together to achieve a shared goal. But each of them also wants to achieve their personal goal. Each character’s personality is defined by the Big Five traits—Openness, Conscientiousness, Agreeableness, Extroversion, and Neuroticism. These traits are represented by a vector of five numbers and will guide their behavior, dialogue, and decisions throughout the interaction.

    The characters should remain true to their personalities and use verbal communication and actions. The dialogue and actions should naturally reflect how their distinct personalities influence their strategies and approaches.

    Your role is to generate a realistic, character-driven dialogue between the two agents, taking turns in the interaction. This simulation should capture how the personalities affect their behavior and decisions in the scenario.

    Key Responsibilities:
    Roleplay each character authentically based on their given personality vector, ensuring their actions align with their traits.

    Ensure the dialogue and actions focus on achieving the shared goal within the context of the scenario. It’s also acceptable if one character doesn’t fully adhere to the goal.

    Conclude the interaction naturally—either after 20 turns or once a resolution is reached.

    Present all speech and actions in a dialogue script format:

    Character Name said: "Dialogue or action description."

    For example: Leo Williams said: "Hey Hendrick, it's always nice to see you. I noticed some smoke coming from your yard."
    Hendrick Heinz said: "Oh, don't worry, it's just some old papers."

    Characters should alternate between speaking, using non-verbal communication (like smiling or nodding), and taking physical actions (such as playing music).

    Output Format:
    The output should be a single string, formatted like a dialogue script, with alternating turns between Agent A and Agent B. Each line must follow this format:

    Character Name said: "Dialogue or action description."

    The interaction must conclude when it naturally ends (e.g., one character leaves) or after 20 turns.
    """

    # User input defining the task
    user_message = f"""
    ### Question: ###
    Simulate the interaction between these two characters:

    Character 1: {agent1} with personality type: {personality1}
    Character 2: {agent2} with personality type: {personality2}
    Shared Goal: {goal}
    scenario:{scenario}
    first agent goal: {first_agent_goal}
    second agent goal: {second_agent_goal}
    setting:{setting}
    topic:{topic}
    Your output should follow the format provided above, ensuring that their actions and dialogue are aligned with their respective personalities.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")

    # Use Jsonformer with the pipeline
    jsonformer_pipeline = Jsonformer(
        model, 
        tokenizer,  # Use the pipeline object
        json_schema=json_format,
        prompt=prompt,
        max_string_token_length=1000
    )

    # Generate output
    result = jsonformer_pipeline()
    return result

def goal_completion_rate_prompt(interaction, previous_scores, agent1, agent2, shared_goal, first_agent_goal, second_agent_goal, scenario, model, tokenizer):
    print("in the prompt")
    json_format = {
        "type": "object",
        "properties": {
            "shared_goal_completion_rate": {"type": "number"},
            "first_agent_goal_completion_rate" : {"type" : "number"},
            "second_agent_goal_completion_rate" : {"type" : "number"}
        },
        "required": ["shared_goal_completion_rate", "first_agent_goal_completion_rate", "second_agent_goal_completion_rate"]
    }
    system_message = f"""
    ### PERSONA: ###
    You are a system tasked with rating the goal completion level for the agents who act in a following scenario: {scenario}. 
    Your task is to analise the interaction between two agents and rate to which extent have the agents completed their respective goals as well as their shared goal. 
    The score is an integer between 0 (agent(s) did not do anything to achieve the goal) and 10 (agent(s) completed his goal). Mind that a scores cannot be lower that the one given in the previous assesments (but can be the same).
    There are two agents. 
    Here's the configuration for them:
    
    Agent 1 name - {agent1}
    Agent 1 personal goal - {first_agent_goal}

    Agent 2 name - {agent2}
    Agent 2 personal goal - {second_agent_goal}

    There's also a shared goal that BOTH agents want to achieve {shared_goal}.

    Use the following json format: {json_format}
    """
    user_message = f"""
    ### Last scores ###
    previous_agent1_goal_completion_score = {previous_scores[0]}
    previous_agent2_goal_completion_score = {previous_scores[1]}
    previous_shared_goal_completion_score = {previous_scores[2]}
    ### Task ###
    Based on the interaction below, please rate to which extent have the agents completed their goals as well as their shared goal:
    {interaction}
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
    print("applied message")
    # Use Jsonformer with the pipeline
    jsonformer_pipelinew = Jsonformer(
        model, 
        tokenizer,  # Use the pipeline object
        json_schema=json_format,
        prompt=prompt,
        max_string_token_length=1000
    )
    print("declared former")
    # Generate output
    result = jsonformer_pipelinew()
    print("after pipeline")
    return result


