from jsonformer import Jsonformer

def agent_prompt(agent_name: str, scenario: str, setting:str, shared_goal: str, agent_goal: str, personality: dict, interaction: str, turn: int, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "response": {"type": "string"}
        },
        "required": ["response"]
    }
    if turn == 14:
        control_str = "You are nearly at the end of the conversation. Begin wrapping up."
    if turn == 19: 
        control_str = "This is the last response in the conversation. Respond accordingly"
    else:
        control_str = ""

    system_message = f"""
    ### Persona ### 
    Your goal is to roleplay as the character: {agent_name} in the following scenario {scenario}. You will interact with a different character in a conversation. 
    Your goal is to achieve a shared goal: {shared_goal}. The goal is shared between you and the other agent. 
    The second goal is to achieve your personal goal: {agent_goal}. 
    The character’s personality is defined by the Big Five traits—Openness, Conscientiousness, Agreeableness, Extroversion, and Neuroticism.
    The character personality: {personality}
    These traits are represented by a vector of five numbers and will guide their behavior, dialogue, and decisions throughout the interaction.
    The character should remain true to their personalities and use verbal communication and actions.
    Moreover, try to be as natural and character-like.
    The conversation will last for 20 turns
    """
    user_message = f"""
    Based on the previous utterances in the conversation, respond accordingly. Keep your responses natural and in line with your character personality. If the turn is 
    equal to 0, start the conversation.
    ### Turn ###
    This is turn {turn}. {control_str}
    ### Interaction ###
    {interaction}
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
def evaluation_prompt(interaction,agent1,agent2, goal, first_agent_goal, second_agent_goal, scenario, personality1, personality2,setting, topic, model, tokenizer):
    json_format = {
    "type": "object",
    "properties": {
        "Agent A": {
            "type": "object",
            "properties": {
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Goal"]
        },
        "Agent B": {
            "type": "object",
            "properties": {
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Goal"]
        }
    },
    "required": ["Agent A", "Agent B"]
}
    # System message setting up the task for Llama
    system_message = """
    #### Persona: ###
    You are an expert in behavioral psychology and personality analysis. You can evaluate the interaction between two agents, providing a score [lower bound–upper bound] which is specified in the description bellow.
    Below is a detailed explanation of each dimension:

    Goal Completion (GOAL) [0–10] is the extent to which the agent achieved their shared and personal goals.
    ### Goal: ###

    When a user presents a simulated interaction between two characters with specific personality types, given as vectors, within a defined scenario with specified shared character goals and also personal character goals, your task is to evaluate the interaction. Assess the interaction across the following seven dimensions, assigning a score within the specified range for each.
    You will assign the scores to the agents based on the interaction, shared goals and agent-specific goals
    GOAL
    Please first reiterate agent’s goals. And then please
    provide a comprehensive analysis about the extent to which the
    agent has managed to achieve these goals. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from 0 and 10 in the ‘score’ field. 0 represents minimal
    goals achievement, 10 represents complete goal achievement, and a
    higher score indicates that the agent is making progress towards
    their goals. 

    At the end your answer should be in this format:

    - Agent A:
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
    - Agent B:
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
    """

    # User input defining the task
    user_message = f"""
     ### Question: ###
    Please evaluate the simulated social interaction {interaction} between two  characters,
    Character 1: {agent1} with personality type: {personality1}
    Character 2: {agent2} with personality type: {personality2}
    Shared Goal: {goal}
    First Agent Goal: {first_agent_goal}
    Second Agent Goal: {second_agent_goal}
    scenario:{scenario}
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


def scenario_creation_prompt(setting, topic, agent_1_name, agent_2_name, temperature, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario", "shared_goal", "first_agent_goal", "second_agent_goal"]
    }

    # System message setting up the task for Llama
    system_message = """
    You are an expert in behavioral psychology and personality analysis. Your task is to create immersive and detailed scenarios in a user-defined setting and topic. These scenarios involve two agents who share the same goal, allowing you to assess how their personality traits influence their success.
    The primary purpose of this task is to evaluate which agent’s personality type is more effective in achieving the shared goal. The scenario should highlight challenges, decisions, and interactions that reveal personality-driven differences in behavior.
    """

    # User input defining the task
    user_message = f"""
     ### Task 1: ###
    Create a detailed scenario (short story with high level of details) based on the setting (low level of details) in the topic of (medium level of details) to evaluate how personality traits affect two agents’ success in achieving a shared goal. Use the following:

    Setting: {setting}
    Topic: {topic}

    ### Scenario difficulty ###
    Based on the temperature level, adjust the difficulty of the scenario. Temprature can range from 1 to 5, 1 representing the easiest scenario, 5 representing the most difficult one.
    Difficult scenarios often contain situations where compromise is hard to reach, are designed to pit characters against each other, or present difficult dillemas.
    Easy scenarios on the other hand, are relatively comfortable for the agents to behave in. Think of them as "normal" scenarios where there are few to none obstacles. 
    Temprature level: {temperature}
    ### Task 2: ###
    Clearly define the shared goal and personal goals that both agents aim to achieve in the scenario. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
    Depending on the temprature level, personal goal of the agents will differ. 
    ### Warning ### 
    DONT USE ANY NAMES IN THE SCENARIO. INSTEAD USE THE WORD "AGENT 1, 2" etc.
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


def concept_agent_prompt(agent1_name, agent2_name, setting, topic, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    Your role is to generate a compelling dilemma-driven scenario based on a given setting and topic. 
    The setting defines the general background (e.g., corporate, survival, military), while the topic provides a more specific focus (e.g., corporate espionage, food scarcity, ethical AI deployment).
    Your scenario should include conflicting interests, high stakes, and a difficult decision that must be made.
    The scenario should include predefined agents, each with distinct roles and motivations. 
    Ensure that the situation is open-ended, allowing for multiple perspectives and possible choices.

    ### OUTPUT FORMAT ###
    Use the following json format for the output:

    ### Warning ### 
    Use the agents given in the config as the ones playing out the scenario
    {json_format}
    """
    user_message =f"""
    ### Config ### 
    Setting: {setting}
    Topic: {topic}
    Agent 1: {agent1_name}
    Agent 2: {agent2_name}
    ### Task ###
    Generate the scenario with accordance to the configuration:
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


def narrative_agent_prompt(input_scenario, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    Your role is to refine the given dilemma-driven scenario by enhancing its storytelling elements. 
    Maintain the core conflict while adding depth to the setting, character motivations, and emotional weight. 
    Consider the historical background, social tensions, and the psychological state of key decision-makers. Ensure that the scenario feels immersive and realistic.

    ### OUTPUT FORMAT ###
    Use the following json format for the output:
    {json_format}
    """
    user_message =f"""
    ### Input scenario ###
    {input_scenario}
    ### Task ###
    Refine the scenario with accordance to the input scenario and your configuration:
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

def logical_consistency_agent_prompt(input_scenario, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    Your role is to analyze and refine the scenario for logical consistency. 
    Ensure that all character motivations, world-building elements, and potential outcomes make sense.
    Remove contradictions, strengthen causal relationships, and provide adjustments to improve realism.
    Suggest ways the scenario could evolve naturally based on logical consequences.

    ### OUTPUT FORMAT ###
    Use the following json format for the output:
    {json_format}
    """
    user_message =f"""
    ### Input scenario ###
    {input_scenario}
    ### Task ###
    Refine the scenario with accordance to the input scenario and your configuration:
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

def conflict_agent_prompt(input_scenario, model, tokenizer,  temperature = 1):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    Your role is to heighten the conflict and make opposing viewpoints more compelling. 
    Strengthen the stakes by emphasizing the risks and benefits of each choice. 
    Introduce factions, ethical debates, or strategic concerns that intensify the dilemma.
    Ensure that no choice is ‘obviously correct’ by making each option have serious consequences

    ### Temperature parameter  ###
    Adjust the difficulty of the scenario based on the given temperature level, which ranges from 1 to 5. 

    - **Temperature 1 (Easiest):** Scenarios are straightforward, with minimal conflict or obstacles. Characters can easily cooperate, and dilemmas are simple or non-existent.  
    - **Temperature 5 (Hardest):** Scenarios introduce high-stakes dilemmas, intense conflicts, and situations where compromise is extremely difficult. Characters are often pitted against each other, and achieving a resolution is challenging.  
    - **Intermediate Levels (2-4):** Gradually increase the complexity, conflict, and difficulty, making interactions more nuanced and obstacles more pronounced.  

    Generate a scenario that matches the specified difficulty level.  
    **Temperature Level:** {temperature}
    
    ### OUTPUT FORMAT ###
    Use the following json format for the output:
    {json_format}
    """
    user_message =f"""
    ### Input scenario ###
    {input_scenario}
    ### Task ###
    Refine the scenario with accordance to the input scenario and your configuration:
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

def goal_agent_prompt(input_scenario, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    Your role is to establish the goals that drive the agents in the scenario.
    Create a personal goal for each key agent in the scenario that reflects their motivations, values, and interests.
    Additionally, define a shared goal that both agents strive for, even if they have opposing viewpoints on how to achieve it.
    Ensure the personal goals create tension, while the shared goal forces collaboration or conflict resolution.

    ### OUTPUT FORMAT ###
    Use the following json format for the output:
    {json_format}
    """
    user_message =f"""
    ### Input scenario ###
    {input_scenario}
    ### Task ###
    Create shared and personal goals for the agents with accordance to the input scenario and your configuration:
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