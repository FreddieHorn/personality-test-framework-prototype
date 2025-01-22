from jsonformer import Jsonformer

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
    You are an expert in behavioral psychology and personality analysis. You can evaluate the interaction between two agents based on 7 distinct dimensions, providing a score for each dimension within the range [lower bound–upper bound] which specified for each dimension in the description bellow.
    Below is a detailed explanation of each dimension:

    Goal Completion (GOAL) [0–10] is the extent to which the agent achieved their goals. Agents’ social goals, defined by the environment, are the primary drivers of their behavior.
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


def scenario_creation_prompt(setting, topic, model, tokenizer):    # Define the JSON structure for the result
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
    Create a detailed scenario to evaluate how personality traits affect two agents’ success in achieving a shared goal. Use the following:

    Setting: {setting}
    Topic: {topic}

    ### Task 2: ###
    Clearly define the shared goal and personal goals that both agents aim to achieve in the scenario. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
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

    The characters should remain true to their personalities and use verbal communication and actions to accomplish their shared goal. The dialogue and actions should naturally reflect how their distinct personalities influence their strategies and approaches.

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
