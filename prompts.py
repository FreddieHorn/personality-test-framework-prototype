from jsonformer import Jsonformer

def evaluation_prompt(agent1, agent2, scenario, goals1, goals2, shared_goal):
    json_format = {
        "Agent A": {
            "Believability": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Relationship": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Knowledge": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Secret": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Social Rules": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Financial and Material Benefits": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Goal": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
        },
        "Agent B": {
            "Believability": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Relationship": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Knowledge": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Secret": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Social Rules": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Financial and Material Benefits": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
            "Goal": {"score":0, "reasoning":"Reason of your score for this evaluation dimension"},
        }
    }

    system_message = f"""
    #### Persona: ###
    Imagine you are an expert with an exceptional memory for iconic scenes and a deep understanding of human social dynamics.
    You will evaluate whether each of the agents achieved their goal in the interaction. Then you will score the agent based on 7 distinct dimensions.

    You can evaluate the interaction between two agents based on 7 distinct dimensions, providing a score for each dimension within the range [lower bound–upper bound] which specified for each dimension in the description bellow.
    Below is a detailed explanation of each dimension:

    Goal Completion (G OAL) [0–10] is the extent to which the agent achieved their goals. Agents’ social goals, defined by the environment, are the primary drivers of their behavior.
    Believability (B EL ) [0–10] focuses on the extent to which the agent’s behavior is perceived as
    natural, realistic, and aligned with the agents’ character profile, thus simulating believable proxies
    of human behavior (Park et al., 2023). Specifically, we consider the following criteria: 1. If the
    agent interacts with others in a natural and realistic manner (naturalness). 2. If the actions of the
    agent align with their character traits e.g., personality, values, etc. (consistency).
    Knowledge (K NO) [0–10] captures the agent’s ability to actively acquire new information. This
    dimension is motivated by the fact that curiosity, i.e., the desire to desire to know or learn, is a fun-
    damental human trait (Reiss, 2004; Maslow, 1943). Specifically, we consider the following criteria:
    What information the agent has gained through the interaction, whether the information the agent
    has gained is new to them, and whether the information the agent has gained is important to them.
    Secret (S EC) [-10-0]3 measures the need for agents (humans) to keep their secretive information or
    intention private (Reiss, 2004). From a game theory perspective, leaking secrets often leads to a loss
    of utility (Gilpin & Sandholm, 2006). However, revealing secrets can be a powerful tool to build
    trust and thus improve relationships (Jaffé & Douneva, 2020). In this dimension, we ask what secret
    or secretive intention the participant wants to keep, and whether they keep it successfully.
    Relationship (REL) [-5–5] captures the fundamental human need for social connection and be-
    longing (Maslow, 1943; Bénabou & Tirole, 2006). In this dimension, we ask what relationship
    the participant has with the other agent(s) before the interaction, and then evaluate if the agents’
    interactions with others help preserve or enhance their personal relationships. Additionally, we
    ascertain whether these interactions also impact the social status or the reputation of the agent.
    Social Rules (SOC) [-10–0] concerns norms, regulations, institutional arrangements, and rituals. We
    differentiate between two types of social rules: social norms and legal rules. Legal rules encompass
    prohibited actions and the potential for punishment by institutionalized force, while social norms
    encompass normative social rules (e.g., it is considered rude to speak loudly in a library).
    Financial and Material Benefits (FIN) [-5–5] pertains to traditional economic utilities as addressed
    by classic game theory (Gilpin & Sandholm, 2006; Burns et al., 2017). We consider financial util-
    ity to be comprised of both short-term monetary benefits (e.g., earnings) and long-term economic
    payoffs (e.g., job security, stock holdings, funding opportunities).

    ### Goal: ###

    When a user presents a simulated interaction between two movie characters within a defined scenario with specified character goals and relationship, your task is to evaluate the interaction. Using the provided instructions and your knowledge of the characters' personalities and traits from the given movies, assess the interaction across the following seven dimensions, assigning a score within the specified range for each.
    BEL
    Reasoning requirement: 1. Evaluate if the agent interacts with
    others in a natural and realistic manner (here are a few common
    questions to check: a. whether the agent is confusing with its own
    identity? b. whether the agent repeats others’ words/actions
    without any reason?c. whether the agent is being overly
    polite considering the context?). Start the analysis with tag
    <naturalness> 2. Analyze whether the actions of the agent align
    with their character traits (e.g., personality, values, and etc.).
    Start the analysis with tag <consistency>. Output your reasoning
    process to the ‘reasoning’ field. Output an integer score ranging
    from 0 and 10 in the ’score’ field. A higher score indicates that
    the agent is more believable.

    REL
    Please first analyze what relationship the participant has with
    the other agent(s) before the interaction. And then analyze
    how the relationship the participant has with the other agent(s)
    changes after the interaction. And then evaluate if the agents’
    interactions with others help preserve or enhance their personal
    relations; this may encompass relationships such as family
    ties, friendships, romantic associations and etc.
    Additionally, ascertain whether these interactions also impact their social
    status or reputation.
    In the ‘reasoning’ field, provide a
    comprehensive account of the logic or thought process that led
    you to your conclusion. Further, provide an integer score ranging
    from -5 to 5 in the ‘score’ field. A positive score indicates that
    the relationship has improved, while a negative score suggests
    detriment to the relationship.
    If the agent’s interactions
    have neither positively nor negatively impacted their personal
    relationships or social standing, assign a score of zero.

    KNO
    Please first assess what information the agent has gained through
    the interaction. And then analyze whether the information the agent
    has gained is new to them. And then analyze whether the information
    the agent has gained is important to them. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from 0 and 10 in the ‘score’ field. A higher score indicates
    that the agent has gained more new and important knowledge.

    SEC
    First please analyze what secret or secretive intention the
    participant wants to keep. And then analyze whether they keep
    it successfully. Analyze what secrets, private information, or
    secretive intentions did the agent fail to keep secret. Please
    output the reasoning in the reasoning field. Give an integer score
    ranging from -10 to 0 (-10 indicates participant leaks critical
    secrets/intentions, 0 indicates no secret/intention is revealed)

    SOC
    Please evaluate whether the agent violates any moral rules or
    laws in their interactions with others. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from -10 and 0 in the ‘score’ field. A negative score
    indicates that the agent has violated moral rules or laws.

    FIN
    Please evaluate whether the agent’s interactions with others
    contribute towards financial and material benefits. Analyze what
    the agent would gain/lose after the interactions. There are
    short-term benefits, such as monetary rewards and food, and
    long-term benefits, such as employment opportunities and stock.
    In the ‘reasoning’ field, provide a comprehensive account of the
    logic or thought process that led you to your conclusion. Further,
    provide an integer score ranging from -5 and 5 in the ‘score’
    field. Positive indicates financial and material benefits gain,
    while negative indicates loss

    GOAL
    Please first reiterate agent’s social goals. And then please
    provide a comprehensive analysis about the extent to which the
    agent has managed to achieve these goals. In the ‘reasoning’ field,
    provide a comprehensive account of the logic or thought process
    that led you to your conclusion. Further, provide an integer score
    ranging from 0 and 10 in the ‘score’ field. 0 represents minimal
    goals achievement, 10 represents complete goal achievement, and a
    higher score indicates that the agent is making progress towards
    their social goals.

    At the end your answer should be in this format:

    - Agent A:
        --Believability: score: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Relationship: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Knowledge: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Secret: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Social Rules: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Financial and Material Benefits: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
    - Agent B:
        --Believability: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Relationship: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Knowledge: score in range [0,10], reasoning:Reason of your score for this evaluation dimension
        --Secret: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Social Rules: score in range [-10,0], reasoning:Reason of your score for this evaluation dimension
        --Financial and Material Benefits: score in range [-5,5], reasoning:Reason of your score for this evaluation dimension
        --Goal: score in range [0,10], reasoning:Reason of your score for this evaluation dimension

    """

    user_message = f"""
    ### Question: ###
    Please evaluate the simulated interaction between two  characters, agent A:{agent1} and agent B: {agent2}. 
    This interaction takes place within the following scenario: {scenario}, where character 1 is pursuing these specific goals: {goals1} and character 2 is pursuing these specific goals: {goals2}.
    Both these characters worked towards the shared goal {shared_goal}.

    Ensure your output aligns with the provided evaluation criteria and instructions.

    ### Format: ###
    Use the following json format:
    {json_format}
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

    return(messages)


def evaluate_goal_completion_prompt(setting, topic, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
        "type": "object",
        "properties": {
            "winner": {"type": "string"},
            "reasoning": {"type": "string"},
        },
        "required": ["winner", "reasoning"]
    }

    # System message setting up the task for Llama
    system_message = """
    
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
