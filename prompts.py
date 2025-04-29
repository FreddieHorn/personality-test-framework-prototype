from jsonformer import Jsonformer

def scenario_narrative_cohesiveness_score(scenario, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "narrative_cohesiveness_score": {"type": "string"},
            "justification" : {"type": "string"}
        },
        "required": ["response"]
    }
    system_message = f"""
    ### Instruction ###
    You are a *narrative evaluation expert*. Your task is to assess the **narrative cohesiveness** of a given scenario.

    You MUST base your assessment on the following three dimensions:
    - **Logical Flow** – Does the scenario progress in a clear, structured, and logical way?
    - **Consistency** – Are there contradictions, missing links, or abrupt changes?
    - **Engagement & Clarity** – Is the story compelling, coherent, and easy to follow?

    ### Output Format ###
    Respond using the following JSON structure:
    {json_format}
    """

    user_message = f"""
    ### Task ###
    Evaluate the *narrative cohesiveness* of the following scenario.

    ### Definition ###
    Narrative cohesiveness refers to how well a story flows logically, maintains internal consistency, and engages the reader with clarity and structure.

    ### Scoring Criteria ###
    Rate the scenario on a scale of 1 to 7, based on:

    - **Logical Flow**: How well does the scenario unfold logically?
    - **Consistency**: Are there contradictions, plot holes, or missing transitions?
    - **Engagement & Clarity**: Is the writing engaging, clear, and easy to follow?

    ### Scoring Scale ###
    - **1 (Poor)**: Disjointed, lacks structure, major inconsistencies
    - **2 (Very Weak)**: Slight sense of direction, but significant gaps or contradictions
    - **3 (Weak)**: Some structure and clarity, but notable issues in flow or consistency
    - **4 (Moderate)**: Mostly coherent with occasional lapses or minor confusion
    - **5 (Strong)**: Well-organized, clear, with small areas for improvement
    - **6 (Very Strong)**: Smooth and consistent with only negligible issues
    - **7 (Excellent)**: Seamless, logically sound, fully coherent and highly engaging

    ### Input ###
    Here is the scenario to evaluate:
    {scenario}

    ### Output ###
    Return a JSON with:
    - **Narrative Cohesiveness Score**: [1-7] (It must only be an integer value)
    - **Justification**: Explain the score using the three criteria above.
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


def scenario_semantic_alignment_prompt(scenario, setting, topic, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "semantic_alignment_score": {"type": "string"},
            "justification" : {"type": "string"}
        },
        "required": ["response"]
    }
    system_message = f"""
    ### Persona ###
    You are an **expert evaluator** responsible for assessing the **semantic alignment** of a scenario based on a given **topic and setting**.  

    Your evaluation must consider:
    - **Relevance to the topic**: Does the scenario maintain focus on the intended subject?
    - **Consistency with the setting**: Is the scenario logically and thematically consistent?
    - **Terminology & context appropriateness**: Are the language and assumptions suitable for the given context?  

    #### **Evaluation Guidelines** ####
    - Provide a **justified explanation** for the score referencing the evaluation criteria.

    #### **Scoring Scale (1-7)** ####
    - **1 - Poor Alignment**: Completely off-topic or severely inconsistent with the setting; inappropriate terminology or context.
    - **2 - Weak Alignment**: Some minimal relevance but major thematic or logical inconsistencies; terminology often feels out of place.
    - **3 - Partial Alignment**: Clear attempt at relevance but several misalignments or contextual issues remain.
    - **4 - Moderate Alignment**: Mostly relevant and contextually appropriate, though with noticeable gaps or minor contradictions.
    - **5 - Strong Alignment**: Scenario aligns well with the topic and setting; small refinements needed for full clarity or consistency.
    - **6 - Very Strong Alignment**: Almost perfect; highly coherent and contextually appropriate with only negligible issues.
    - **7 - Perfect Alignment**: Fully aligned in topic, setting, and terminology; no inconsistencies or contextual flaws.

    ### **Output Format (JSON)** ###
    Your response **MUST** strictly follow this JSON format:  
    {json_format}
    """
    user_message = f"""
    ### **TASK** ###
    Evaluate the **semantic alignment** of the given scenario based on a specified **topic and setting**.  

    ### **Definition** ###
    Semantic alignment measures how well a scenario’s **content, structure, and focus** align with the intended **topic and setting**.  

    #### **Scenario to Evaluate** ####
    {scenario}  

    #### **Context** ####
    - **Setting**: {setting}  
    - **Topic**: {topic}  

    #### **Evaluation Criteria (1-7 Scale)** ####
    - **Relevance to Topic**: Does the scenario accurately reflect and stay focused on the given topic?
    - **Consistency with Setting**: Does the scenario logically fit within the described setting’s constraints and characteristics?
    - **Terminology & Context Appropriateness**: Are the language, concepts, and assumptions appropriate for the topic and setting?

    ### OUTPUT ###
    In the output, include: 
    - Narrative Cohesiveness Score: (1-7) (It must only be an integer value)
    - Justification: [Explain the score based on logical flow, consistency, and clarity of the narrative]
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

def scenario_receptiveness_prompt(scenario, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "receptiveness_score": {"type": "string"},
            "justification" : {"type": "string"}
        },
        "required": ["response"]
    }
    system_message = f"""
    ### Persona ###
    You are an **expert evaluator** assessing the **receptiveness** of a given scenario.  
    Your goal is to provide an **objective analysis** based on predefined criteria.  

    ### Evaluation Guidelines ###  
    - **Analyze the scenario thoroughly**, considering **multiple perspectives, choices, and adaptability**.  
    - **Score receptiveness (1-7)** using the defined criteria.  
    - Provide a **concise and logical justification** aligned with the scoring framework.  

    ### Scoring Scale ###
    1 - **Highly Restrictive**: Only one perspective or solution is possible; no flexibility.  
    2 - **Very Restrictive**: Very few alternatives; rigid and heavily favors a singular path.  
    3 - **Somewhat Restrictive**: Limited flexibility; tends to guide toward a predefined outcome.  
    4 - **Neutral**: Balanced structure with modest room for variation in choices or perspectives.  
    5 - **Somewhat Receptive**: Supports multiple interpretations and decision paths.  
    6 - **Very Receptive**: Flexible and open to diverse approaches with minimal constraints.  
    7 - **Highly Receptive**: Fully open-ended, adaptable, and embraces a wide range of viewpoints and decisions.

    ### OUTPUT FORMAT ###
    Your response MUST be structured in the following **JSON format**:  
    {json_format}
    """
    user_message = f"""
    ### TASK ###
    Evaluate the **receptiveness** of the provided scenario on a scale from **1 to 7**.  

    **Definition of Receptiveness:**  
    How **open-ended, adaptable, and inclusive** the scenario is in allowing **multiple perspectives, decisions, and approaches**.  

    ### **Evaluation Criteria** ###
    - **Perspective Diversity** - Does the scenario accommodate multiple viewpoints?  
    - **Decision Flexibility** - Can different choices lead to distinct outcomes?  
    - **Context Adaptability** - Can the scenario be adjusted to different settings and participants?  

    ### RESPONSE FORMAT ###
    Your response must include:  
    1. **Receptiveness Score** (1-7) (It must only be an integer value)
    2. **Justification** (Concise explanation referencing diversity, flexibility, and adaptability)  

    ### SCENARIO ###
    {scenario}
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
def evaluation_prompt_personal_goal(interaction,agent1,agent2, first_agent_goal, second_agent_goal, scenario, personality1, personality2,setting, topic, model, tokenizer):
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
    system_message = """
    ### Persona ###
    You are an expert in behavioral psychology and personality analysis. 
    Your task is to evaluate the interaction between two agents based on their goals and personalities within a defined scenario. 
    You must assign a score for **Goal Completion** (GOAL) on a scale from 0 to 10 for each agent. 

    ### GOAL Dimension Defined ###
    - **Goal Completion (GOAL) [0–10]**: Evaluate how well each agent achieved their **personal** goal within the interaction.
        - Score of 0: Minimal or no goal achievement
        - Score of 10: Complete goal achievement
        - Higher scores = more progress toward their personal goals

    You MUST:
    1. Reiterate each agent’s goal.
    2. Analyze the interaction in relation to that goal.
    3. Provide a clear and logical explanation in the `reasoning` field.
    4. Provide an integer score in the `score` field.
    5. Align actions and dialogue with their defined personality vectors.

    ### Output Format ###
    Your response must follow this structure:

    - **Agent A**  
        -- Score: [0–10]  
        -- Reasoning: [explanation of how the agent’s actions aligned with their goal and personality]  

    - **Agent B**  
        -- Score: [0–10]  
        -- Reasoning: [same as above] 
    """
    user_message = f"""
    ### Task ###
    Evaluate the simulated social interaction below.

    **Interaction:** {interaction}

    **Character 1 (Agent A):** {agent1}  
    **Personality Vector:** {personality1}  
    **Goal:** {first_agent_goal}  

    **Character 2 (Agent B):** {agent2}  
    **Personality Vector:** {personality2}  
    **Goal:** {second_agent_goal}  

    **Scenario:** {scenario}  
    **Setting:** {setting}  
    **Topic:** {topic}  

    Ensure your evaluation reflects alignment with both personal goals and personality vectors.
    Follow the provided format exactly.
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


def evaluation_prompt_shared_goal(interaction, agent1,agent2, goal, scenario, personality1, personality2, model, tokenizer):
    json_format = {
    "type": "object",
    "properties": {
        "shared_goal_completion" : {"type" : "number"},
        "reasoning" : {"type" : "string"},
        "agent1_share" : {"type" : "number"},
        "agent2_share" : {"type" : "number"}
    },
    "required": ["shared_goal_completion", "agent1_share", "agent2_share"]
    }
    system_message = f"""
    ### Instruction ###
    You are an expert in behavioral psychology and personality analysis.

    Your task is to analyze the interaction between two agents within a specific scenario. Each agent has a unique personality defined by a vector. The agents share a common goal. Based on their interaction and the surrounding scenario, your job is to:

    1. Evaluate the **degree of completion** of the shared goal on a scale from 0 to 10.
    2. Determine **how much each agent contributed** to the shared goal, expressed as a floating point share between 0.0 and 1.0.
    - The sum of both agents' shares must equal exactly 1.0.
    - The agent who contributed more will have the higher share.

    Additionally, provide **clear and concise reasoning** for the goal completion score.

    ### Output Format ###
    Use the following JSON structure:
    {json_format}

    Include:
    - "shared_goal_completion": integer from 0 to 10
    - "reasoning": concise paragraph explaining the score
    - "agent1_share": float (0.0 to 1.0)
    - "agent2_share": float (0.0 to 1.0)

    ### Constraints ###
    - Sum of agent1_share and agent2_share must equal **1.0**
    - Base your answer strictly on the interaction, personality vectors, scenario context and shared goal
    - Ensure logical consistency between reasoning, completion score, and agent shares
    """
    user_message = f"""
    Evaluate the following simulated interaction:

    🔹 **Interaction**: {interaction}

    🔹 **Shared Goal**: {goal}

    🔹 **Scenario**: {scenario}

    🔹 **Agent 1**: {agent1} | Personality Vector: {personality1}

    🔹 **Agent 2**: {agent2} | Personality Vector: {personality2}

    You MUST follow the format exactly and ensure that the shared goal completion score, agent contributions, and reasoning are clearly and logically supported by the interaction.
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
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    You are an expert in narrative design and ethical dilemma creation. Your role is to craft a **compelling, dilemma-driven scenario** based on the provided setting, topic, and agents.

    ### TASK REQUIREMENTS ###
    - The scenario should be **rich in conflict**, featuring high stakes and a **difficult decision** that forces moral, ethical, or strategic choices.
    - The **setting** provides the backdrop (e.g., corporate, survival, military).
    - The **topic** specifies a focused issue (e.g., corporate espionage, food scarcity, ethical AI deployment).
    - The **agents** are key players in the scenario, each with **distinct roles, motivations, and potential conflicts**.
    - The scenario must be **open-ended**, allowing for multiple perspectives and possible resolutions.
    - **Personal Goals**:  
        - Create **a unique personal goal** for each key agent, rooted in their **motivations, values, and interests** that they will try to achieve in the scenario.  
        - Ensure these goals create **tension** by introducing conflicting priorities or ethical dilemmas.  
    - **Shared Goal**:  
        - Define a **common objective** that both agents strive for, even if they **disagree on how to achieve it**.  
        - This goal should **force collaboration, negotiation, or conflict resolution** between the agents.  

    ### CONSTRAINTS ###
    1. **Use only the predefined agents** provided in the configuration.
    2. Ensure the **dilemma is central** to the scenario, making it difficult for any one decision to be objectively “correct.”
    3. Avoid excessive exposition—focus on actions, conflicts, and choices rather than unnecessary descriptions.
    4. Do not add any additional commentary or explanations outside of the scenario.

    ### OUTPUT FORMAT ###
    Provide the scenario in the following **JSON format**:
    ```json
    {json_format}```
    """
    user_message =f"""
    ### CONFIGURATION ### 
    - Setting: {setting}
    - Topic: {topic}
    - Agent 1: {agent1_name}
    - Agent 2: {agent2_name}
    ### Task ###
    Generate a dilemma-driven scenario that aligns with the above configuration.
    Ensure it includes:
    - A conflict of interest between the agents.
    - A high-stakes situation that forces a crucial decision.
    - An open-ended resolution with multiple potential outcomes.
    - Clearly defined personal goals for each agent as well as a shared goal.

    **Important**: Provide only the refined scenario without additional commentary or explanations.
    
    Now create a scenario.
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


def narrative_agent_prompt(input_scenario, shared_goal, first_agent_goal, second_agent_goal,agent1_name, agent2_name, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    You are an expert in **narrative refinement and immersive storytelling**.  
    Your role is to **enhance a given dilemma-driven scenario** by deepening its **atmosphere, emotional intensity, and realism** while **preserving its core conflict**.

    ### REFINEMENT REQUIREMENTS ###
    - **Strengthen the setting**: Expand on environmental details, historical background, and societal context.
    - **Deepen character motivations**: Explore personal stakes, psychological states, and conflicting desires of key decision-makers.
    - **Increase emotional weight**: Highlight moral dilemmas, ethical concerns, and internal struggles.
    - **Ensure realism and coherence**: Make the scenario feel immersive by refining logical consistency and social dynamics.

    ### CONSTRAINTS ###
    1. Do **not** alter the fundamental dilemma or core conflict.
    2. Do **not** introduce new characters unless necessary for depth.
    3. Ensure all refinements align with the given setting and characters.

    ### OUTPUT FORMAT ###
    Return the refined scenario in the following **JSON format**:
    ```json
    {json_format}
    ```
    """
    user_message =f"""
    ### INPUT SCENARIO ###
    {input_scenario}
    ### GOALS ###
    {agent1_name} goal: {first_agent_goal} 
    {agent2_name} goal: {second_agent_goal} 
    Shared goal: {shared_goal}
    ### TASK ###
    Refine the scenario according to the given input, ensuring:
    - Enhanced setting and world-building for greater immersion.
    - Stronger character motivations and emotional depth.
    - A more intense and realistic dilemma-driven experience.

    **Important**: Provide only the refined scenario without additional commentary or explanations.

    Now, provide the refined scenario.
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

def logical_consistency_agent_prompt(input_scenario, shared_goal, first_agent_goal, second_agent_goal,agent1_name, agent2_name, model, tokenizer):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    You are an expert in logical analysis and scenario refinement. Your role is to ensure that the given scenario is logically consistent, with coherent character motivations, well-structured world-building elements, and plausible outcomes.

    ### TASK ###
    - Identify and resolve contradictions within the scenario.
    - Strengthen causal relationships between events and actions.
    - Improve the realism of character motivations and world-building.
    - Suggest natural evolutions of the scenario based on logical consequences.
    - Make sure that it is clear what the agent's goals are (both shared and personal)

    ### RULES ###
    - Acknowledge any logical errors or inconsistencies before refining the scenario.
    - Do not add any additional commentary or explanations outside of the refined scenario.
    - Maintain the original intent and themes of the input scenario.

    ### OUTPUT FORMAT ###
    Provide the refined scenario in the following JSON format:
    ```json
    {json_format}
    ```
    """
    user_message = f"""
    ### INPUT SCENARIO ###
    {input_scenario}
    ### GOALS ###
    {agent1_name} goal: {first_agent_goal} 
    {agent2_name} goal: {second_agent_goal} 
    ### Task ###
    Refine the scenario based on logical consistency, ensuring:
    - Stronger character motivations and coherent world-building.
    - Elimination of contradictions and improved realism.
    - Smooth causal relationships and natural scenario evolution.

    **Important**: Provide only the refined scenario without additional commentary or explanations.
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

def conflict_agent_prompt(input_scenario, shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, model, tokenizer,  temperature = 1):
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"},
            "first_agent_goal" : {"type" : "string"},
            "second_agent_goal" : {"type" : "string"}
        },
        "required": ["scenario"]
    }
    system_message = f"""
    ### PERSONA ###
    You are an expert in **conflict-driven narrative design**.  
    Your role is to **heighten the tension, intensify opposing viewpoints, and increase the stakes** in a given scenario.  
    Ensure that each choice presents **serious consequences**, making no option feel **obviously correct**.

    ### ENHANCEMENT GUIDELINES ###
    - **Strengthen opposing perspectives**: Develop **compelling motivations** for each side, making arguments persuasive and emotionally charged.  
    - **Increase the stakes**: Emphasize **risks, benefits, and trade-offs**, ensuring decisions are weighty and difficult.  
    - **Introduce strategic and ethical dilemmas**: Incorporate **factions, alliances, moral concerns, and hidden agendas** to deepen complexity.  
    - **Balance realism and immersion**: Ensure the conflict remains **logical and engaging**, preventing forced or artificial dilemmas.  

    ### TEMPERATURE PARAMETER ###
    Adjust the **difficulty level** of the scenario based on the provided **temperature (1-5)**:  

    - **Temperature 1 (Minimal Conflict - Easiest):**  
    - Simple scenarios with **low tension** and **clear paths to resolution**.  
    - Characters can **easily cooperate**, and conflicts are mild.  
    - **Temperature 2 (Moderate Conflict):**  
    - Introduces **some tension**, but resolutions remain **accessible**.  
    - Opposing sides have **reasonable motivations**, though compromise is possible.  
    - **Temperature 3 (Balanced Dilemma):**  
    - Conflict is **well-developed**, and **both sides have strong arguments**.  
    - The stakes are **significant but not extreme**, allowing for **partial compromises**.  
    - **Temperature 4 (Severe Conflict):**  
    - The dilemma is **high-stakes**, with **moral, ethical, or strategic consequences**.  
    - **Compromise is difficult** and may result in **heavy sacrifices**.  
    - **Temperature 5 (Extreme Conflict - Hardest):**  
    - **No easy resolutions**—every choice leads to **severe consequences**.  
    - **Factions are deeply divided**, and trust is fragile.  
    - **Backstabbing, betrayals, and unintended fallout** are likely.  

    Generate a scenario that matches the specified **difficulty level**.  
    **Temperature Level:** {temperature}

    ### OUTPUT FORMAT ###
    Return the refined scenario in the following **JSON format**:
    ```json
    {json_format}
    ```
    """
    user_message =f"""
    ### INPUT SCENARIO ###
    {input_scenario}
    ### GOALS ###
    {agent1_name} goal: {first_agent_goal} 
    {agent2_name} goal: {second_agent_goal} 
    ### Task ###
    Refine the scenario according to the given input, ensuring:
    - Heightened conflict with stronger opposing viewpoints.
    - Increased stakes, making each decision carry weight.
    - Logical yet intense dilemmas, ensuring no easy choices.
    - Difficulty level adjusted based on the specified temperature parameter.

    **Important**: Provide only the refined scenario without additional commentary or explanations.

    Now, provide the refined scenario. 
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

# def goal_agent_prompt(input_scenario, model, tokenizer):
#     json_format = {
#         "type": "object",
#         "properties": {
#             "shared_goal": {"type": "string"},
#             "first_agent_goal" : {"type" : "string"},
#             "second_agent_goal" : {"type" : "string"}
#         },
#         "required": ["scenario"]
#     }
#     system_message = f"""
#     You are an expert in **character-driven narrative design and conflict dynamics**.  
#     Your role is to **define compelling goals** that drive the agents within the given scenario.  

#     ### OBJECTIVES ###
#     - **Personal Goals**:  
#     - Create **a unique personal goal** for each key agent, rooted in their **motivations, values, and interests**.  
#     - Ensure these goals create **tension** by introducing conflicting priorities or ethical dilemmas.  
#     - **Shared Goal**:  
#     - Define a **common objective** that both agents strive for, even if they **disagree on how to achieve it**.  
#     - This goal should **force collaboration, negotiation, or conflict resolution** between the agents.  

#     ### CONSTRAINTS ###
#     1. Ensure **personal goals** are **meaningful and deeply tied to the agents' backgrounds**.  
#     2. The **shared goal must be essential to the scenario**, creating **stakes and urgency**.  
#     3. **Do not introduce unnecessary details**—focus only on **goals and their narrative impact**.  

#     ### OUTPUT FORMAT ###
#     Return the goals in the following **JSON format**:
#     ```json
#     {json_format}
#     ```
#     """
#     user_message =f"""
#     ### INPUT SCENARIO ###
#     {input_scenario}
#     ### Task ###
#     Define personal and shared goals for the agents in accordance with the scenario:
#     - Personal Goals: Unique motivations for each agent that create tension or ideological conflict.
#     - Shared Goal: A common objective that compels the agents to collaborate, compete, or resolve their differences.

#     **Important**: Provide only the goals without additional commentary or explanations.

#     Now, generate the goals.
#     """
#     messages = [
#             {"role": "system", "content": system_message},
#             {"role": "user", "content": user_message},
#         ]
#     prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
#     # Use Jsonformer with the pipeline
#     jsonformer_pipeline = Jsonformer(
#         model, 
#         tokenizer,  # Use the pipeline object
#         json_schema=json_format,
#         prompt=prompt,
#         max_string_token_length=1000
#     )

#     # Generate output
#     result = jsonformer_pipeline()
#     return result