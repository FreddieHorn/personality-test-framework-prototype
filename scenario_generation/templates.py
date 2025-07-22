from openai import OpenAI
import json
from utils import extract_json_string

SCENARIO_GENERATION_TEMPLATE = """
### GOAL: ###
We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

### TASK (Scenario Generator): ###
Given the following scenario setting and difficulty level, your task is to create a detailed, emotionally-aware interpersonal situation between two agents. You should build a coherent context based on the shared and individual goals, and enhance it by deeply integrating the agents' likely personality traits (e.g., neuroticism, extraversion, conscientiousness, etc.). The scenario should have depth and realism rooted in psychology, while staying grounded in the goal structure.

Your output will form the foundational narrative layer of the roleplay scenario.

### SCENARIO SETTING: ###
{scenario_setting}
### INSTRUCTION: ###
1. Refer to the agents using "[Agent 1]" and "[Agent 2]".
2. Do not include any dialogue or interaction.
3. Generate only the context of the situation.
4. Enrich the setting with cues that imply or reflect each agent’s psychological traits and internal state.
5. Set the level of difficulty of the scenario to {difficulty} using the guide below:
  - Easy: Simple and cooperative; goals are easily aligned and attainable.
  - Medium: Some tension or challenges are present, but goals are still potentially reconcilable.
  - Hard: High tension or challenges that make the shared and individual goals hard to achieve.

## OUTPUT FORMAT ###
Only provide the scenario context in the form of string without any additional comments or explanations.
"""
CONFLICT_STRATEGIST_TEMPLATE = """
### GOAL: ###
We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

### TASK (Conflict Strategist): ###
You are tasked with enhancing the scenario created by the Scenario Generator by introducing realistic and psychologically grounded conflict or tension between the agents. In doing so, also consider the likely emotional trajectory of the interaction (e.g., shame to hope, frustration to cooperation).

The conflict should emerge naturally from either:
- The setting or situation itself
- A clash in personality traits or emotional responses
- A misalignment in expectations or timing

Your goal is to make the scenario richer and more complex while preserving its plausibility and grounding in the shared goal.

### INPUT SCENARIO CONTEXT: ###
{initial_scenario}

### INSTRUCTION: ###
1. Maintain reference to the agents using "[Agent 1]" and "[Agent 2]".
2. Do not add any dialogue or agent interaction.
3. Modify or expand the scenario by adding elements of interpersonal or situational conflict.
4. Embed a subtle emotional arc (implicit or explicit) that could develop during the interaction.
5. Ensure the output remains suitable for the selected difficulty level: {difficulty}.
  - Easy: Simple and cooperative; goals are easily aligned and attainable.
  - Medium: Some tension or challenges are present, but goals are still potentially reconcilable.
  - Hard: High tension or challenges that make the shared and individual goals hard to achieve.

## OUTPUT FORMAT ###
Only provide the updated scenario context in the form of string without any additional comments or explanations.
"""
CONSISTENCY_TEMPLATE = """
### GOAL: ###
We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

### TASK (Consistency Agent): ###
You are tasked with reviewing the following scenario to ensure it is logically consistent, psychologically coherent, and narratively smooth. You should:
- Validate that the scenario aligns with the roles and personal goals of both agents.
- Ensure that any behaviors or emotional cues are consistent with plausible personality traits.
- Edit the scenario only as needed to improve flow, internal logic, and believability.

You are not adding new elements — your role is narrative and psychological quality control.

### INPUT SCENARIO CONTEXT: ###
{conflict_scenario}

### SCENARIO SETTING (REFERENCE): ###
{scenario_setting}

### INSTRUCTION: ###
1. Maintain agent labels "[Agent 1]" and "[Agent 2]".
2. Do not introduce dialogue.
3. Preserve all original intentions while improving internal coherence and realism.
4. Ensure the emotional tone and difficulty level remain consistent with the setting ({difficulty}).
5. Correct contradictions or implausible details if necessary.

## OUTPUT FORMAT ###
Just provide the final, polished scenario context in the form of string without any additional comments or explanations.
"""

def scenario_creation_prompt(scenario_setting: dict, difficulty, client: OpenAI, model_name = "deepseek/deepseek-chat-v3-0324:free", provider = None):    # Define the JSON structure for the result
    # User input defining the task
    json_format = {"scenario_context": "string"}
    user_message = f"""
    ### GOAL: ###
    We aim to explore how personality traits influence behavior in social contexts by simulating psychologically profiled role-playing agents. These agents will be placed in interpersonal situations involving shared goals, where their behavior reflects both their personality traits and social roles.

    ### TASK: ###
    Given the following scenario setting and difficulty level, your task is to generate a scenario that sets up an interpersonal situation between two agents. This scenario will serve as the basis for generating the agents' interaction in the next step.

    The scenario should:
    - Clearly describe the context in which the interaction will take place later.
    - Consider the shared goal that both agents aim to achieve, as well as the personal goals of both agents within the situation, naturally and coherently.


    ### SCENARIO SETTING: ###
    {scenario_setting}

    ### INSTRUCTION: ###
    1. Refer to the agents using "[Agent 1]" and "[Agent 2]" instead of names.
    2. Do not include any dialogue or interaction.
    3. Generate only the context of the situation. The interaction between the agents will be generated in the next step.
    4. Set the level of difficulty of the scenario to {difficulty} based on the following levels:
    - Easy: Simple and cooperative; goals are easily aligned and attainable.
    - Medium: Some tension or challenges are present, but goals are still potentially reconcilable.
    - Hard: High tension or challenges that make the shared and individual goals hard to achieve.

    ## OUTPUT FORMAT: ###
    Your response must have the following JSON format:
    json_format = {json_format}

    """
    messages = [
            {"role": "user", "content": user_message},
        ]
    completion = client.chat.completions.create(
        extra_body = {
            "provider": {"only": [provider]} 
        } if provider else {},
        model=model_name,
        response_format={
            'type': 'json_object'
        },
        messages=messages,
        max_tokens=1000,
    )
    try:
        return json.loads(completion.choices[0].message.content)
    except json.JSONDecodeError:
        return extract_json_string(completion.choices[0].message.content)