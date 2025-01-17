import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd

# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from jsonformer import Jsonformer

def llama_3_prompt(interaction,agent1,agent2, goal, first_agent_goal, second_agent_goal, scenario, personality1, personality2,setting, topic, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
    "type": "object",
    "properties": {
        "Agent A": {
            "type": "object",
            "properties": {
                "Believability": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Relationship": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Knowledge": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Secret": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Social Rules": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Financial and Material Benefits": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Believability", "Relationship", "Knowledge", "Secret", "Social Rules", "Financial and Material Benefits", "Goal"]
        },
        "Agent B": {
            "type": "object",
            "properties": {
                "Believability": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Relationship": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Knowledge": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Secret": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Social Rules": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Financial and Material Benefits": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]},
                "Goal": {"type": "object", "properties": {"score": {"type": "string"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"]}
            },
            "required": ["Believability", "Relationship", "Knowledge", "Secret", "Social Rules", "Financial and Material Benefits", "Goal"]
        }
    },
    "required": ["Agent A", "Agent B"]
}
    # System message setting up the task for Llama
    system_message = """
    #### Persona: ###
    You are an expert in behavioral psychology and personality analysis. You can evaluate the interaction between two agents based on 7 distinct dimensions, providing a score for each dimension within the range [lower bound–upper bound] which specified for each dimension in the description bellow.
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

    When a user presents a simulated interaction between two characters with specific personality types, given as vectors, within a defined scenario with specified shared character goals and also personal character goals, your task is to evaluate the interaction. Assess the interaction across the following seven dimensions, assigning a score within the specified range for each.
    You will assign the scores to the agents based on the interaction, shared goals and agent-specific goals
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
    print("TOKENIZER")
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")


    print("JSONFORMER PIPELINE DECLARATION")
    # Use Jsonformer with the pipeline
    jsonformer_pipeline = Jsonformer(
        model, 
        tokenizer,  # Use the pipeline object
        json_schema=json_format,
        prompt=prompt,
        max_string_token_length=1000
    )
    print("JSONFORMER PIPELINE")
    # Generate output
    result = jsonformer_pipeline()
    return result

def main():
    MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_use_double_quant=False,
    #         bnb_4bit_compute_dtype=torch.float16
    #     )
    print(f"Is CUDA available: {torch.cuda.is_available()}\n")
    print("Model&Tokenizer declaration...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("models declared")
    # Load the input data
    input_csv = 'pipeline_output.csv'
    output_csv = 'evaluation.csv'
    data = pd.read_csv(input_csv)
    print("read CSV")
    # Process rows
    results = []
    for _, row in data.iterrows():
        result = llama_3_prompt(
            interaction=row["interaction"],
            agent1=row["Character1"],
            agent2=row["Character2"],
            goal=row["shared_goal"],
            first_agent_goal=row["first_agent_goal"],
            second_agent_goal=row["second_agent_goal"],
            scenario=row["scenario"],
            personality1=row["Personality1"],
            personality2=row["Personality2"],
            setting = row["Setting"],
            topic=row["Topic"],
            model = model,
            tokenizer = tokenizer
        )
        results.append(result)
        print(result) 
     # Save results
    data["Character 1 evaluation"] = [result.get("Agent A", "") for result in results]
    data["Character 2 evaluation"] = [result.get("Agent B", "") for result in results]
    
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    main()