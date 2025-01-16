import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from json_schemas import JSON_SCHEMA_SCENARIO
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from jsonformer import Jsonformer

def llama_3_prompt(agent1,agent2, goal, scenario, personality1, personality2,setting, topic, model, tokenizer):    # Define the JSON structure for the result
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
    You are an expert in behavioral psychology and roleplay simulation. Your task is to roleplay as two distinct characters within a given scenario, where both must work together to achieve a shared goal. Each character’s personality is defined by the Big Five traits—Openness, Conscientiousness, Agreeableness, Extroversion, and Neuroticism. These traits are represented by a vector of five numbers and will guide their behavior, dialogue, and decisions throughout the interaction.

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

    # Load the input data
    input_csv = '/home/s33zganj/personality-test-framework/pesonality_test_Sheet.csv'
    data = pd.read_csv(input_csv)

    # Process rows
    results = []
    for _, row in data.iterrows():
        result = llama_3_prompt(
            agent1=row["Character1"],
            agent2=row["Character2"],
            goal=row["shared_goal"],
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
    data["interaction"] = [result.get("interaction", "") for result in results]
    
    data.to_csv(input_csv, index=False)
    print(f"Results saved to {input_csv}")


if __name__ == "__main__":
    main()