import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from json_schemas import JSON_SCHEMA_SCENARIO
# from llm_funcs import create_scenario
# from environment import Agent, AgentInteraction
from jsonformer import Jsonformer

def llama_3_prompt(setting, topic, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"}
        },
        "required": ["scenario", "shared_goal"]
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
    Clearly define the shared goal or goals that both agents aim to achieve. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
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
            setting=row["Setting"],
            topic=row["Topic"],
            model = model,
            tokenizer = tokenizer
        )
        results.append(result)
        print(result) 
     # Save results
    data["scenario"] = [result.get("scenario", "") for result in results]
    data["shared_goal"] = [result.get("shared_goal", "") for result in results]
    
    data.to_csv(input_csv, index=False)
    print(f"Results saved to {input_csv}")

    # SETTING = "PROFFESIONAL "
    # TOPIC = "Office collaboration of a tight deadline"

    # print(f"Scenario creation with a setting: {SETTING} and a topic: {TOPIC}...")
    # scenario_config = create_scenario(setting=SETTING, topic=TOPIC, model=model, tokenizer=tokenizer, json_schema=JSON_SCHEMA_SCENARIO)

    # with open("scenario.json", "w") as file:
    #     json.dump(scenario_config, file, indent=4)  # 'indent' makes the file more readable

    # # Below code has errors heh
    # agents_config = {"agent_1_personality" : "Donald Trump",
    #                 "agent_2_personality" : "Clark Kent"}
    # interaction_framework = AgentInteraction(agents_config = agents_config,
    #                                         scenario_config = scenario_config, turns=10)
    # print(f"Beggining the conversation between two agents...")
    # conversation = interaction_framework.conduct_interaction()

    # print(f"Writing the convo to the file")
    # with open(filename, "w") as file:
    #     for item in conversation:
    #         file.write(f"{item}\n")

    # Evaluate agents


if __name__ == "__main__":
    main()