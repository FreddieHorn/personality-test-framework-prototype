import json
import pandas as pd
from prompts import generate_interaction_prompt, agent_prompt, goal_completion_rate_prompt

def generate_interaction(input_csv: str, output_csv: str, model, tokenizer):
    print("_______GENERATING INTERACTION____________")
    data = pd.read_csv(input_csv)
    # Process rows
    results = []
    # 1 pos - first agent goal, 2 pos - second agent goal, 3 pos - shared goal
    all_completion_rates = []
    for _, row in data.iterrows():
        completion_rate = [0,0,0]
        completion_rates = [completion_rate]
        interaction = ""
        agent_1_name = row["Character1"]
        agent_2_name = row["Character2"]
        shared_goal=row["shared_goal"]
        first_agent_goal=row["first_agent_goal"]
        second_agent_goal=row["second_agent_goal"]
        scenario=row["scenario"]
        personality1=row["Personality1"]
        personality2=row["Personality2"]
        setting = row["Setting"]

        for i in range(1, 21, 2):
            print("in inner loop")
            response = agent_prompt(agent_1_name, scenario, setting, 
            shared_goal, first_agent_goal, personality1, interaction, model = model, tokenizer = tokenizer, turn = i-1)
            interaction+=f"{agent_1_name}:{response["response"]}\n"
            response = agent_prompt(agent_2_name, scenario, setting, 
            shared_goal, second_agent_goal, personality2, interaction, model = model, tokenizer = tokenizer, turn = i)
            interaction+=f"{agent_2_name}:{response["response"]}\n"
            completion_rates_json = goal_completion_rate_prompt(interaction, completion_rate, agent_1_name, agent_2_name, shared_goal, first_agent_goal, second_agent_goal,
                                                                scenario, model = model, tokenizer = tokenizer)
            completion_rate = [completion_rates_json["first_agent_goal_completion_rate"], completion_rates_json["second_agent_goal_completion_rate"], 
                                    completion_rates_json["shared_goal_completion_rate"]]
            completion_rates.append(completion_rate)
            print(completion_rate)

        results.append(interaction)
        all_completion_rates.append(completion_rates)
        print(interaction)
     # Save results
    data["interaction"] = results
    data["completion_rates"] = all_completion_rates 
    
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")