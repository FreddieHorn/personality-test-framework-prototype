import json
import pandas as pd
from prompts import generate_interaction_prompt

def generate_interaction(input_csv: str, output_csv: str, model, tokenizer):
    print("_______GENERATING INTERACTION____________")
    data = pd.read_csv(input_csv)
    # Process rows
    results = []
    for _, row in data.iterrows():
        result = generate_interaction_prompt(
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
    data["interaction"] = [result.get("interaction", "") for result in results]
    
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")