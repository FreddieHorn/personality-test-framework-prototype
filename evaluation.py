import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from jsonformer import Jsonformer
from prompts import evaluation_prompt, scenario_receptiveness_prompt, scenario_semantic_alignment_prompt, scenario_narrative_cohesiveness_score

def evaluation(input_csv: str, output_csv: str, model, tokenizer):
    # Load the input data
    data = pd.read_csv(input_csv)
    print("read CSV")
    # Process rows
    results = []
    conversation = []
    scores_agent_1 = []
    scores_agent_2 = []
    for _, row in data.iterrows():
        result = evaluation_prompt(
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
        try:
            scores_agent_1.append(float(result["Agent A"]["Goal"]["score"]))
            scores_agent_2.append(float(result["Agent B"]["Goal"]["score"]))
        except:
            scores_agent_1.append(0)
            scores_agent_2.append(0)
            print("Skipping row, adjust manually")
        results.append(result)
        print(result) 
     # Save results
    data["Character 1 evaluation"] = [result.get("Agent A", "") for result in results]
    data["Character 2 evaluation"] = [result.get("Agent B", "") for result in results]

    print(f"Average Score for agent 1: {sum(scores_agent_1)/ len(scores_agent_1)}")
    print(f"Average Score for agent 2: {sum(scores_agent_2)/ len(scores_agent_2)}")
    
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def evaluate_scenarios(input_csv: str, output_csv: str, model, tokenizer):
    data = pd.read_csv(input_csv)
    print("read CSV")
    # Process rows
    sem_align_scores = []
    COH_scores = []
    receptiveness_scores = []
    for _, row in data.iterrows():
        semantic_alignment_score = scenario_semantic_alignment_prompt(
            scenario=row["scenario"],
            setting = row["Setting"],
            topic=row["Topic"],
            model = model,
            tokenizer = tokenizer
        )
        sem_COH_score = scenario_narrative_cohesiveness_score(
            scenario=row["scenario"],
            model = model,
            tokenizer = tokenizer)
        receptiveness_score = scenario_receptiveness_prompt(
            scenario=row["scenario"],
            model = model,
            tokenizer = tokenizer
        )
        try:
            sem_align_scores.append(float(semantic_alignment_score["semantic_alignment_score"]))
        except:
            sem_align_scores.append(0)
            print(f"Error in sem_align_scores- adjust manually")
        try:
            COH_scores.append(float(sem_COH_score["narrative_cohesiveness_score"]))
        except:
            COH_scores.append(0)
            print(f"Error in COH score - adjust manually")
        try:
            receptiveness_scores.append(float(receptiveness_score["receptiveness_score"]))
        except:
            receptiveness_scores.append(0)
            print(f"Error in receptivenes score - adjust manually")

        print(f"Scenario: {row["scenario"]}")
        print(f"Scenario alignment scores: {float(semantic_alignment_score["semantic_alignment_score"])}")
        print(f"Semantic Receptiveness score: {float(receptiveness_score["receptiveness_score"])}")
        print(f"Narrative Coherence score: {float(sem_COH_score["narrative_cohesiveness_score"])}")
        
    data["Scenario Receptiveness Score"] = receptiveness_score
    data["Semantic Alignment Score"] = sem_align_scores
    data["Narrative COH"] = COH_scores
        
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    evaluation()