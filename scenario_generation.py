import torch
import pandas as pd
import numpy as np
import re
from transformers import AutoModel, AutoTokenizer
from prompts import scenario_creation_prompt, concept_agent_prompt, narrative_agent_prompt, logical_consistency_agent_prompt, conflict_agent_prompt
emb_model_name = "sentence-transformers/all-MiniLM-L6-v2"
emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
emb_model = AutoModel.from_pretrained(emb_model_name)

def scenario_generation(input_csv: str, output_csv: str,  model, tokenizer, mode = 'default', temperature = 1):
    # Load the input data
    data = pd.read_csv(input_csv)

    # Process rows
    results = []

    if mode == 'default':
        for _, row in data.iterrows():
            result = scenario_creation_prompt(
                setting=row["Setting"],
                topic=row["Topic"],
                agent_1_name = row["Character1"],
                agent_2_name = row["Character2"],
                temperature = temperature,
                model = model,
                tokenizer = tokenizer
            )
            results.append(result)
            print(result) 
    elif mode == 'agentic':
        for _, row in data.iterrows():
            agent1_name = row["Character1"], 
            agent2_name= row["Character2"], 
            setting=row["Setting"],
            topic=row["Topic"],
            step_1_scenario = concept_agent_prompt(agent1_name, 
                                                agent2_name, 
                                                setting,
                                                topic,
                                                model=model, tokenizer=tokenizer)
            shared_goal = step_1_scenario["shared_goal"]
            first_agent_goal = step_1_scenario["first_agent_goal"]
            second_agent_goal = step_1_scenario["second_agent_goal"]
            step_2_scenario = narrative_agent_prompt(step_1_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, model=model, tokenizer=tokenizer)
            step_3_scenario = conflict_agent_prompt(step_2_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, model=model, tokenizer = tokenizer, temperature = temperature)
            result_scenario = logical_consistency_agent_prompt(step_3_scenario["scenario"], shared_goal, first_agent_goal, second_agent_goal, agent1_name, agent2_name, model = model, tokenizer = tokenizer)
            results.append(result_scenario)

     # Save results
    data["scenario"] = [result.get("scenario", "") for result in results]
    data["shared_goal"] = [result.get("shared_goal", "") for result in results]
    data["first_agent_goal"] = [result.get("first_agent_goal", "") for result in results]
    data["second_agent_goal"] = [result.get("second_agent_goal", "") for result in results]
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def get_embedding(sentence):
    tokens = emb_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = emb_model(**tokens)
    return output.last_hidden_state.mean(dim=1)  # Get sentence embedding

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors"""
    return torch.nn.functional.cosine_similarity(vec1, vec2).item()

def split_sentences(text):
    """Split text into sentences using regex"""
    return re.split(r'(?<=[.!?]) +', text)  # Splitting at punctuation marks

def semantic_alignment_emb_score(scenario_text: str, setting:str, topic: str):
    scenario_embedding = get_embedding(scenario_text)
    setting_embedding = get_embedding(setting)
    topic_embedding = get_embedding(topic)

    setting_scenario_similarity = cosine_similarity(scenario_embedding, setting_embedding)
    topic_scenario_similarity = cosine_similarity(scenario_embedding, topic_embedding)

    overall_score = (setting_scenario_similarity + topic_scenario_similarity)/2

    return overall_score, setting_scenario_similarity, topic_scenario_similarity

def narrative_coherence_emb_score(scenario_text: str):
    """Compute the narrative coherence score"""
    sentences = split_sentences(scenario_text)

    if len(sentences) < 2:
        return 1.0  # If only one sentence, coherence is perfect

    embeddings = [get_embedding(sent) for sent in sentences]
    
    similarities = []

    for i in range(len(embeddings) - 1):
        sentence_similarity = cosine_similarity(embeddings[i], embeddings[i+1])  # Original sentence-to-sentence similarity
        
        similarities.append(sentence_similarity)
    
    coherence_score = np.mean(similarities)  # Average similarity across sentences
    return coherence_score