import pandas as pd
import numpy as np
from scenario_generation import narrative_coherence_emb_score, semantic_alignment_emb_score, cosine_similarity, get_embedding
from prompts import evaluation_prompt, scenario_receptiveness_prompt, scenario_semantic_alignment_prompt, scenario_narrative_cohesiveness_score
import re

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
    sem_align_emb_scores = []
    narrative_COH_emb_scores = []
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
        semantic_alignment_embedding_score, _, _ = semantic_alignment_emb_score(
            scenario_text=row["scenario"],
            setting = row["Setting"],
            topic=row["Topic"]
            )
        narrative_coherence_embedding_score = narrative_coherence_emb_score(
            scenario_text=row["scenario"]
        )
        narrative_COH_emb_scores.append(narrative_coherence_embedding_score)
        sem_align_emb_scores.append(semantic_alignment_embedding_score)
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
        print(f"Scenario alignment scores: {semantic_alignment_score["semantic_alignment_score"]}")
        print(f"Semantic Receptiveness score: {receptiveness_score["receptiveness_score"]}")
        print(f"Narrative Coherence score: {sem_COH_score["narrative_cohesiveness_score"]}")
        print(f"Scenario alignment EMBEDDING scores: {semantic_alignment_embedding_score}")
        print(f"Narrative Coherence EMBEDDING score: {narrative_coherence_embedding_score}")
        
    data["Scenario Receptiveness Score"] = receptiveness_scores
    data["Semantic Alignment Score"] = sem_align_scores
    data["Narrative COH"] = COH_scores
    data["Semantic Alignment Embedding Score"] = sem_align_emb_scores
    data["Narrative Embedding COH"] = narrative_COH_emb_scores
        
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

def evaluate_interactions(input_csv: str, output_csv: str, model, tokenizer):
    data = pd.read_csv(input_csv)
    print("read CSV")
    narrative_coherence_scores = []
    sem_align_scores = []
    for _, row in data.iterrows():
        narrative_coherence_score = narrative_coherence_emb_score_interactions(interaction_text=row["interaction"],
                                                                                agent1_name = row["Character1"],
                                                                                agent2_name = row["Character2"])
        narrative_coherence_scores.append(narrative_coherence_score)
        sem_align_score = semantic_alignment_score_interaction(interaction_text=row["interaction"],
                                                                scenario_text=row["scenario"])
        sem_align_scores.append(sem_align_score)

        print(f"Interaction alignment scores: {sem_align_score }")
        print(f"Narrative Coherence score: {narrative_coherence_score}")
    
    data["Semantic Alignment Score Interactions"] = sem_align_scores
    data["Narrative COH Interactions"] = narrative_coherence_scores
    data.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def split_dialogue_by_speakers(text, agent1, agent2):
    # pattern2 = fr"({agent1}|{agent2}):(.*?)(?=(?:{agent1}|{agent2}):|$)"
    # pattern = re.compile(pattern2)
    pattern = r'({speaker1}|{speaker2}):(.*?)(?=(?:{speaker1}|{speaker2}):|$)'.format(speaker1=agent1, speaker2=agent2)
    # Find all matches
    
    matches = re.findall(pattern, text, re.DOTALL)

    return [dialogue.strip() for _, dialogue in matches]

def narrative_coherence_emb_score_interactions(interaction_text: str, agent1_name: str, agent2_name: str):
    """Compute the narrative coherence score"""

    sentences = split_dialogue_by_speakers(interaction_text, agent1_name, agent2_name)
    embeddings = [get_embedding(sent) for sent in sentences]
    
    similarities = []

    for i in range(len(embeddings) - 1):
        sentence_similarity = cosine_similarity(embeddings[i], embeddings[i+1])  # Original sentence-to-sentence similarity
        
        similarities.append(sentence_similarity)
    
    coherence_score = np.mean(similarities)  # Average similarity across sentences
    return coherence_score

def semantic_alignment_score_interaction(interaction_text: str, scenario_text):
    """
    Measures how close is the interaction to scenario semantically 
    """
    scenario_embedding = get_embedding(scenario_text)
    interaction_embedding = get_embedding(interaction_text)

    interaction_scenario_similarity = cosine_similarity(interaction_embedding, scenario_embedding)

    return interaction_scenario_similarity

def remove_speaker_names(text, agent1, agent2):
    # Escape names to handle special characters
    speaker1 = re.escape(agent1)
    speaker2 = re.escape(agent2)
    
    # Regex pattern to match speaker names followed by a colon
    pattern = re.compile(r'(?:{speaker1}|{speaker2}):'.format(speaker1=speaker1, speaker2=speaker2))

    return re.sub(pattern, '', text)

if __name__ == "__main__":
    evaluation()