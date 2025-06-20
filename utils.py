import csv
import random
import re
import json

def sample_shared_goal(goals_csv: str):
    """
    Sample a specified number of goals from a CSV file containing goals.
    
    Args:
        goals_csv (str): Path to the CSV file containing goals.
        num_samples (int): Number of goals to sample. Default is 2.

    Returns:
        dict: A dictionary containing sampled goals.
    """
    with open(goals_csv, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        records = list(csv_reader)  # Convert to list to work with it

    # Randomly select 1 unique record
    selected_records = random.sample(records, 1)

    # Assign to variables
    base_goal_shared_abbreviation = selected_records[0]['Abbreviation']
    base_goal_shared_full_label = selected_records[0]['Full label']

    return {
        "base_goal_shared_abbreviation": base_goal_shared_abbreviation,
        "base_goal_shared_full_label": base_goal_shared_full_label
    }


def extract_json_string(raw_response: str) -> str:
    """
    Extracts a JSON block from a Markdown-style raw response and returns it as a JSON-formatted string.
    
    Args:
        raw_response (str): The raw response containing a JSON block (e.g., inside ```json ... ```).
        
    Returns:
        str: A valid JSON-formatted string.
        
    Raises:
        ValueError: If no JSON block is found or if JSON is invalid.
    """
    # match = re.search(r"```json\s*(\{.*?\})\s*```", raw_response, re.DOTALL)
    match = re.search(r'\{.*\}', raw_response, re.DOTALL)
    json_block = match.group()
    if not match:
        raise ValueError("No valid JSON block found in the response.")
    
    data = json.loads(json_block)
    return data