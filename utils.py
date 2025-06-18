import csv
import random


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
