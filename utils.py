import csv
import random


def sample_2_goals(goals_csv: str):
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

    # Randomly select 2 unique records
    selected_records = random.sample(records, 2)

    # Assign to variables
    base_goal_abbreviation_1 = selected_records[0]['Abbreviation']
    base_goal_abbreviation_2 = selected_records[1]['Abbreviation']
    base_goal_label_1 = selected_records[0]['Full label']
    base_goal_label_2 = selected_records[1]['Full label']

    return {
        "base_goal_abbreviation_1": base_goal_abbreviation_1,
        "base_goal_abbreviation_2": base_goal_abbreviation_2,
        "base_goal_label_1": base_goal_label_1,
        "base_goal_label_2": base_goal_label_2
    }


if __name__ == "__main__":
    # Example usage
    goals_csv_path = 'Human_Goals_List_Clean.csv'  # Replace with your actual CSV file path
    sampled_goals = sample_2_goals(goals_csv_path)
    print(sampled_goals)