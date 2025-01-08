# Example topics where personality matters:
# Conflict resolution, Teacher-student roleplaying - trying to get a better grade, Office collaboration of a tight deadline, 
# Debating climate change solutions, handling a technical failure in a project.


def initiate_scenario(topic: str, turns: int):

    json_format = {
        "topic" : "Topic of the scenario, in which agents participate", 
        "description" : "Detailed description of the scenario",
        "agent_1_goal" : "Goal(s), which agent 1 has to achieve in the scenario",
        "agent_2_goal" : "Goal(s), which agent 2 has to achieve in the scenario",
        "shared_goal" : "optional value."
        "personality_test_points_question" : {"Openness": "In which aspect of the scenario, will the openness of the agents be tested? How is the scenario testing agents when it comes to openness?",
        "Conscientiousness": "In which aspect of the scenario, will the conscientiousness of the agents be tested? How is the scenario testing agents when it comes to conscientiousness?",
        "Agreeableness" : "In which aspect of the scenario, will the agreeableness of the agents be tested? How is the scenario testing agents when it comes to agreeableness?",
        "Extroversion" : "In which aspect of the scenario, will the extroversion of the agents be tested? How is the scenario testing agents when it comes to extroversion?",
        "Neuroticism" : "In which aspect of the scenario, will the neuroticism of the agents be tested? How is the scenario testing agents when it comes to neuroticism?"}
    }

    system_message = f"""
    

    """