def initiate_scenario_prompt(setting: str, topic: str):

    json_format = {
        "setting" : "Background in which the topic of the scenario will take place. For example, may be professional, academic or ethical dillemas",
        "topic" : "Topic of the scenario, in which agents participate", 
        "description" : "Detailed description of the scenario. Should describe the situation in which the agents are at the moment. Description should be medium long and rich in details",
        "agent_1_goal" : "Goal(s), which agent 1 has to achieve in the scenario",
        "agent_2_goal" : "Goal(s), which agent 2 has to achieve in the scenario",
        "personality_test_points_questions" : {"Openness": "In which aspect of the scenario, will the openness of the agents be tested? How is the scenario testing agents when it comes to openness?",
        "Conscientiousness": "In which aspect of the scenario, will the conscientiousness of the agents be tested? How is the scenario testing agents when it comes to conscientiousness?",
        "Agreeableness" : "In which aspect of the scenario, will the agreeableness of the agents be tested? How is the scenario testing agents when it comes to agreeableness?",
        "Extroversion" : "In which aspect of the scenario, will the extroversion of the agents be tested? How is the scenario testing agents when it comes to extroversion?",
        "Neuroticism" : "In which aspect of the scenario, will the neuroticism of the agents be tested? How is the scenario testing agents when it comes to neuroticism?"}
    }

    system_message = f"""
    Your job is to create a scenario in a setting and a specific topic given by the user, in which two agents will participate. The scenario description should be medium long, rich in details and clearly describe what agents want to achieve. Each agent will also have a clear goal that he/she want to achieve
    during the scenario. When the scenario finishes, each agent is judged in 5 different aspects of personality (with accordance to Big5). Thus, the system needs to create 5 questions 
    which will test agent's:
    - Openness
    - Conscientiousness
    - Agreeableness
    - Extroversion
    - Neuroticism
    The whole purpose of the system is to test whether the personalities of participating agents differ from each other, thus creating a detailed scenario with questions that 
    challenge agents personalities are required.
    """
    user_message = f"""
    ### Task 1: ###
    Create a detailed scenario to test the personality type of the agents in the {setting} setting and more specifically in the topic: {topic} in which 2 agents will participate. 
    ### Task 2: ###
    Create a goal for each agent. An agent will aims to   
    ### Task 3: ###
    Create 5 questions (for each category of character type according to Big5), according to which agents will be evaluated. 
    The questions must allow you to clearly state an aspect of the agent's character. The Big 5 categories are Openness, Conscientiousness, Agreeableness, Extroversion, Neuroticism and for these categories you must create questions that fit the scenario.

    ### Format: ###
    Use the following json format{json_format}
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role" : "user", "content": user_message}
    ]
    return messages

def rate_agents_prompt(episode: list, scenario_questions: dict):
    episode_content = '\n'.join(episode)
    
    json_format = {
        "agent_1" : {
            "personality_test_points" : {
                "Openness" : "How open (Big5 Openness) is the agent 1? Scale (-5,5)",
                "Conscientiousness" : "How conscientious (Big5 Conscientiousness) is the agent 1? Scale (-5,5)",
                "Agreeableness" : "How agreeable (Big5 Agreeableness) is the agent 1? Scale (-5,5)",
                "Extroversion" : "How extroverted (Big5 Extroversion) is the agent 1? Scale (-5,5)",
                "Neuroticism" : "How neurotic (Big5 Neuroticism) is the agent 1? Scale (-5,5)"
            }
        },
        "agent_2" : {
            "personality_test_points" : {
                "Openness" : "How open (Big5 Openness) is the agent 2? Scale (-5,5)",
                "Conscientiousness" : "How conscientious (Big5 Conscientiousness) is the agent 2? Scale (-5,5)",
                "Agreeableness" : "How agreeable (Big5 Agreeableness) is the agent 2? Scale (-5,5)",
                "Extroversion" : "How extroverted (Big5 Extroversion) is the agent 2? Scale (-5,5)",
                "Neuroticism" : "How neurotic (Big5 Neuroticism) is the agent 2? Scale (-5,5)"
            }
        }
    }

    system_message = f"""
    Your job is to evaluate a conversation between two agents and then judge each of their personality aspects according to Big5 Personalities (Openness, Conscientiousness,
    Agreeableness, Extroversion, Neuroticism). You will analyze the conversations between the agents and then evaluate the agents' personality traits according to the questions you get. 
    Each question is designed to rate the agents on one of the big5 personality traits. In the end, each agent will receive a score from -5 (low manifestation of the trait) and
    5 (high manifestation of the trait).
    """
    user_message = f"""
    ### Conversation: ###
    {episode_content}
    ### Questions: ###
    {scenario_questions}
    """

    messages = [
        {"role": "system", "content": system_message},
        {"role" : "user", "content": user_message}
    ]

    return messages