JSON_SCHEMA_SCENARIO = {
        "type": "object",
        "properties" : {
            "setting" : {"type": "string"},
            "topic" : {"type": "string"},
            "description" : {"type": "string"},
            "agent_1_goal" : {"type": "string"},
            "agent_2_goal" : {"type": "string"},
            "personality_test_points_questions" : {"type" : "object", "properties":{"Openness": {"type": "string"},
        "Conscientiousness": {"type": "string"},
        "Agreeableness" : {"type": "string"},
        "Extroversion" : {"type": "string"},
        "Neuroticism" : {"type": "string"}}
        }
    },
    "required": ["setting", "topic", "description","agent_1_goal", "agent_2_goal","personality_test_points_questions"]}

JSON_SCHEMA_AGENT_TURN = {
    "type" : "object",
    "properties" : {"response" : {"type": "string"}}
}