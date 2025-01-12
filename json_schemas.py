JSON_SCHEMA_SCENARIO = {
        "type": "object",
        "properties" : {
            "setting" : "string"
            "topic" : "string", 
            "description" : "string",
            "agent_1_goal" : "string",
            "agent_2_goal" : "string",
            "personality_test_points_questions" : {"type" : "object", "properties":{"Openness": "string",
        "Conscientiousness": "string",
        "Agreeableness" : "string",
        "Extroversion" : "string",
        "Neuroticism" : "string"}
        }
    }}

JSON_SCHEMA_AGENT_TURN = {
    "type" : "object",
    "properties" : {"response" : "string"}
}