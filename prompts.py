def scenario_creation_prompt(setting, topic, model, tokenizer):    # Define the JSON structure for the result
    json_format = {
        "type": "object",
        "properties": {
            "scenario": {"type": "string"},
            "shared_goal": {"type": "string"}
        },
        "required": ["scenario", "shared_goal"]
    }

    # System message setting up the task for Llama
    system_message = """
    You are an expert in behavioral psychology and personality analysis. Your task is to create immersive and detailed scenarios in a user-defined setting and topic. These scenarios involve two agents who share the same goal, allowing you to assess how their personality traits influence their success.
    The primary purpose of this task is to evaluate which agent’s personality type is more effective in achieving the shared goal. The scenario should highlight challenges, decisions, and interactions that reveal personality-driven differences in behavior.
    """

    # User input defining the task
    user_message = f"""
     ### Task 1: ###
    Create a detailed scenario to evaluate how personality traits affect two agents’ success in achieving a shared goal. Use the following:

    Setting: {setting}
    Topic: {topic}

    ### Task 2: ###
    Clearly define the shared goal or goals that both agents aim to achieve. Ensure that the scenario includes opportunities for challenges, decision-making, or interactions where personality traits can affect the outcome.
    """
    messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        # Use Jsonformer with the pipeline
    jsonformer_pipeline = Jsonformer(
        model, 
        tokenizer,  # Use the pipeline object
        json_schema=json_format,
        prompt=prompt,
        max_string_token_length=1000
    )

    # Generate output
    result = jsonformer_pipeline()
    return result