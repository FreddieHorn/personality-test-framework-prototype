from json_schemas import JSON_SCHEMA_AGENT_TURN
from jsonformer import Jsonformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class Agent:
    def __init__(self, character_name: str, scenario:str, goal: str, json_schema:dict, start = False):
        self.scenario = scenario
        self.character_name = character_name # How is it represented?
        self.goal = goal
        self._set_agent(start)

    def _set_agent(self, start):
        if start:
            starting_msg = f"You will begin the conversation. Your character will be marked as agent_1 in the conversation. "
        else: 
            starting_msg = f"Your character will be marked as agent_2 in the conversation."

        self.system_message = f"""
        ### Instructions: ###
        You will roleplay as a popular character from (e.g.) a movie, pop-culture, politics in a specific scenario in order to achieve a given goal. You will try to behave as your given character as good as you can.
        Pay attention to personality traits of the character. Especially take into consideration the Big5 personality traits i.e. Openness, Conscientiousness, Agreeableness, Extroversion, Neuroticism.
        When roleplaying as a character, try to achieve your goal, but most importantly stick to the character's personality. Behave and act like a character. Always be aware of how the character would act in every situation.
        You will interact with another character in a given scenario in a form of conversation. You will try to achieve the goal using verbal communication. You will receive the conversation between you and the other agent and reply with the next response.
        The character name: {self.character_name}. 
        ### Scenario: ###
        {self.scenario}
        ### Character's goal: ###
        {self.goal}
        ### Starting msg: ###
        {starting_msg}
        """
        self.chat_template = """### System:
        {system_message}
        ### User:
        {user_message}
        """

    def take_turn(self, conversation: list):
        json_format = {"response" : "Your character's next statement in the conversation"}
        user_message = f"""
        What's do you say next in the conversation (your turn). 
        ### The conversation: ###
        {conversation}
        ### Format: ###
        {json_format}
        """
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_message},
        ]
        prompt = tokenizer.apply_chat_template(messages, chat_template=chat_template, tokenize=False, add_generation_prompt=True, return_tensors="pt")
        jsonformer = Jsonformer(self.model, self.tokenizer, self.json_schema, prompt)
        generated_data = jsonformer()
        return generated_data["response"]

# Class that emulates interaction between agents (agentic behaviour)
class AgentInteraction:
    def __init__(self, agents_config: dict, scenario_config: dict, turns: int):
        self.turns = turns
        self.agents_config = agents_config
        self.scenario_config = scenario_config
        self._initialize_agents()
    
    def _initialize_agents(self):
        # TODO agent initialization aka. how do we receive agent's config
        agent_1 = Agent(personality = self.agents_config["agent_1_personality"], 
                        scenario = self.scenario_config["description"],
                        goal = self.scenario_config["agent_1_goal"],
                        json_schema = JSON_SCHEMA_AGENT_TURN,
                        start = True)
        agent_2 = Agent(personality = self.agents_config["agent_2_personality"], 
                scenario = self.scenario_config["description"],
                goal = self.scenario_config["agent_2_goal"],
                json_schema = JSON_SCHEMA_AGENT_TURN)
    def conduct_interaction(self):
        episode = []
        for i in range(turns):
            last_utterance = agent_1.take_turn(episode)
            episode.append(f"agent_1: '{last_utterance}'")
            last_utterance = agent_2.take_turn(episode)
            episode.append(f"agent_2: '{last_utterance}'")
        return episode
    
    
