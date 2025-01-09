
class Agent:
    def __init__(self, personality, scenario:str, goal: str, start = False):
        self.scenario = scenario
        self.personality = personality # How is it represented?
        self.goal = goal
        self.agent_engine = self._set_personality()

    def _set_personality(self):
        # Initialization based on config - LLM 
        return None
    
    def take_turn(self, incoming_message = None):
        if not incoming_message:
            pre_message = "you will begin the conversation with the other agent in order to achieve your goal in the given scenario"
            output_message = self.agent_engine(pre_message+)
        else:
            output_message = self.agent_engine(incoming_message) # LLM processes utterance from the other agent and responds
        return output_message

# Class that emulates interaction between agents (agentic behaviour)
class AgentInteraction:
    def __init__(self, agents_config: dict, scenario_config: dict, turns: int)
        self.turns = turns
        self.agents_config = agents_config
        self.scenario_config = scenario_config
        self._initialize_agents()
    
    def _initialize_agents(self):
        # TODO agent initialization aka. how do we receive agent's config
        agent_1 = Agent(personality = self.agents_config["agent_1_personality"], 
                        scenario = self.scenario_config["description"],
                        goal = self.scenario_config["agent_1_goal"],
                        start = True)
        agent_2 = Agent(personality = self.agents_config["agent_2_personality"], 
                scenario = self.scenario_config["description"],
                goal = self.scenario_config["agent_2_goal"]
                )
    def conduct_interaction(self):
        episode = []
        last_utterance = None
        for i in range(turns):
            last_utterance = agent_1.take_turn(last_utterance)
            episode.append(f"agent_1: '{last_utterance}'")
            last_utterance = agent_2.take_turn(last_utterance)
            episode.append(f"agent_2: '{last_utterance}'")
        return episode
    
    
