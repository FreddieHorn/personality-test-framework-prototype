from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableMap, RunnableLambda, RunnablePassthrough, RunnableParallel
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import SequentialChain, LLMChain

import os
from scenario_generation.templates import SCENARIO_GENERATION_TEMPLATE, CONFLICT_STRATEGIST_TEMPLATE, CONSISTENCY_TEMPLATE

load_dotenv()

class MultiAgentScenarioGeneration:
    def __init__(self, model_name: str = "deepseek/deepseek-chat-v3-0324:free", provider: str | None = None):
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPEN_ROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            extra_body={
                "provider": {"only": [provider]}
            } if provider else {}
        )
        scenario_generation_template = PromptTemplate.from_template(
            template=SCENARIO_GENERATION_TEMPLATE
        )

        conflict_strategist_template = PromptTemplate.from_template(
            template=CONFLICT_STRATEGIST_TEMPLATE
        )

        consistency_template = PromptTemplate.from_template(
            template=CONSISTENCY_TEMPLATE
        )
        scenario_gen_chain = (
             scenario_generation_template
            | llm
            | StrOutputParser()
        )

        conflict_chain = (
            conflict_strategist_template
            | llm
            | StrOutputParser()
        )

        consistency_chain = (
            consistency_template
            | llm
            | StrOutputParser()
        )
        self.barebones_chain = RunnableSequence({
                "initial_scenario": scenario_gen_chain,
                "difficulty": RunnablePassthrough(),
                "scenario_setting": RunnablePassthrough()
            },
            {
                "conflict_scenario": conflict_chain,
                "scenario_setting": RunnablePassthrough(),
                "difficulty": RunnablePassthrough()
            },
            {
                "final_scenario": consistency_chain
            })

    def generate_scenario(self, scenario_setting: str, difficulty: str) -> str:
        result = self.barebones_chain.invoke({
            "scenario_setting": scenario_setting,
            "difficulty": difficulty
        })
        return result["final_scenario"]


def log_step(input_dict, step_name):
    print(f"\n--- {step_name} ---\nInput: {input_dict}\n")
    return input_dict
