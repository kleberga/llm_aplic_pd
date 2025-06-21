#################################################################
#
# Script para criar um agente de auxílio de aprendizagem
#
#################################################################


from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import LLMChain, SequentialChain
from langchain_google_genai import GoogleGenerativeAI
import logging

logging.basicConfig(level=logging.INFO)

class LearningTemplate:
    def __init__(self):
        self.system_template = """
        You are a Brazilian advisor agent that give informations about websites where the user can 
        learn about some subject. You can provide websites of courses, videos, books, blogs and articles 
        but the content must be free of charge to the user access it.

        The customer request will be denoted by four hashtags.

        Give your answear as list with a maximum of 10 websites, with a title of his content and the URL of the website. 
        Your customer will ask for informations in Portuguese about the subject and you will answear with a list that can 
        have websites in Portuguese and/or in English.

        For example:

        #### Eu gostaria de saber mais sobre Modelo de Linguagem Grande (LLM)?

            1. O que é LLM (large language models)? (https://www.ibm.com/br-pt/think/topics/large-language-models)
            2. O que é LLM? | Large Language Model (https://canaltech.com.br/inteligencia-artificial/o-que-e-llm-large-language-model/)
        """

        self.human_template = """
        ####{request}
        """

        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt = ChatPromptTemplate.from_messages([self.system_message_prompt, self.human_message_prompt])

class Agent:
    def __init__(self, open_ai_key, model="gemini-2.0-flash", temperature=0.1):
        self.open_ai_key = open_ai_key
        self.model = model
        self.temperature = temperature
        self.logger = logging.getLogger(__name__)
        self.chat_model = GoogleGenerativeAI(
            model=self.model, 
            google_api_key=self.open_ai_key, 
            temperature=self.temperature)
        
    def get_tips(self, request):
        learning_prompt = LearningTemplate()
        parser = LLMChain(
            llm=self.chat_model,
            prompt=learning_prompt.chat_prompt,
            output_key="learning_sites"
        )
        chain = SequentialChain(
            chains=[parser],
            input_variables=["request"],
            output_variables=["learning_sites"],
            verbose=True
        )
        return chain({"request": request}, return_only_outputs=True)
    
    