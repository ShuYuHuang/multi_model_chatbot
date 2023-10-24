import langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

from langchain.llms.base import LLM, BaseLLM
from typing import Optional, List, Dict, Mapping, Any
from pydantic import Field

import pdb

import requests
# import os
# os.environ['CURL_CA_BUNDLE'] = ''

VICUNA_URL = 'http://10.100.100.104:8013/v1/completions'
VICUNA_HEADER = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
}

# messages_template = [
#     {
#       "content": "You are a helpful assistant.",
#       "role": "system"
#     },
#     {
#       "content": "What is the capital of France?",
#       "role": "user"
#     }
#   ]


class VicunaLLM(LLM):
    model_name: str = Field(None, alias='model_name')
    model_url: str = Field(None, alias='model_url')
    
    temperature: str = Field(None, alias='temperature')

    max_tokens: str = Field(None, alias='max_tokens')
    """The maximum number of tokens to generate in the completion.
    -1 returns as many tokens as possible given the prompt and
    the models maximal context size."""
    frequency_penalty: float = 0.0
    """Penalizes repeated tokens according to frequency."""
    presence_penalty: float = 0.0
    """Penalizes repeated tokens."""

    @property
    def _llm_type(self) -> str:
        return "vicuna"
    
    def __init__(
            self,
            model_name: str = "vicuna-13b-v1.5-16k",
            model_url: str = VICUNA_URL,
            max_tokens: int = 512,
            temperature: int = 0.0):
        super().__init__()
        self.model_name = model_name
        self.model_url  = model_url
        """What sampling temperature to use."""
        self.max_tokens = max_tokens
        self.temperature = temperature
    @property
    def _get_model_default_parameters(self):
        return {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """
        Get all the identifying parameters
        """
        return {
            # 'model_path' : self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    
    def _call(self, prompt: str, suffix: str = "", stop: Optional[List[str]] = ["Observation:"]) -> str:
        import requests
        json_data = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            'prompt': prompt,
            'temperature': self.temperature,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
        }
        response = requests.post(
            self.model_url,
            headers=VICUNA_HEADER,
            json=json_data)
        # response.raise_for_status()
        return response.json()['choices'][0]['text']


if __name__ == '__main__':
    from time import time

    print(requests.__file__)
    print(langchain.__file__)
    langchain.verbose=True


    # import langchain_visualizer
    # import asyncio

    # from langchain import set_tracing_callback_manager 

    # First, let's load the language model we're going to use to control the agent.
    llm = VicunaLLM()

    QUERY_TEMPLATE="""{question}"""
    template = PromptTemplate(
        input_variables=["question"],
        template=QUERY_TEMPLATE,
        validate_template=True
    )

    simple_chatbot = LLMChain(
        llm=llm,
        prompt=template
    )

    # prompts = ['Write me a python script that calculate 5+5',
    # 'Write me a python script that print "Hello World"' ,
    # 'Write me a poam about Paris',
    # 'What is the biggist country?',
    # 'What is the capital of UAE? give me simple answer']

    prompts = [
    """Determine the characters for the transcript in SRT format, and assign the character to the beginning of transcript, e.g. \n99\n00:32:01.000 --> 00 00:32:02.000\nSpeaker C: The suspect didn't show up last night\n\nHere is the input:\n0\n00:00:00.000 --> 00:00:03.200\n Doesn't really look like anyone's been doing cocaine off that table, does it?\n\n1\n00:00:05.600 --> 00:00:07.900\n With all due respect, I'm not sure you know how that works.\n\n2\n00:00:08.800 --> 00:00:10.200\n I'm asking if you do.\n\n3\n00:00:10.800 --> 00:00:12.200\n You've testified you've done cocaine.\n\n4\n00:00:12.200 --> 00:00:12.800\n I have.\n\n5\n00:00:13.300 --> 00:00:17.400\n Doesn't really look like Mr. Depp or anyone was doing cocaine off that table, does it?\n\n6\n00:00:17.600 --> 00:00:20.100\n Uh, I beg to differ with you on that.\n\n7\n00:00:20.100 --> 00:00:23.000\n When you snort cocaine, typically it goes into your nose.\n\n\n\n\n\nReturn:\n"""
    ]

    # result = llm._acall(prompt=question) # result in multi calls
    # result = llm._call(prompt=question) # result in multi calls

    


    

    # async def search_agent_demo():
    #     return simple_chatbot.run(question=question)

    # result = langchain_visualizer.visualize(search_agent_demo)
    for prompt in prompts:
        print("---")
        start_time = time()
        result = simple_chatbot.run(question=prompt)
        print(result)
        print(f"\n---{time()- start_time} seconds")
