from pydantic import BaseModel, Field
from typing import Dict, Union, Optional

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable.passthrough import RunnableAssign
from functools import partial
from dotenv import load_dotenv
import os

load_dotenv()
os.environ.get("NVIDIA_API_KEY")

ct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")
instruct_llm = ct_chat | StrOutputParser()

class KnowledgeBase(BaseModel):
    ## Fields of the BaseModel, which will be validated/assigned when the knowledge base is constructed
    topic: str = Field('general', description="Current conversation topic")
    user_preferences: Dict[str, Union[str, int]] = Field({}, description="User preferences and choices")
    session_notes: list = Field([], description="Notes on the ongoing session")
    unresolved_queries: list = Field([], description="Unresolved user queries")
    action_items: list = Field([], description="Actionable items identified during the conversation")

print(repr(KnowledgeBase(topic = "Travel")))


instruct_string = PydanticOutputParser(pydantic_object=KnowledgeBase).get_format_instructions()
print(instruct_string)


def RExtract(pydantic_class, llm, prompt):
    '''
    Runnable Extraction module
    Returns a knowledge dictionary populated by slot-filling extraction
    '''
    parser = PydanticOutputParser(pydantic_object=pydantic_class)
    instruct_merge = RunnableAssign({'format_instructions' : lambda x: parser.get_format_instructions()})
    def preparse(string):
        if '{' not in string: string = '{' + string
        if '}' not in string: string = string + '}'
        string = (string
            .replace("\\_", "_")
            .replace("\n", " ")
            .replace("\]", "]")
            .replace("\[", "[")
        )
        # print(string)  ## Good for diagnostics
        return string
    return instruct_merge | prompt | llm | preparse | parser


parser_prompt = ChatPromptTemplate.from_template(
    "Update the knowledge base: {format_instructions}. Only use information from the input."
    "\n\nNEW MESSAGE: {input}"
)

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)

knowledge = extractor.invoke({'input' : "I love burgers so much! The big mac's are amazing! Can you buy me some?"})
print(knowledge)



class KnowledgeBase(BaseModel):
    firstname: str = Field('unknown', description="Chatting user's first name, unknown if unknown")
    lastname: str = Field('unknown', description="Chatting user's last name, unknown if unknown")
    location: str = Field('unknown', description="Where the user is located")
    summary: str = Field('unknown', description="Running summary of conversation. Update this with new input")
    response: str = Field('unknown', description="An ideal response to the user based on their new message")


parser_prompt = ChatPromptTemplate.from_template(
    "You are chatting with a user. The user just responded ('input'). Please update the knowledge base."
    " Record your response in the 'response' tag to continue the conversation."
    " Do not hallucinate any details, and make sure the knowledge base is not redundant."
    " Update the entries frequently to adapt to the conversation flow."
    "\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nNEW MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE:"
)

## Switch to a more powerful base model
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1") | StrOutputParser()

extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
info_update = RunnableAssign({'know_base' : extractor})

## Initialize the knowledge base and see what you get
state = {'know_base' : KnowledgeBase()}
state['input'] = "My name is Carmen Sandiego! Guess where I am! Hint: It's somewhere in the United States."
state = info_update.invoke(state)
print(state)