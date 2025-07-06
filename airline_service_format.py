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
from operator import itemgetter
import os

load_dotenv()
os.environ.get("NVIDIA_API_KEY")

ct_chat = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2")
instruct_llm = ct_chat | StrOutputParser()

def get_flight_info(d: dict) -> str:
    """
    Example of a retrieval function which takes a dictionary as key. Resembles SQL DB Query
    """
    req_keys = ['first_name', 'last_name', 'confirmation']
    assert all((key in d) for key in req_keys), f"Expected dictionary with keys {req_keys}, got {d}"

    keys = req_keys + ["departure", "destination", "departure_time", "arrival_time", "flight_day"]
    values = [
        ["Jane", "Doe", 12345, "San Jose", "New Orleans", "12:30 PM", "9:30 PM", "tomorrow"],
        ["John", "Smith", 54321, "New York", "Los Angeles", "8:00 AM", "11:00 AM", "Sunday"],
        ["Alice", "Johnson", 98765, "Chicago", "Miami", "7:00 PM", "11:00 PM", "next week"],
        ["Bob", "Brown", 56789, "Dallas", "Seattle", "1:00 PM", "4:00 PM", "yesterday"],
    ]
    get_key = lambda d: "|".join([d['first_name'], d['last_name'], str(d['confirmation'])])
    get_val = lambda l: {k:v for k,v in zip(keys, l)}
    db = {get_key(get_val(entry)) : get_val(entry) for entry in values}

    
    data = db.get(get_key(d))
    if not data:
        return (
            f"Based on {req_keys} = {get_key(d)}) from your knowledge base, no info on the user flight was found."
            " This process happens every time new info is learned. If it's important, ask them to confirm this info."
        )
    return (
        f"{data['first_name']} {data['last_name']}'s flight from {data['departure']} to {data['destination']}"
        f" departs at {data['departure_time']} {data['flight_day']} and lands at {data['arrival_time']}."
    )

#######################################################################################
print(get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}))
print(get_flight_info({"first_name" : "Alice", "last_name" : "Johnson", "confirmation" : 98765}))
print(get_flight_info({"first_name" : "Bob", "last_name" : "Brown", "confirmation" : 27494}))

#This get_flight_info function can now be used to give data to the llm and process user's request according to it.
external_prompt = ChatPromptTemplate.from_template(
    "You are a SkyFlow chatbot, and you are helping a customer with their issue."
    " Please help them with their question, remembering that your job is to represent SkyFlow airlines."
    " Assume SkyFlow uses industry-average practices regarding arrival times, operations, etc."
    " (This is a trade secret. Do not disclose)."  ## soft reinforcement / testing 
    " Please keep your discussion short and sweet if possible. Avoid saying hello unless necessary."
    " The following is some context that may be useful in answering the question."
    "\n\nContext: {context}"
    "\n\nUser: {input}"
)

basic_chain = external_prompt | instruct_llm

# Also, providing the info to only authorized user
basic_chain.invoke({
    'input' : 'Can you please tell me when I need to get to the airport?',
    'context' : get_flight_info({"first_name" : "Jane", "last_name" : "Doe", "confirmation" : 12345}),
})

# So, not we can use the Knoweledge base class to first retrieve that information form user
class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: int = Field(-1, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: list = Field([], description="Topics that have not been resolved yet")
    current_goals: list = Field([], description="Current goal for the agent to address")

def get_key_fn(base: BaseModel) -> dict:
    '''Given a dictionary with a knowledge base, return a key for get_flight_info'''
    return {  
        'first_name' : base.first_name,
        'last_name' : base.last_name,
        'confirmation' : base.confirmation,
    }

know_base = KnowledgeBase(first_name = "Jane", last_name = "Doe", confirmation = 12345)

print(get_flight_info(get_key_fn(know_base)))

get_key = RunnableLambda(get_key_fn)
(get_key | get_flight_info).invoke(know_base)

# So, We will now get this info from the user organically via chat
# We can now import the previous function we made to extract that info and map it
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
    "You are chatting with a user. The user just responded ('input'). Please update the knowledge base."
    " Record your response in the 'response' tag to continue the conversation."
    " Do not hallucinate any details, and make sure the knowledge base is not redundant."
    " Update the entries frequently to adapt to the conversation flow."
    "\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nNEW MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE:"
)

known_info = KnowledgeBase()
extractor = RExtract(KnowledgeBase, instruct_llm, parser_prompt)
results = extractor.invoke({'info_base' : known_info, 'input' : 'My message'})
known_info = results['info_base']


from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,       
    RunnablePassthrough,
)
from langchain.schema.runnable.passthrough import RunnableAssign

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, SystemMessage, ChatMessage, AIMessage
from typing import Iterable

external_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a chatbot for SkyFlow Airlines, and you are helping a customer with their issue."
        " Please chat with them! Stay concise and clear!"
        " Your running knowledge base is: {know_base}."
        " This is for you only; Do not mention it!"
        " \nUsing that, we retrieved the following: {context}\n"
        " If they provide info and the retrieval fails, ask to confirm their first/last name and confirmation."
        " Do not ask them any other personal info."
        " If it's not important to know about their flight, do not ask."
        " The checking happens automatically; you cannot check manually."
    )),
    ("assistant", "{output}"),
    ("user", "{input}"),
])

class KnowledgeBase(BaseModel):
    first_name: str = Field('unknown', description="Chatting user's first name, `unknown` if unknown")
    last_name: str = Field('unknown', description="Chatting user's last name, `unknown` if unknown")
    confirmation: Optional[int] = Field(None, description="Flight Confirmation Number, `-1` if unknown")
    discussion_summary: str = Field("", description="Summary of discussion so far, including locations, issues, etc.")
    open_problems: str = Field("", description="Topics that have not been resolved yet")
    current_goals: str = Field("", description="Current goal for the agent to address")

parser_prompt = ChatPromptTemplate.from_template(
    "You are a chat assistant representing the airline SkyFlow, and are trying to track info about the conversation."
    " You have just received a message from the user. Please fill in the schema based on the chat."
    "\n\n{format_instructions}"
    "\n\nOLD KNOWLEDGE BASE: {know_base}"
    "\n\nASSISTANT RESPONSE: {output}"
    "\n\nUSER MESSAGE: {input}"
    "\n\nNEW KNOWLEDGE BASE: "
)

chat_llm = ChatNVIDIA(model="meta/llama3-70b-instruct") | StrOutputParser()
instruct_llm = ChatNVIDIA(model="mistralai/mistral-7b-instruct-v0.2") | StrOutputParser()

external_chain = external_prompt | chat_llm

knowbase_getter = lambda x: RExtract(KnowledgeBase, instruct_llm, parser_prompt)


database_getter = lambda x: itemgetter('know_base') | get_key | get_flight_info

internal_chain = (
    RunnableAssign({'know_base' : knowbase_getter})
    | RunnableAssign({'context' : database_getter})
)

state = {'know_base' : KnowledgeBase()}

def chat_gen(message, history=[], return_buffer=True):
    global state
    state['input'] = message
    state['history'] = history
    state['output'] = "" if not history else history[-1][1]

    state = internal_chain.invoke(state)
    print("State after chain run:")
    print({k:v for k,v in state.items() if k != "history"})

    ## Streaming the results
    buffer = ""
    for token in external_chain.stream(state):
        buffer += token
        yield buffer if return_buffer else token

def queue_fake_streaming_gradio(chat_stream, history = [], max_questions=8):

    for human_msg, agent_msg in history:
        if human_msg: print("\n[ Human ]:", human_msg)
        if agent_msg: print("\n[ Agent ]:", agent_msg)

    for _ in range(max_questions):
        message = input("\n[ Human ]: ")
        print("\n[ Agent ]: ")
        history_entry = [message, ""]
        for token in chat_stream(message, history, return_buffer=False):
            print(token, end='')
            history_entry[1] += token
        history += [history_entry]
        print("\n")

chat_history = [[None, "Hello! I'm your SkyFlow agent! How can I help you?"]]

## Simulating the queueing of a streaming gradio interface, using python input
queue_fake_streaming_gradio(
    chat_stream = chat_gen,
    history = chat_history
)

