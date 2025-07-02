from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
load_dotenv()
os.environ.get("NVIDIA_API_KEY")


chatllm = ChatNVIDIA(model = "meta/llama3-8b-instruct")

prompt  = ChatPromptTemplate.from_messages([
        ("system", "ONLY RESPOND DUMB"),
        ("user", "{input}")
    ])

chain = prompt | chatllm | StrOutputParser()

print(chain.invoke({"input": "Hey tell me something about moon"}))





import gradio as gr

#######################################################
## Non-streaming Interface 
# def rhyme_chat(message, history):
#     return rhyme_chain.invoke({"input" : message})

# gr.ChatInterface(rhyme_chat).launch()

#######################################################
## Streaming Interface

def rhyme_chat_stream(message, history):
    ## This is a generator function, where each call will yield the next entry
    buffer = ""
    for token in chain.stream({"input" : message}):
        buffer += token
        yield buffer

## Uncomment when you're ready to try this.
demo = gr.ChatInterface(rhyme_chat_stream).queue()
window_kwargs = {} # or {"server_name": "0.0.0.0", "root_path": "/7860/"}
demo.launch(share=True, debug=True, **window_kwargs)
