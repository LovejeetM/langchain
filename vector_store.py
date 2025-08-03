from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_transformers import LongContextReorder
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain.schema.runnable.passthrough import RunnableAssign
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter

from functools import partial
from operator import itemgetter

# NVIDIAEmbeddings.get_available_models()
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1", truncate="END")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")


conversation = [  
    "[User]  Hello! My name is Beras, and I'm a big blue bear! Can you please tell me about the rocky mountains?",
    "[Agent] The Rocky Mountains are a beautiful and majestic range of mountains that stretch across North America",
    "[Beras] Wow, that sounds amazing! Ive never been to the Rocky Mountains before, but Ive heard many great things about them.",
    "[Agent] I hope you get to visit them someday, Beras! It would be a great adventure for you!"
    "[Beras] Thank you for the suggestion! Ill definitely keep it in mind for the future.",
    "[Agent] In the meantime, you can learn more about the Rocky Mountains by doing some research online or watching documentaries about them."
    "[Beras] I live in the arctic, so I'm not used to the warm climate there. I was just curious, ya know!",
    "[Agent] Absolutely! Lets continue the conversation and explore more about the Rocky Mountains and their significance!"
]

convstore = FAISS.from_texts(conversation, embedding=embedder)
retriever = convstore.as_retriever()

print(retriever.invoke("What is your name?"))

print(retriever.invoke("Where are the Rocky Mountains?"))


def RPrint(preface=""):
    """Simple passthrough "prints, then returns" chain"""
    def print_and_return(x, preface):
        if preface: print(preface, end="")
        print(x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def docs2str(docs, title="Document"):
    """Useful utility for making chunks into context string. Optional, but useful"""
    out_str = ""
    for doc in docs:
        doc_name = getattr(doc, 'metadata', {}).get('Title', title)
        if doc_name:
            out_str += f"[Quote from {doc_name}] "
        out_str += getattr(doc, 'page_content', str(doc)) + "\n"
    return out_str

long_reorder = RunnableLambda(LongContextReorder().transform_documents)


context_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {question}"
    "\nAnswer the user conversationally. User is not aware of context."
)

chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'question': (lambda x:x)
    }
    | context_prompt
    # | RPrint()
    | instruct_llm
    | StrOutputParser()
)

print(chain.invoke("Where does Beras live?"))

print(chain.invoke("Where are the Rocky Mountains?"))

print(chain.invoke("Where are the Rocky Mountains? Are they close to California?"))


convstore = FAISS.from_texts(conversation, embedding=embedder)

def save_memory_and_get_output(d, vstore):
    """Accepts 'input'/'output' dictionary and saves to convstore"""
    vstore.add_texts([f"User said {d.get('input')}", f"Agent said {d.get('output')}"])
    return d.get('output')

########################################################################

# instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x22b-instruct-v0.1")

chat_prompt = ChatPromptTemplate.from_template(
    "Answer the question using only the context"
    "\n\nRetrieved Context: {context}"
    "\n\nUser Question: {input}"
    "\nAnswer the user conversationally. Make sure the conversation flows naturally.\n"
    "[Agent]"
)


conv_chain = (
    {
        'context': convstore.as_retriever() | long_reorder | docs2str,
        'input': (lambda x:x)
    }
    | RunnableAssign({'output' : chat_prompt | instruct_llm | StrOutputParser()})
    | partial(save_memory_and_get_output, vstore=convstore)
)

print(conv_chain.invoke("I'm glad you agree! I can't wait to get some ice cream there! It's such a good food!"))
print()
print(conv_chain.invoke("Can you guess what my favorite food is?"))
print()
print(conv_chain.invoke("Actually, my favorite is honey! Not sure where you got that idea?"))
print()
print(conv_chain.invoke("I see! Fair enough! Do you know my favorite food now?"))


