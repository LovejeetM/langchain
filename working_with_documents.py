from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()
from langchain_core.runnables import RunnableLambda
from functools import partial
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.passthrough import RunnableAssign
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from pydantic import BaseModel, Field
from typing import List
from IPython.display import clear_output


console = Console()
base_style = Style(color="#76B900", bold=True)
pprint = partial(console.print, style=base_style)

def RPrint(preface="State: "):
    def print_and_return(x, preface=""):
        print(f"{preface}{x}")
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

def PPrint(preface="State: "):
    def print_and_return(x, preface=""):
        pprint(preface, x)
        return x
    return RunnableLambda(partial(print_and_return, preface=preface))

documents = ArxivLoader(query="2404.16130").load() 

print("Number of Documents Retrieved:", len(documents))
print(f"Sample of Document 1 Content (Total Length: {len(documents[0].page_content)}):")
print(documents[0].page_content[:1000])

pprint(documents[0].metadata)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

## Some nice custom preprocessing
# documents[0].page_content = documents[0].page_content.replace(". .", "")
docs_split = text_splitter.split_documents(documents)

print(len(docs_split))

for i in (0, 1, 2, 15, -1):
    pprint(f"[Document {i}]")
    print(docs_split[i].page_content)
    pprint("="*64)


class DocumentSummaryBase(BaseModel):
    running_summary: str = Field("", description="Running description of the document. Do not override; only update!")
    main_ideas: List[str] = Field([], description="Most important information from the document (max 3)")
    loose_ends: List[str] = Field([], description="Open questions that would be good to incorporate into summary, but that are yet unknown (max 3)")


summary_prompt = ChatPromptTemplate.from_template(
    "You are generating a running summary of the document. Make it readable by a technical user."
    " After this, the old knowledge base will be replaced by the new one. Make sure a reader can still understand everything."
    " Keep it short, but as dense and useful as possible! The information should flow from chunk to (loose ends or main ideas) to running_summary."
    " The updated knowledge base keep all of the information from running_summary here: {info_base}."
    "\n\n{format_instructions}. Follow the format precisely, including quotations and commas"
    "\n\nWithout losing any of the info, update the knowledge base with the following: {input}"
)


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
        # print(string)  
        return string
    return instruct_merge | prompt | llm | preparse | parser


latest_summary = ""


def RSummarizer(knowledge, llm, prompt, verbose=False):
    '''
    Exercise: Create a chain that summarizes
    '''
    def summarize_docs(docs):        
        parse_chain = RunnableAssign({'info_base' : RExtract(knowledge.__class__, llm, prompt)})
        state = {'info_base' : knowledge}

        global latest_summary  
        
        for i, doc in enumerate(docs):
            state['input'] = doc.page_content
            state = parse_chain.invoke(state)

            assert 'info_base' in state 
            if verbose:
                print(f"Considered {i+1} documents")
                pprint(state['info_base'])
                latest_summary = state['info_base']
                clear_output(wait=True)

        return state['info_base']


pprint(latest_summary)
