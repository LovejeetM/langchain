from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda
from functools import partial

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import ArxivLoader

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

# Output: 
"""
Number of Documents Retrieved: 1
Sample of Document 1 Content (Total Length: 89583):
From Local to Global: A GraphRAG Approach to
Query-Focused Summarization
Darren Edge1†
Ha Trinh1†
Newman Cheng2
Joshua Bradley2
Alex Chao3
Apurva Mody3
Steven Truitt2
Dasha Metropolitansky1
Robert Osazuwa Ness1
Jonathan Larson1
1Microsoft Research
2Microsoft Strategic Missions and Technologies
3Microsoft Office of the CTO
{daedge,trinhha,newmancheng,joshbradley,achao,moapurva,
steventruitt,dasham,robertness,jolarso}@microsoft.com
†These authors contributed equally to this work
Abstract
The use of retrieval-augmented generation (RAG) to retrieve relevant informa-
tion from an external knowledge source enables large language models (LLMs)
to answer questions over private and/or previously unseen document collections.
However, RAG fails on global questions directed at an entire text corpus, such
as “What are the main themes in the dataset?”, since this is inherently a query-
focused summarization (QFS) task, rather than an explicit retrieval task. Prior
QFS methods, meanwhile, do not scal
{
    'Published': '2025-02-19',
    'Title': 'From Local to Global: A Graph RAG Approach to Query-Focused Summarization',
    'Authors': 'Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert     
Osazuwa Ness, Jonathan Larson',
    'Summary': 'The use of retrieval-augmented generation (RAG) to retrieve relevant\ninformation from an external knowledge source enables   
large language models\n(LLMs) to answer questions over private and/or previously unseen document\ncollections. However, RAG fails on global   
questions directed at an entire text\ncorpus, such as "What are the main themes in the dataset?", since this is\ninherently a query-focused   
summarization (QFS) task, rather than an explicit\nretrieval task. Prior QFS methods, meanwhile, do not scale to the quantities of\ntext      
indexed by typical RAG systems. To combine the strengths of these\ncontrasting methods, we propose GraphRAG, a graph-based approach to        
question\nanswering over private text corpora that scales with both the generality of\nuser questions and the quantity of source text. Our    
approach uses an LLM to\nbuild a graph index in two stages: first, to derive an entity knowledge graph\nfrom the source documents, then to    
pregenerate community summaries for all\ngroups of closely related entities. Given a question, each community summary is\nused to generate a  
partial response, before all partial responses are again\nsummarized in a final response to the user. For a class of global
sensemaking\nquestions over datasets in the 1 million token range, we show that GraphRAG\nleads to substantial improvements over a
conventional RAG baseline for both the\ncomprehensiveness and diversity of generated answers.'
}  """





from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", ";", ",", " ", ""],
)

# documents[0].page_content = documents[0].page_content.replace(". .", "")
docs_split = text_splitter.split_documents(documents)

# def include_doc(doc):
#     ## Some chunks will be overburdened with useless numerical data, so we'll filter it out
#     string = doc.page_content
#     if len([l for l in string if l.isalpha()]) < (len(string)//2):
#         return False
#     return True

# docs_split = [doc for doc in docs_split if include_doc(doc)]
print(len(docs_split))

for i in (0, 1, 2, 15, -1):
    pprint(f"[Document {i}]")
    print(docs_split[i].page_content)
    pprint("="*64)
