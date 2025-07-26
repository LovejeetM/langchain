from functools import partial
from rich.console import Console
from rich.style import Style
from rich.theme import Theme

from langchain_nvidia_ai_endpoints import ChatNVIDIA
ChatNVIDIA.get_available_models()
from langchain_core.runnables import RunnableLambda
from functools import partial

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

