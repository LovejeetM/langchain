from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

def print_and_return(x, preface=""):
    print(f"{preface}{x}")
    return x

################################################################################
## Example of dictionary enforcement methods
def make_dictionary(v, key):
    if isinstance(v, dict):
        return v
    return {key : v}

def RInput(key='input'):
    '''Coercing method to mold a value (i.e. string) to in-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def ROutput(key='output'):
    '''Coercing method to mold a value (i.e. string) to out-like dict'''
    return RunnableLambda(partial(make_dictionary, key=key))

def RPrint(preface=""):
    return RunnableLambda(partial(print_and_return, preface=preface))

################################################################################
## Common LCEL utility for pulling values from dictionaries
from operator import itemgetter

up_and_down = (
    RPrint("A: ")
    ## Custom ensure-dictionary process
    | RInput()
    | RPrint("B: ")
    ## Pull-values-from-dictionary utility
    | itemgetter("input")
    | RPrint("C: ")
    ## Anything-in Dictionary-out implicit map
    | {
        'word1' : (lambda x : x.split()[0]),
        'word2' : (lambda x : x.split()[1]),
        'words' : (lambda x: x),  ## <- == to RunnablePassthrough()
    }
    | RPrint("D: ")
    | itemgetter("word1")
    | RPrint("E: ")
    ## Anything-in anything-out lambda application
    | RunnableLambda(lambda x: x.upper())
    | RPrint("F: ")
    ## Custom ensure-dictionary process
    | ROutput()
)

up_and_down.invoke({"input" : "Hello World"})