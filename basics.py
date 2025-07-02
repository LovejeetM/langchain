from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from functools import partial

identity = RunnableLambda(lambda x: f"{x[0]} {x[1]}")

def print_and_show(x, prefix = ""):
    print(f"{prefix}{x}")
    return x

rprint0 = RunnableLambda(print_and_show)

rprint1 = RunnableLambda(partial(print_and_show, prefix= "1. :"))

def rprint2(prefix):
    return RunnableLambda(partial(print_and_show, prefix = prefix))

chain =  identity | rprint0 | rprint1 | rprint2("2 : ")

print(identity.invoke(("Hello", "Name")))

print(chain.invoke(("Hello", "Name")))



