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


def get_flight_info(d: dict) -> str:
    """
    Example of a retrieval function which takes a dictionary as key. Resembles SQL DB Query
    """
    req_keys= ["first_name", "last_name", "confirmation"]