import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings

load_dotenv()
os.environ.get("NVIDIA_API_KEY")


[m for m in NVIDIAEmbeddings.get_available_models() if "embed" in m.id]

# NVIDIAEmbeddings.get_available_models()
# embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-mistral-7b-v2")
# embedder = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
# embedder = NVIDIAEmbeddings(model="nvidia/embed-qa-4")
# embedder = NVIDIAEmbeddings(model="snowflake/arctic-embed-l")
embedder = NVIDIAEmbeddings(model="nvidia/nv-embed-v1")

# ChatNVIDIA.get_available_models()
instruct_llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
