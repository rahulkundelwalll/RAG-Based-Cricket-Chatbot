from Split_data_in_chunk import SplitDataInChunkSaveChromaDB
from Rag import RAG_CHABOT

rag = RAG_CHABOT()
response = rag.chatbot("What is the capital of France?")
print(response)