from Split_data_in_chunk import SplitDataInChunkSaveChromaDB
from Rag import RAG_CHABOT

if __name__ == "__main__":
    query = "How did Mohammed Shami and Kuldeep Yadav perform in their comeback match against England?"
    a = SplitDataInChunkSaveChromaDB()
    a.save_chunks_into_chroma_db()
    rag = RAG_CHABOT()
    print(rag.chatbot(query))