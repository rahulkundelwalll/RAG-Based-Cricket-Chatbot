from chromaDBRetriber import ChromaDBRetriever
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from Split_data_in_chunk import SplitDataInChunkSaveChromaDB
# from langchain_ollama import OllamaLLM

load_dotenv()

class RAG_CHATBOT:
    def __init__(self):
        self.model = Ollama(model="llama3")  # Using the Ollama model (LLaMA 3)

    def chatbot(self, query):
        chromaRe = ChromaDBRetriever()
        relevant_docs = chromaRe.retrieve_documents(query)
        
        combined_input = (
            query
            + "\n\nRelevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\nPlease provide a rough answer with metadata based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
        )
        
        # Define the messages for the model
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content=combined_input),
        ]
        
        # Invoke the model with the combined input
        result = self.model.invoke(messages)
        
        return result

# Example usage
if __name__ == "__main__":
    query = "How did Mohammed Shami and Kuldeep Yadav perform in their comeback match against England?"
    a = SplitDataInChunkSaveChromaDB()
    a.save_chunks_into_chroma_db()
    rag = RAG_CHATBOT()
    print(rag.chatbot(query))
