import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv

load_dotenv()

class SplitDataInChunkSaveChromaDB:
    def __init__(self, txt_file_path="BBC_Sport_Output/data.txt", embedding_model="all-MiniLM-L6-v2", output_dir="db"):
        self.txt_file_path = txt_file_path
        self.embedding_model = embedding_model
        self.output_dir = output_dir

    def load_text_convert_into_chunks(self):
        """Loads text and splits it into chunks"""
        if os.path.exists(self.output_dir):
            print("\n--- Vector store already exists. Skipping chunking. ---")
            return None

        print("\n--- Processing document and creating chunks ---")
        loader = TextLoader(self.txt_file_path, encoding="utf-8")
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(document)

        print(f"\n--- Number of document chunks: {len(docs)} ---")
        print(f"Sample chunk:\n{docs[0].page_content}\n")
        return docs

    def save_chunks_into_chroma_db(self):
        """Creates embeddings and stores chunks in ChromaDB"""
        if os.path.exists(self.output_dir):
            print("\n--- Vector store already exists. Skipping embedding creation. ---")
            return Chroma(persist_directory=self.output_dir, embedding_function=SentenceTransformerEmbeddings(model_name=self.embedding_model))

        print("\n--- Creating embeddings ---")
        embeddings = SentenceTransformerEmbeddings(model_name=self.embedding_model)

        docs = self.load_text_convert_into_chunks()
        if not docs:
            return None  # If docs is None, return early

        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=self.output_dir)
        print("\n--- Finished creating vector store ---")
        return db

if __name__ == "__main__":
    splitter = SplitDataInChunkSaveChromaDB()
    splitter.save_chunks_into_chroma_db()
