import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def get_relevant_context(query: str, db_path: str = "./data/chroma_db") -> str:
    """Retrieves context layers cleanly via matching local HuggingFace embedding frameworks."""
    if not os.path.exists(db_path):
        return "No database found. Please ingest campus documents first."
        
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=db_path, embedding_function=embeddings)
    
    docs = vector_store.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])