import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_relevant_context(query: str) -> str:
    """Connects safely to the pre-compiled local ChromaDB sqlite file and returns match blocks."""
    db_path = os.path.join(os.getcwd(), "data", "chroma_db")
    
    if not os.path.exists(db_path):
        return "System Warning: Local knowledge base file index path missing. Fallback to base configuration data."
        
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings
    )
    
    # Extract the top 3 most statistically relevant text data context strings
    docs = db.similarity_search(query, k=3)
    
    if not docs:
        return "No specific administrative policy mapping found inside the campus handbook documentation."
        
    return "\n\n".join([doc.page_content for doc in docs])