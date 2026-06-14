import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def run_ingestion(data_dir: str = "./data/handbooks", persist_dir: str = "./data/chroma_db"):
    """Loads campus PDFs, chunks them, and indexes them into ChromaDB using local HuggingFace embeddings."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"📁 Created missing data directory at '{data_dir}'. Drop your campus PDFs there!")
        return
        
    print(f"🔄 Initializing ingestion from target directory: {data_dir}")
    loader = PyPDFDirectoryLoader(data_dir)
    raw_documents = loader.load()
    
    if not raw_documents:
        print("⚠️ No PDFs found in './data/handbooks/'. Add documents and re-run ingestion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)
    
    # Using a high-efficiency production local embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print(f"🧱 Vectorizing {len(documents)} document chunks locally into {persist_dir}...")
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("✅ Local Knowledge Base successfully compiled.")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    run_ingestion()