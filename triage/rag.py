from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from triage.llm import get_embeddings


def load_pdf_to_chunks(file_path: str):
    """
    Loads a PDF file and splits it into chunks of text.
    """
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    return chunks

def chunks_to_vector_store(chunks, vector_db: Chroma, embeddings_model: get_embeddings):
    """
    Stores embeddings in a vector database.
    """
    from pathlib import Path
    here = Path(__file__).parent
    vector_db_path = here / "vector_store"
    vector_store = Chroma(collection_name="first_collection", persist_directory=str(vector_db_path), embedding_function=embeddings_model)
    vector_store.add_documents(chunks)

    return vector_store

def query_vector_retriever(vector_store: Chroma, query: str, embeddings_model: get_embeddings):
    """
    Queries the vector store for relevant documents based on the input query.
    """
    if not query:
        raise ValueError("Query cannot be empty.")
    query_embedding = embeddings_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=3)

    return results

