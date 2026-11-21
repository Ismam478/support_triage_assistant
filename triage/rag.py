from langchain_text_splitters import MarkdownHeaderTextSplitter
import pymupdf4llm
from langchain_chroma import Chroma
from triage.llm import get_embeddings
from pathlib import Path

here = Path(__file__).parent
DATA_FILE_PATH = here.parent / "data" / "faq.pdf"
VECTOR_STORE_PATH = here.parent / "vector_store"

def load_pdf_to_chunks(file_path: str):
    """
    Loads a PDF file and splits it into chunks of text.
    """
    documents = pymupdf4llm.to_markdown(file_path)

    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Title"), ("##", "Topic"), ("###", "Subtopic")], strip_headers=False)
    chunks = splitter.split_text(documents)

    return chunks

def get_faq_vector_store():
    """
    Retrieves the vector store for the FAQ.
    """

    embeddings_model = get_embeddings()
    if not VECTOR_STORE_PATH.exists():
        chunks = load_pdf_to_chunks(DATA_FILE_PATH)
        vector_store = Chroma(collection_name="first_collection", persist_directory=str(VECTOR_STORE_PATH), embedding_function=embeddings_model)
        vector_store.add_documents(chunks)
    else:
        vector_store = Chroma(collection_name="first_collection", persist_directory=str(VECTOR_STORE_PATH), embedding_function=embeddings_model)

    return vector_store, embeddings_model

def query_vector_retriever(vector_store: Chroma, query: str, embeddings_model: get_embeddings, k: int = 3):
    """
    Queries the vector store for relevant documents based on the input query.
    """
    if not query:
        raise ValueError("Query cannot be empty.")
    query_embedding = embeddings_model.embed_query(query)
    results = vector_store.similarity_search_by_vector(embedding=query_embedding, k=k)

    return results