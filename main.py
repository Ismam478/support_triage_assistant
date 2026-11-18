from triage.rag import load_pdf_to_chunks, chunks_to_vector_store, query_vector_retriever
from langchain_chroma import Chroma
from triage.llm import get_embeddings
from pathlib import Path

here = Path(__file__).parent

FILE_PATH = here / "data" / "faq.pdf"

def main():
    
    chunks = load_pdf_to_chunks(FILE_PATH)
    embeddings_model=get_embeddings()
    vector_store = chunks_to_vector_store(chunks, vector_db=Chroma, embeddings_model=embeddings_model)
    results = query_vector_retriever(vector_store, query="What is the return policy?", embeddings_model=embeddings_model)

    for doc in results:
        print(f"Document: {doc.page_content}\n")



if __name__ == "__main__":
    main()
