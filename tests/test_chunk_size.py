from triage.rag import load_pdf_to_chunks, DATA_FILE_PATH

def test_faq_has_one_chunk_per_entry():
    assert len(load_pdf_to_chunks(DATA_FILE_PATH)) == 14