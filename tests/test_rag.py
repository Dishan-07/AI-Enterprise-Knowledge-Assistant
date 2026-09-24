from pathlib import Path

from pypdf import PdfReader

from rag import create_chunks, load_documents

def test_documents_exist():
    documents = load_documents("documents")
    assert documents
    assert len(documents) >= 6

def test_chunks_are_created():
    documents = load_documents("documents")
    chunks, metadata = create_chunks(documents)
    assert chunks
    assert len(chunks) == len(metadata)

def test_demo_pdfs_are_readable():
    paths = list(Path("documents").rglob("*.pdf"))
    assert len(paths) >= 6
    for path in paths:
        assert len(PdfReader(path).pages) >= 1
