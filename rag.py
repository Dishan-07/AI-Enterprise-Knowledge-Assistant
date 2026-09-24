import os
import re
from io import BytesIO
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import (
    EMBEDDING_MODEL,
    TOP_K,
    SIMILARITY_THRESHOLD,
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

_embedding_model = None


def get_embedding_model():
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    return _embedding_model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD PDF FROM DISK
# ============================================================

def load_pdf(
    file_path,
    department
):
    reader = PdfReader(file_path)

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        text = clean_text(text)

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": os.path.basename(
                    str(file_path)
                ),
                "page": page_number,
                "department": department,
            }
        )

    return documents


# ============================================================
# LOAD UPLOADED PDF
# ============================================================

def load_uploaded_pdf(
    file_bytes,
    filename,
    department
):
    reader = PdfReader(
        BytesIO(file_bytes)
    )

    documents = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text()

        text = clean_text(text)

        if not text:
            continue

        documents.append(
            {
                "text": text,
                "source": filename,
                "page": page_number,
                "department": department,
            }
        )

    return documents


# ============================================================
# LOAD ALL BASE DOCUMENTS
# ============================================================

def load_documents(
    base_folder="documents"
):
    all_documents = []

    departments = [
        "hr",
        "technical",
        "projects",
    ]

    for department in departments:

        folder = Path(
            base_folder
        ) / department

        if not folder.exists():
            continue

        for file_path in sorted(
            folder.glob("*.pdf")
        ):

            documents = load_pdf(
                file_path,
                department
            )

            all_documents.extend(
                documents
            )

    return all_documents


# ============================================================
# CREATE CHUNKS
# ============================================================

def create_chunks(
    documents,
    chunk_size=700,
    overlap=100
):
    chunks = []
    metadata = []

    step = max(
        1,
        chunk_size - overlap
    )

    for document in documents:

        text = document["text"]

        if len(text) <= chunk_size:

            ranges = [
                (
                    0,
                    len(text)
                )
            ]

        else:

            ranges = []

            start = 0

            while start < len(text):

                end = min(
                    start + chunk_size,
                    len(text)
                )

                ranges.append(
                    (
                        start,
                        end
                    )
                )

                if end >= len(text):
                    break

                start += step

        for start, end in ranges:

            chunk = text[
                start:end
            ].strip()

            if not chunk:
                continue

            chunks.append(chunk)

            metadata.append(
                {
                    "source":
                        document["source"],

                    "page":
                        document["page"],

                    "department":
                        document["department"],
                }
            )

    return (
        chunks,
        metadata
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(
    chunks
):
    if not chunks:
        raise ValueError(
            "No text chunks were created."
        )

    model = get_embedding_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = embeddings.astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(
    question,
    chunks,
    metadata,
    index,
    department=None,
    top_k=TOP_K,
    similarity_threshold=SIMILARITY_THRESHOLD,
):
    if (
        not chunks
        or index is None
    ):
        return []

    model = get_embedding_model()

    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    question_embedding = (
        question_embedding.astype(
            "float32"
        )
    )

    search_k = min(
        max(
            top_k * 4,
            top_k
        ),
        len(chunks)
    )

    scores, indices = index.search(
        question_embedding,
        search_k
    )

    results = []

    seen_sources = set()

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        index_id = int(
            index_id
        )

        item_metadata = metadata[
            index_id
        ]

        if (
            department
            and item_metadata[
                "department"
            ] != department
        ):
            continue

        score = float(score)

        if score < similarity_threshold:
            continue

        source_key = (
            item_metadata["source"],
            item_metadata["page"],
            item_metadata["department"],
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(
            source_key
        )

        results.append(
            {
                "text":
                    chunks[index_id],

                "metadata":
                    item_metadata,

                "similarity":
                    score,
            }
        )

        if len(results) >= top_k:
            break

    return results


# ============================================================
# VECTOR STORE
# ============================================================

class VectorStore:

    def __init__(
        self,
        chunks,
        metadata
    ):
        self.chunks = chunks
        self.metadata = metadata

        self.index = build_index(
            chunks
        )

    def search(
        self,
        question,
        department=None,
        top_k=TOP_K
    ):
        return retrieve(
            question,
            self.chunks,
            self.metadata,
            self.index,
            department=department,
            top_k=top_k,
        )