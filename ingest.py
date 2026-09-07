"""Ingest PDFs from docs/ into the local EEG knowledge base.

Pipeline: PDF -> page text (PyMuPDF) -> overlapping chunks -> nomic-embed-text
embeddings (Ollama) -> ChromaDB. A JSON snapshot of all chunks is written so rag.py
can build a BM25 index for hybrid retrieval.

Idempotent: re-running re-indexes everything from scratch (cheap for a docs/ folder).
Safe to run on an empty docs/ — it just reports that there's nothing to ingest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

from rag import (BM25_CACHE, COLLECTION_NAME, PERSIST_DIR, get_client,
                 get_embedder, reset_caches)

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHUNK_SIZE = 1000      # characters
CHUNK_OVERLAP = 200
EMBED_BATCH = 32


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def load_pdf_pages(path: str):
    """Yield (page_number, text) for each non-empty page."""
    import fitz  # PyMuPDF

    doc = fitz.open(path)
    for pno in range(len(doc)):
        text = doc[pno].get_text("text").strip()
        if text:
            yield pno + 1, text
    doc.close()


def collect_chunks():
    """Walk docs/ and return parallel lists (ids, documents, metadatas)."""
    ids, documents, metadatas = [], [], []
    if not os.path.isdir(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        return ids, documents, metadatas

    # Walk recursively so PDFs in subfolders are found (a flat os.listdir silently ignored them).
    # `fname` is the path relative to DOCS_DIR so it stays unique and readable across subfolders.
    pdfs, txts = [], []
    for root, _dirs, files in os.walk(DOCS_DIR):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), DOCS_DIR)
            if f.lower().endswith(".pdf"):
                pdfs.append(rel)
            elif f.lower().endswith((".txt", ".md")):
                txts.append(rel)
    pdfs.sort()
    txts.sort()

    for fname in pdfs:
        path = os.path.join(DOCS_DIR, fname)
        try:
            for page, text in load_pdf_pages(path):
                for ci, chunk in enumerate(chunk_text(text)):
                    cid = hashlib.md5(f"{fname}:{page}:{ci}".encode()).hexdigest()
                    ids.append(cid)
                    documents.append(chunk)
                    metadatas.append({"source": fname, "page": page, "chunk": ci})
        except Exception as exc:
            print(f"  ! skipped {fname}: {exc}")

    for fname in txts:
        path = os.path.join(DOCS_DIR, fname)
        with open(path, "r", errors="ignore") as fh:
            text = fh.read()
        for ci, chunk in enumerate(chunk_text(text)):
            cid = hashlib.md5(f"{fname}:{ci}".encode()).hexdigest()
            ids.append(cid)
            documents.append(chunk)
            # Chroma rejects None metadata values, so omit page for text files.
            metadatas.append({"source": fname, "chunk": ci})

    return ids, documents, metadatas


def embed_batched(documents):
    embedder = get_embedder()
    vectors = []
    for i in range(0, len(documents), EMBED_BATCH):
        batch = documents[i:i + EMBED_BATCH]
        vectors.extend(embedder.embed_documents(batch))
        print(f"  embedded {min(i + EMBED_BATCH, len(documents))}/{len(documents)} chunks")
    return vectors


def main():
    os.makedirs(PERSIST_DIR, exist_ok=True)
    print(f"Scanning {DOCS_DIR} ...")
    ids, documents, metadatas = collect_chunks()

    client = get_client()
    # Fresh rebuild so removed PDFs don't linger.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    if not documents:
        # still write an empty BM25 snapshot so rag.py reads a consistent state
        with open(BM25_CACHE, "w") as fh:
            json.dump({"ids": [], "documents": [], "metadatas": []}, fh)
        reset_caches()
        print("No documents found in docs/. Knowledge base is empty but initialised.")
        print("Drop .pdf / .txt / .md files into docs/ and re-run `python ingest.py`.")
        return

    print(f"Found {len(documents)} chunks across "
          f"{len(set(m['source'] for m in metadatas))} file(s). Embedding ...")
    vectors = embed_batched(documents)

    print("Writing to ChromaDB ...")
    for i in range(0, len(documents), 256):
        collection.add(
            ids=ids[i:i + 256],
            embeddings=vectors[i:i + 256],
            documents=documents[i:i + 256],
            metadatas=metadatas[i:i + 256],
        )

    with open(BM25_CACHE, "w") as fh:
        json.dump({"ids": ids, "documents": documents, "metadatas": metadatas}, fh)
    reset_caches()

    print(f"Done. {collection.count()} chunks indexed in {PERSIST_DIR}")


if __name__ == "__main__":
    sys.exit(main())
