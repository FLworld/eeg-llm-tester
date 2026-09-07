"""Hybrid retrieval over the local EEG knowledge base.

Combines semantic search (ChromaDB + nomic-embed-text via Ollama) with lexical
search (BM25 over the same chunks) using Reciprocal Rank Fusion (RRF). Everything
is local; no API keys.

Shared constants are imported by ingest.py so the two stay in lockstep.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import chromadb
from langchain_ollama import OllamaEmbeddings
from rank_bm25 import BM25Okapi

# --------------------------------------------------------------------------- #
# Config (shared with ingest.py)
# --------------------------------------------------------------------------- #
# Honors STATE_DIR (a mounted volume in a container) so an ingested knowledge base survives
# restarts; defaults to the module dir for a plain local checkout.
PERSIST_DIR = os.path.join(
    os.environ.get("STATE_DIR", os.path.dirname(__file__)), "eeg_knowledge")
COLLECTION_NAME = "eeg_docs"
EMBED_MODEL = "nomic-embed-text"
BM25_CACHE = os.path.join(PERSIST_DIR, "bm25_corpus.json")
RRF_K = 60  # reciprocal-rank-fusion damping constant


@lru_cache(maxsize=1)
def get_embedder() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=EMBED_MODEL)


@lru_cache(maxsize=1)
def get_client() -> "chromadb.api.ClientAPI":
    return chromadb.PersistentClient(path=PERSIST_DIR)


def get_collection():
    """Return the Chroma collection, creating it if needed."""
    return get_client().get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(
        c.lower() if c.isalnum() else " " for c in text
    ).split() if len(t) > 1]


@lru_cache(maxsize=1)
def _load_bm25():
    """Build a BM25 index from the persisted corpus snapshot."""
    if not os.path.exists(BM25_CACHE):
        return None, [], []
    with open(BM25_CACHE, "r") as fh:
        corpus = json.load(fh)  # {"ids": [...], "documents": [...], "metadatas": [...]}
    docs = corpus["documents"]
    if not docs:
        return None, [], []
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    return bm25, corpus["ids"], corpus


def retrieve_context(query: str, k: int = 5) -> dict:
    """Hybrid retrieve. Returns {"text": str, "sources": [...], "chunks": [...]}.

    Empty knowledge base -> empty context (the app falls back to the model's own
    EEG expertise).
    """
    collection = get_collection()
    try:
        count = collection.count()
    except Exception:
        count = 0
    if count == 0:
        return {"text": "", "sources": [], "chunks": []}

    pool = min(max(k * 4, 10), count)

    # --- semantic ranking ---
    q_emb = get_embedder().embed_query(query)
    sem = collection.query(query_embeddings=[q_emb], n_results=pool)
    sem_ids = sem["ids"][0]
    sem_docs = {i: d for i, d in zip(sem_ids, sem["documents"][0])}
    sem_meta = {i: m for i, m in zip(sem_ids, sem["metadatas"][0])}

    # --- lexical ranking ---
    bm25, bm25_ids, corpus = _load_bm25()
    lex_ids: list[str] = []
    if bm25 is not None:
        scores = bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        lex_ids = [bm25_ids[i] for i in order[:pool]]
        for i in order[:pool]:
            cid = bm25_ids[i]
            sem_docs.setdefault(cid, corpus["documents"][i])
            sem_meta.setdefault(cid, corpus["metadatas"][i])

    # --- reciprocal rank fusion ---
    fused: dict[str, float] = {}
    for rank, cid in enumerate(sem_ids):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)
    for rank, cid in enumerate(lex_ids):
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

    top = sorted(fused, key=fused.get, reverse=True)[:k]

    chunks, sources, blocks = [], [], []
    for cid in top:
        doc = sem_docs.get(cid, "")
        meta = sem_meta.get(cid, {})
        src = meta.get("source", "unknown")
        page = meta.get("page")
        tag = f"{src}" + (f" p.{page}" if page is not None else "")
        if src not in sources:
            sources.append(src)
        chunks.append({"id": cid, "source": tag, "text": doc, "score": fused[cid]})
        blocks.append(f"[{tag}]\n{doc}")

    return {"text": "\n\n---\n\n".join(blocks), "sources": sources, "chunks": chunks}


def reset_caches():
    """Drop memoised BM25/embedder/client state (call after re-ingesting)."""
    _load_bm25.cache_clear()
    get_client.cache_clear()
    get_embedder.cache_clear()


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "What is the alpha rhythm and where is it strongest?"
    res = retrieve_context(q)
    print(f"Query: {q}\nSources: {res['sources']}\n")
    print(res["text"][:1500] or "(knowledge base empty — drop PDFs in docs/ and run ingest.py)")
