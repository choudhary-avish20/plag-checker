"""
Plagiarism Checker v2 - Ingestion Pipeline
Reads arxiv (JSONL) + wikipedia (JSON array) data, chunks text,
embeds with sentence-transformers, and stores in SQLite + FAISS.

Run a small test first before committing to the full dataset.
"""

import json
import sqlite3
import re
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import nltk

# CONFIG 
ARXIV_PATH = "arxiv-metadata-oai-snapshot.json"
WIKI_PATH = "wiki_cleaned.json"

DB_PATH = "plagiarism.db"
FAISS_PATH = "index.faiss"

MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384
BATCH_SIZE = 128

# Wiki chunking
WIKI_WINDOW = 3
WIKI_STRIDE = 2

# process full file
ARXIV_LIMIT = 500
WIKI_LIMIT = 500

# Make sure nltk's sentence tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from nltk.tokenize import sent_tokenize

# DB setup

def init_db(conn: sqlite3.Connection):
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS documents (
        doc_id TEXT PRIMARY KEY,
        source TEXT NOT NULL,
        title TEXT,
        metadata TEXT
    );

    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
        doc_id TEXT NOT NULL,
        chunk_text TEXT NOT NULL,
        position INTEGER NOT NULL,
        faiss_id INTEGER UNIQUE,
        FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
    );

    CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
    """)
    conn.commit()


# Cleaning helper

def clean_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Source readers - each yields (doc_id, source, title, metadata_dict, [chunk_texts])
def read_arxiv(path: str, limit=None):
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            abstract = clean_text(rec.get("abstract", ""))
            if not abstract:
                continue

            doc_id = f"arxiv:{rec.get('id')}"
            title = rec.get("title", "").replace("\n", " ").strip()
            metadata = {
                "authors": rec.get("authors"),
                "categories": rec.get("categories"),
            }
            # Abstracts are short -> treat as a single chunk
            yield doc_id, "arxiv", title, metadata, [abstract]
            count += 1


def sliding_window_chunks(text: str, window: int, stride: int):
    sentences = sent_tokenize(text)
    if not sentences:
        return []
    if len(sentences) <= window:
        return [" ".join(sentences)]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = " ".join(sentences[i:i + window])
        chunks.append(chunk)
        if i + window >= len(sentences):
            break
        i += stride
    return chunks


def read_wiki(path: str, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    for rec in data:
        if limit is not None and count >= limit:
            break
        title = rec.get("title", "").strip()
        text = clean_text(rec.get("text", ""))
        if not text or not title:
            continue

        doc_id = f"wiki:{title}"
        chunks = sliding_window_chunks(text, WIKI_WINDOW, WIKI_STRIDE)
        if not chunks:
            continue

        yield doc_id, "wiki", title, {}, chunks
        count += 1



# Main ingestion
def main():
    print(f"Loading model: {MODEL_NAME} (CPU)")
    model = SentenceTransformer(MODEL_NAME, device="cpu")

    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    faiss_index = faiss.IndexIDMap(faiss.IndexFlatIP(EMBED_DIM))

    faiss_id_counter = 0
    pending_texts = []      # chunk texts waiting to be embedded
    pending_meta = []       # (doc_id, position) for each pending chunk

    def flush_batch():
        """Embed pending_texts, write to SQLite + FAISS, clear buffers."""
        nonlocal faiss_id_counter, pending_texts, pending_meta
        if not pending_texts:
            return

        embeddings = model.encode(
            pending_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # so inner product = cosine similarity
        )
        embeddings = np.asarray(embeddings, dtype="float32")

        ids = np.arange(
            faiss_id_counter, faiss_id_counter + len(pending_texts)
        ).astype("int64")
        faiss_index.add_with_ids(embeddings, ids)

        cur = conn.cursor()
        for text, (doc_id, position), fid in zip(pending_texts, pending_meta, ids):
            cur.execute(
                "INSERT INTO chunks (doc_id, chunk_text, position, faiss_id) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, text, position, int(fid)),
            )
        conn.commit()

        faiss_id_counter += len(pending_texts)
        pending_texts = []
        pending_meta = []

    def process_source(reader, label):
        doc_count = 0
        chunk_count = 0
        cur = conn.cursor()

        for doc_id, source, title, metadata, chunk_list in reader:
            cur.execute(
                "INSERT OR IGNORE INTO documents (doc_id, source, title, metadata) "
                "VALUES (?, ?, ?, ?)",
                (doc_id, source, title, json.dumps(metadata)),
            )
            doc_count += 1

            for pos, chunk_text in enumerate(chunk_list):
                pending_texts.append(chunk_text)
                pending_meta.append((doc_id, pos))
                chunk_count += 1

                if len(pending_texts) >= BATCH_SIZE:
                    flush_batch()

            if doc_count % 200 == 0:
                print(f"[{label}] processed {doc_count} docs, {chunk_count} chunks so far...")

        conn.commit()
        print(f"[{label}] done: {doc_count} docs, {chunk_count} chunks total")

    print("\n--- Ingesting arXiv ---")
    process_source(read_arxiv(ARXIV_PATH, limit=ARXIV_LIMIT), "arxiv")

    print("\n--- Ingesting Wikipedia ---")
    process_source(read_wiki(WIKI_PATH, limit=WIKI_LIMIT), "wiki")

    flush_batch()  # flush any remaining partial batch

    print(f"\nSaving FAISS index to {FAISS_PATH}")
    faiss.write_index(faiss_index, FAISS_PATH)

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"\nDone. {total_docs} documents, {total_chunks} chunks indexed.")
    print(f"SQLite DB: {DB_PATH}")
    print(f"FAISS index: {FAISS_PATH}")

    conn.close()


if __name__ == "__main__":
    main()