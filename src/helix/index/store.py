"""Persistence for collections, chunks, and monitors."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

from helix.config import DB_PATH, EMBED_STORE, ensure_dirs
from helix.corpus.library import EVAL_IDS, documents
from helix.ingest.parse import chunk_pages, parse_path


def now_local() -> str:
    """Local timestamp with the timezone offset."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    """Open the Helix database and ensure the schema exists."""
    ensure_dirs()
    EMBED_STORE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS collections (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS documents (
          id TEXT PRIMARY KEY,
          collection_id TEXT NOT NULL,
          cite_id TEXT NOT NULL,
          filename TEXT,
          title TEXT,
          source_type TEXT,
          created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS chunks (
          id TEXT PRIMARY KEY,
          document_id TEXT NOT NULL,
          collection_id TEXT NOT NULL,
          cite_id TEXT NOT NULL,
          page INTEGER NOT NULL,
          section TEXT,
          text TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS monitors (
          id TEXT PRIMARY KEY,
          collection_id TEXT NOT NULL,
          topic TEXT NOT NULL,
          created_at TEXT NOT NULL,
          last_run TEXT,
          last_brief TEXT
        );
        CREATE TABLE IF NOT EXISTS trajectories (
          id TEXT PRIMARY KEY,
          collection_id TEXT,
          question TEXT,
          mode TEXT,
          created_at TEXT NOT NULL,
          payload TEXT NOT NULL
        );
        """
    )
    return conn


def create_collection(name: str) -> dict:
    """Create an empty collection."""
    row = {"id": uuid.uuid4().hex, "name": name.strip() or "Untitled", "created_at": now_local()}
    with connect() as conn:
        conn.execute(
            "INSERT INTO collections (id, name, created_at) VALUES (?, ?, ?)",
            (row["id"], row["name"], row["created_at"]),
        )
    return row


def list_collections() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT c.id, c.name, c.created_at,
                   (
                     SELECT COUNT(*) FROM documents d
                     WHERE d.collection_id = c.id
                   ) AS documents,
                   (
                     SELECT COUNT(*) FROM chunks ch
                     WHERE ch.collection_id = c.id
                   ) AS chunks
            FROM collections c
            ORDER BY c.created_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def list_documents(collection_id: str) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, cite_id, filename, title, source_type, created_at
            FROM documents WHERE collection_id = ? ORDER BY created_at DESC
            """,
            (collection_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_chunks(collection_id: str, document_ids: list[str] | None = None) -> list[dict]:
    query = """
        SELECT ch.id, ch.document_id, ch.collection_id, ch.cite_id, ch.page,
               ch.section, ch.text, d.title AS title
        FROM chunks ch
        LEFT JOIN documents d ON d.id = ch.document_id
        WHERE ch.collection_id = ?
    """
    params: list = [collection_id]
    if document_ids:
        marks = ",".join("?" for _ in document_ids)
        query += f" AND ch.document_id IN ({marks})"
        params.extend(document_ids)
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def add_document(
    collection_id: str,
    filename: str,
    title: str,
    source_type: str,
    pages: list[dict],
    cite_id: str | None = None,
) -> dict:
    """Chunk pages, store them, and return the document record."""
    document_id = uuid.uuid4().hex
    label = cite_id or _cite_label(filename, document_id)
    created = now_local()
    chunks = chunk_pages(pages)
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO documents
              (id, collection_id, cite_id, filename, title, source_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (document_id, collection_id, label, filename, title, source_type, created),
        )
        for chunk in chunks:
            conn.execute(
                """
                INSERT INTO chunks
                  (id, document_id, collection_id, cite_id, page, section, text)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    document_id,
                    collection_id,
                    label,
                    chunk["page"],
                    chunk["section"],
                    chunk["text"],
                ),
            )
    _encode_collection(collection_id)
    return {
        "id": document_id,
        "cite_id": label,
        "filename": filename,
        "title": title,
        "source_type": source_type,
        "created_at": created,
        "chunks": len(chunks),
    }


def add_upload(collection_id: str, path: Path) -> dict:
    """Parse a file from disk and add it to a collection."""
    pages = parse_path(path)
    if not pages:
        raise ValueError("No text could be extracted from that file.")
    return add_document(
        collection_id=collection_id,
        filename=path.name,
        title=path.stem.replace("_", " "),
        source_type=path.suffix.lower().lstrip("."),
        pages=pages,
    )


def delete_document(document_id: str) -> None:
    with connect() as conn:
        row = conn.execute(
            "SELECT collection_id FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        if row is None:
            return
        collection_id = row["collection_id"]
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    _encode_collection(collection_id)


def reindex_all() -> None:
    """Recompute embeddings for every collection."""
    for collection in list_collections():
        _encode_collection(collection["id"])


def demo_documents() -> list:
    """Training notes shown in the demo collection.

    Held-out eval notes stay in the library for scoring, but they are not
    searchable from the default collection.
    """
    return [doc for doc in documents() if doc.doc_id not in EVAL_IDS]


def seed_demo() -> dict:
    """Load training notes into the demo collection when the database is empty.

    Held-out eval notes already stored as seed documents are removed.
    """
    _drop_heldout_seeds()
    existing = list_collections()
    if existing:
        return existing[0]
    collection = create_collection("Sample library")
    for doc in demo_documents():
        pages = [
            {"page": page.page, "section": page.section, "text": page.text}
            for page in doc.pages
        ]
        add_document(
            collection_id=collection["id"],
            filename=f"{doc.doc_id}.txt",
            title=doc.title,
            source_type="seed",
            pages=pages,
            cite_id=doc.doc_id,
        )
    return collection


def _drop_heldout_seeds() -> None:
    """Delete seeded copies of held-out eval notes and refresh embeddings."""
    heldout = tuple(sorted(EVAL_IDS))
    marks = ",".join("?" for _ in heldout)
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT id, collection_id FROM documents
            WHERE source_type = 'seed' AND cite_id IN ({marks})
            """,
            heldout,
        ).fetchall()
        doc_ids = [row["id"] for row in rows]
        collections = {row["collection_id"] for row in rows}
        if doc_ids:
            doc_marks = ",".join("?" for _ in doc_ids)
            conn.execute(
                f"DELETE FROM chunks WHERE document_id IN ({doc_marks})",
                doc_ids,
            )
            conn.execute(
                f"DELETE FROM documents WHERE id IN ({doc_marks})",
                doc_ids,
            )
    for collection_id in collections:
        _encode_collection(collection_id)


def create_monitor(collection_id: str, topic: str) -> dict:
    row = {
        "id": uuid.uuid4().hex,
        "collection_id": collection_id,
        "topic": topic.strip(),
        "created_at": now_local(),
        "last_run": None,
        "last_brief": None,
    }
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO monitors (id, collection_id, topic, created_at, last_run, last_brief)
            VALUES (?, ?, ?, ?, NULL, NULL)
            """,
            (row["id"], row["collection_id"], row["topic"], row["created_at"]),
        )
    return row


def list_monitors(collection_id: str | None = None) -> list[dict]:
    query = "SELECT * FROM monitors"
    params: tuple = ()
    if collection_id:
        query += " WHERE collection_id = ?"
        params = (collection_id,)
    query += " ORDER BY created_at DESC"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        if item["last_brief"]:
            item["last_brief"] = json.loads(item["last_brief"])
        result.append(item)
    return result


def save_monitor_brief(monitor_id: str, brief: dict) -> None:
    with connect() as conn:
        conn.execute(
            "UPDATE monitors SET last_run = ?, last_brief = ? WHERE id = ?",
            (now_local(), json.dumps(brief), monitor_id),
        )


def get_monitor(monitor_id: str) -> dict | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM monitors WHERE id = ?", (monitor_id,)).fetchone()
    return dict(row) if row else None


def save_trajectory(collection_id: str, question: str, mode: str, payload: dict) -> str:
    row_id = uuid.uuid4().hex
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO trajectories
              (id, collection_id, question, mode, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (row_id, collection_id, question, mode, now_local(), json.dumps(payload)),
        )
    return row_id


def _cite_label(filename: str, document_id: str) -> str:
    stem = Path(filename).stem
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in stem)
    cleaned = cleaned.strip("-")[:24] or document_id[:8]
    return cleaned.upper()


def _encode_collection(collection_id: str) -> None:
    from helix.models.embedder import get_embedder

    chunks = get_chunks(collection_id)
    path = EMBED_STORE / f"{collection_id}.npz"
    if not chunks:
        if path.exists():
            path.unlink()
        return
    vectors = get_embedder().encode([chunk["text"] for chunk in chunks])
    np.savez(
        path,
        ids=np.array([chunk["id"] for chunk in chunks]),
        vectors=vectors.astype(np.float32),
    )


def load_embeddings(collection_id: str) -> tuple[list[str], np.ndarray] | None:
    path = EMBED_STORE / f"{collection_id}.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=False)
    return list(data["ids"]), data["vectors"]
