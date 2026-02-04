"""SQLite storage layer.

This module owns:
- schema creation / incremental evolution
- persistence of captures, metadata, handles, chunks, embeddings
- retrieval helpers (FTS + vector KNN + RRF fusion)

Vector search uses the `sqlite-vec` extension.

Python binding docs show how to load the extension via `sqlite_vec.load(conn)`.
https://alexgarcia.xyz/sqlite-vec/python.html
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


# OpenAI text-embedding-3-large has 3072 dimensions.
# We keep this here so we can validate vectors and serialize consistently.
EMBEDDING_DIM = 3072


def connect(db_path: Path) -> sqlite3.Connection:
    """Open database, creating tables if needed."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)
    _init_schema(conn)
    return conn


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Attempt to load sqlite-vec (optional).

    If the extension isn't available (e.g., during minimal test environments),
    we keep going—FTS and capture still work.
    """
    try:
        import sqlite_vec  # type: ignore

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)  # registers SQL functions like vec_distance_cosine
        conn.enable_load_extension(False)
    except Exception:
        # Intentionally swallow: vector search will fail loudly if called.
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist (and evolve gently)."""

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS captures (
            hash TEXT PRIMARY KEY,
            source_uri TEXT NOT NULL,
            content_type TEXT,
            captured_at TEXT DEFAULT CURRENT_TIMESTAMP,
            raw_path TEXT
        );

        CREATE TABLE IF NOT EXISTS metadata (
            hash TEXT PRIMARY KEY REFERENCES captures(hash),
            title TEXT,
            author TEXT,
            date TEXT,
            topics_json TEXT,
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS handles (
            id INTEGER PRIMARY KEY,
            hash TEXT REFERENCES captures(hash),
            handle_index INTEGER,
            text TEXT NOT NULL,
            style TEXT,
            UNIQUE(hash, handle_index)
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS handles_fts USING fts5(
            text,
            content='handles',
            content_rowid='id'
        );

        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            hash TEXT REFERENCES captures(hash),
            chunk_index INTEGER,
            start_offset INTEGER,
            end_offset INTEGER,
            text TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            content='chunks',
            content_rowid='id'
        );

        CREATE TABLE IF NOT EXISTS embeddings (
            id INTEGER PRIMARY KEY,
            hash TEXT REFERENCES captures(hash),
            kind TEXT,  -- 'handle', 'chunk', (future: 'claim')
            chunk_index INTEGER,  -- for 'chunk': chunk_index; for 'handle': handle_index
            vector BLOB
        );
        """
    )

    # Gentle evolution: add metadata.summary if missing.
    try:
        conn.execute("ALTER TABLE metadata ADD COLUMN summary TEXT")
    except sqlite3.OperationalError:
        pass

    conn.commit()


def store_capture(
    conn: sqlite3.Connection,
    content: str,
    source_uri: str,
    content_type: str | None = None,
    storage_dir: Path | None = None,
) -> str:
    """Store raw content, return hash."""
    h = hashlib.sha256(content.encode()).hexdigest()

    raw_path = None
    if storage_dir:
        subdir = storage_dir / h[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        raw_path = subdir / h
        raw_path.write_text(content, encoding="utf-8")

    conn.execute(
        "INSERT OR IGNORE INTO captures (hash, source_uri, content_type, raw_path) VALUES (?, ?, ?, ?)",
        (h, source_uri, content_type, str(raw_path) if raw_path else None),
    )
    conn.commit()
    return h


def store_metadata(
    conn: sqlite3.Connection,
    hash: str,
    title: str | None = None,
    author: str | None = None,
    date: str | None = None,
    topics: list[str] | None = None,
    summary: str | None = None,
) -> None:
    """Store extracted metadata for a capture."""
    conn.execute(
        """
        INSERT OR REPLACE INTO metadata (hash, title, author, date, topics_json, summary)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (hash, title, author, date, json.dumps(topics) if topics else None, summary),
    )
    conn.commit()


def store_handles(
    conn: sqlite3.Connection,
    hash: str,
    handles: list[str],
    style: str | None = None,
) -> None:
    """Store retrieval handles (query-shaped strings)."""
    # Wipe existing rows + their FTS entries.
    old_ids = [r[0] for r in conn.execute("SELECT id FROM handles WHERE hash = ?", (hash,))]
    if old_ids:
        conn.execute(
            f"DELETE FROM handles_fts WHERE rowid IN ({','.join(['?'] * len(old_ids))})",
            old_ids,
        )
    conn.execute("DELETE FROM handles WHERE hash = ?", (hash,))
    for i, htxt in enumerate(handles):
        cur = conn.execute(
            "INSERT INTO handles (hash, handle_index, text, style) VALUES (?, ?, ?, ?)",
            (hash, i, htxt, style),
        )
        handle_id = cur.lastrowid
        # External content table: keep handles_fts in sync manually.
        conn.execute("INSERT INTO handles_fts(rowid, text) VALUES (?, ?)", (handle_id, htxt))
    conn.commit()


def store_chunks(
    conn: sqlite3.Connection,
    hash: str,
    chunks: list[str],
) -> None:
    """Store chunks and populate FTS."""
    # Also wipe FTS rows for this hash.
    # Since chunks_fts is an external content table, deleting from chunks does
    # NOT delete from chunks_fts automatically.
    old_ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE hash = ?", (hash,))]
    if old_ids:
        conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN ({','.join(['?'] * len(old_ids))})",
            old_ids,
        )
    conn.execute("DELETE FROM chunks WHERE hash = ?", (hash,))

    for i, text in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks (hash, chunk_index, start_offset, end_offset, text) VALUES (?, ?, ?, ?, ?)",
            (hash, i, None, None, text),
        )
        rowid = cur.lastrowid
        conn.execute("INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)", (rowid, text))

    conn.commit()


def _serialize_vec_f32(vec: list[float]) -> bytes:
    """Serialize a Python list[float] into sqlite-vec's float32 blob format."""
    import sqlite_vec  # type: ignore

    # sqlite-vec docs: serialize_float32 packs float32-compatible BLOBs.
    # https://alexgarcia.xyz/sqlite-vec/python.html
    return sqlite_vec.serialize_float32(vec)


def store_embedding(
    conn: sqlite3.Connection,
    hash: str,
    kind: str,
    vector: list[float],
    chunk_index: int | None = None,
) -> None:
    """Store an embedding vector."""
    blob = _serialize_vec_f32(vector)
    conn.execute(
        "INSERT INTO embeddings (hash, kind, chunk_index, vector) VALUES (?, ?, ?, ?)",
        (hash, kind, chunk_index, blob),
    )
    conn.commit()


def delete_embeddings(conn: sqlite3.Connection, hash: str, kind: str) -> None:
    conn.execute("DELETE FROM embeddings WHERE hash = ? AND kind = ?", (hash, kind))
    conn.commit()


def search_fts_chunks(conn: sqlite3.Connection, query: str, k: int = 50) -> list[dict[str, Any]]:
    """Full-text search over chunks."""
    sql = """
        SELECT chunks.id as chunk_id, chunks.hash, chunks.chunk_index, chunks.text,
               captures.source_uri, metadata.title, metadata.summary
        FROM chunks
        JOIN chunks_fts ON chunks.id = chunks_fts.rowid
        JOIN captures ON chunks.hash = captures.hash
        LEFT JOIN metadata ON chunks.hash = metadata.hash
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, (query, k)).fetchall()
    except sqlite3.OperationalError:
        # FTS5 query syntax is picky; fall back to a quoted phrase.
        rows = conn.execute(sql, (f'"{query}"', k)).fetchall()
    return [dict(r) for r in rows]


def search_fts_handles(conn: sqlite3.Connection, query: str, k: int = 50) -> list[dict[str, Any]]:
    """Full-text search over handles."""
    sql = """
        SELECT handles.id as handle_id, handles.hash, handles.handle_index, handles.text,
               captures.source_uri, metadata.title, metadata.summary
        FROM handles
        JOIN handles_fts ON handles.id = handles_fts.rowid
        JOIN captures ON handles.hash = captures.hash
        LEFT JOIN metadata ON handles.hash = metadata.hash
        WHERE handles_fts MATCH ?
        ORDER BY rank
        LIMIT ?
    """
    try:
        rows = conn.execute(sql, (query, k)).fetchall()
    except sqlite3.OperationalError:
        rows = conn.execute(sql, (f'"{query}"', k)).fetchall()
    return [dict(r) for r in rows]


def knn_embeddings(
    conn: sqlite3.Connection,
    query_vec: list[float],
    *,
    kind: str,
    k: int = 50,
    distance: str = "cosine",
) -> list[dict[str, Any]]:
    """Brute-force KNN over embeddings using sqlite-vec scalar distance functions.

    sqlite-vec supports vec_distance_L2 / vec_distance_L1 / vec_distance_cosine.
    https://alexgarcia.xyz/sqlite-vec/features/knn.html
    """
    q = _serialize_vec_f32(query_vec)
    if distance == "cosine":
        dist_fn = "vec_distance_cosine"
    elif distance == "l2":
        dist_fn = "vec_distance_L2"
    elif distance == "l1":
        dist_fn = "vec_distance_L1"
    else:
        raise ValueError(f"Unknown distance metric: {distance}")

    rows = conn.execute(
        f"""
        SELECT e.id, e.hash, e.kind, e.chunk_index,
               {dist_fn}(e.vector, ?) AS distance,
               captures.source_uri, metadata.title, metadata.summary
        FROM embeddings e
        JOIN captures ON e.hash = captures.hash
        LEFT JOIN metadata ON e.hash = metadata.hash
        WHERE e.kind = ?
        ORDER BY distance ASC
        LIMIT ?
        """,
        (q, kind, k),
    ).fetchall()
    return [dict(r) for r in rows]


@dataclass(frozen=True)
class HitKey:
    hash: str
    kind: str  # 'chunk' or 'handle'
    index: int


def rrf_fuse(
    ranked_lists: dict[str, list[HitKey]],
    *,
    k: int = 60,
) -> dict[HitKey, float]:
    """Reciprocal Rank Fusion over several ranked lists.

    Score = sum(1 / (k + rank)).
    Reference: Cormack et al., SIGIR 2009.
    https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
    """
    scores: dict[HitKey, float] = {}
    for _, hits in ranked_lists.items():
        for rank, hk in enumerate(hits, start=1):
            scores[hk] = scores.get(hk, 0.0) + 1.0 / (k + rank)
    return scores


def search_hybrid(
    conn: sqlite3.Connection,
    query: str,
    query_vec: list[float] | None,
    *,
    k_each: int = 50,
    rrf_k: int = 60,
) -> tuple[list[dict[str, Any]], dict[str, list[HitKey]]]:
    """Hybrid retrieval: FTS + handle vectors + chunk vectors, fused by RRF.

    Returns (ranked_results, debug_ranked_lists).
    """
    ranked: dict[str, list[HitKey]] = {}

    # Lexical (chunks)
    fts_chunks = search_fts_chunks(conn, query, k=k_each)
    ranked["fts_chunks"] = [HitKey(r["hash"], "chunk", int(r["chunk_index"])) for r in fts_chunks]

    # Lexical (handles)
    fts_handles = search_fts_handles(conn, query, k=k_each)
    ranked["fts_handles"] = [HitKey(r["hash"], "handle", int(r["handle_index"])) for r in fts_handles]

    # Vector channels (optional if query_vec provided)
    vec_handles: list[dict[str, Any]] = []
    vec_chunks: list[dict[str, Any]] = []
    if query_vec is not None:
        vec_handles = knn_embeddings(conn, query_vec, kind="handle", k=k_each, distance="cosine")
        ranked["vec_handles"] = [HitKey(r["hash"], "handle", int(r["chunk_index"])) for r in vec_handles]

        vec_chunks = knn_embeddings(conn, query_vec, kind="chunk", k=k_each, distance="cosine")
        ranked["vec_chunks"] = [HitKey(r["hash"], "chunk", int(r["chunk_index"])) for r in vec_chunks]

    fused = rrf_fuse(ranked, k=rrf_k)

    # Gather details for top fused hits.
    # We fetch payloads in two batches: chunks and handles.
    top_keys = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[: k_each]

    chunk_ids = [(hk.hash, hk.index) for hk, _ in top_keys if hk.kind == "chunk"]
    handle_ids = [(hk.hash, hk.index) for hk, _ in top_keys if hk.kind == "handle"]

    chunk_map: dict[tuple[str, int], dict[str, Any]] = {}
    if chunk_ids:
        placeholders = ",".join(["(?,?)"] * len(chunk_ids))
        flat: list[Any] = [x for pair in chunk_ids for x in pair]
        rows = conn.execute(
            f"""
            SELECT chunks.hash, chunks.chunk_index, chunks.text,
                   captures.source_uri, metadata.title, metadata.summary
            FROM chunks
            JOIN captures ON chunks.hash = captures.hash
            LEFT JOIN metadata ON chunks.hash = metadata.hash
            WHERE (chunks.hash, chunks.chunk_index) IN ({placeholders})
            """,
            flat,
        ).fetchall()
        for r in rows:
            chunk_map[(r["hash"], int(r["chunk_index"]))] = dict(r)

    handle_map: dict[tuple[str, int], dict[str, Any]] = {}
    if handle_ids:
        placeholders = ",".join(["(?,?)"] * len(handle_ids))
        flat = [x for pair in handle_ids for x in pair]
        rows = conn.execute(
            f"""
            SELECT handles.hash, handles.handle_index, handles.text,
                   captures.source_uri, metadata.title, metadata.summary
            FROM handles
            JOIN captures ON handles.hash = captures.hash
            LEFT JOIN metadata ON handles.hash = metadata.hash
            WHERE (handles.hash, handles.handle_index) IN ({placeholders})
            """,
            flat,
        ).fetchall()
        for r in rows:
            handle_map[(r["hash"], int(r["handle_index"]))] = dict(r)

    out: list[dict[str, Any]] = []
    for hk, score in top_keys:
        payload: dict[str, Any] | None
        if hk.kind == "chunk":
            payload = chunk_map.get((hk.hash, hk.index))
        else:
            payload = handle_map.get((hk.hash, hk.index))
        if not payload:
            continue
        payload = dict(payload)
        payload["rrf_score"] = score
        payload["hit_kind"] = hk.kind
        payload["hit_index"] = hk.index
        out.append(payload)

    return out, ranked
