"""SQLite storage layer.

This module owns:
- schema creation / incremental evolution
- persistence of captures, metadata, handles, chunks, embeddings
- retrieval helpers (FTS + vector KNN + RRF fusion)

Vector search uses the `sqlite-vec` extension.
https://alexgarcia.xyz/sqlite-vec/python.html
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EMBEDDING_DIM = 3072  # text-embedding-3-large


def connect(db_path: Path) -> sqlite3.Connection:
    """Open database, creating tables if needed."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _load_sqlite_vec(conn)
    _init_schema(conn)
    return conn


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    """Attempt to load sqlite-vec (optional)."""
    try:
        import sqlite_vec

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except Exception:
        try:
            conn.enable_load_extension(False)
        except Exception:
            pass


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
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
            kind TEXT,
            chunk_index INTEGER,
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


# -----------------------------------------------------------------------------
# Basic CRUD
# -----------------------------------------------------------------------------


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
        conn.execute("INSERT INTO handles_fts(rowid, text) VALUES (?, ?)", (handle_id, htxt))
    conn.commit()


def store_chunks(
    conn: sqlite3.Connection,
    hash: str,
    chunks: list[str],
) -> None:
    """Store chunks and populate FTS."""
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
    import sqlite_vec

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


def delete_embeddings(conn: sqlite3.Connection, hash: str, kind: str | None = None) -> None:
    """Delete embeddings for a hash. If kind is None, delete all kinds."""
    if kind:
        conn.execute("DELETE FROM embeddings WHERE hash = ? AND kind = ?", (hash, kind))
    else:
        conn.execute("DELETE FROM embeddings WHERE hash = ?", (hash,))
    conn.commit()


# -----------------------------------------------------------------------------
# List / Get / Delete
# -----------------------------------------------------------------------------


@dataclass
class CaptureInfo:
    """Summary info for a capture."""

    hash: str
    source_uri: str
    captured_at: str
    title: str | None
    author: str | None
    date: str | None
    summary: str | None
    chunk_count: int
    handle_count: int


def list_captures(conn: sqlite3.Connection) -> list[CaptureInfo]:
    """List all captures with metadata, ordered by capture date descending."""
    rows = conn.execute(
        """
        SELECT 
            c.hash,
            c.source_uri,
            c.captured_at,
            m.title,
            m.author,
            m.date,
            m.summary,
            (SELECT COUNT(*) FROM chunks WHERE chunks.hash = c.hash) as chunk_count,
            (SELECT COUNT(*) FROM handles WHERE handles.hash = c.hash) as handle_count
        FROM captures c
        LEFT JOIN metadata m ON c.hash = m.hash
        ORDER BY c.captured_at DESC
        """
    ).fetchall()

    return [
        CaptureInfo(
            hash=r["hash"],
            source_uri=r["source_uri"],
            captured_at=r["captured_at"],
            title=r["title"],
            author=r["author"],
            date=r["date"],
            summary=r["summary"],
            chunk_count=r["chunk_count"],
            handle_count=r["handle_count"],
        )
        for r in rows
    ]


@dataclass
class CaptureDetail:
    """Full detail for a single capture."""

    hash: str
    source_uri: str
    captured_at: str
    raw_path: str | None
    title: str | None
    author: str | None
    date: str | None
    summary: str | None
    handles: list[str]
    chunks: list[str]


def get_capture(conn: sqlite3.Connection, hash_prefix: str) -> CaptureDetail | None:
    """Get a capture by hash or hash prefix. Returns None if not found or ambiguous."""
    # Try exact match first
    row = conn.execute(
        """
        SELECT c.hash, c.source_uri, c.captured_at, c.raw_path,
               m.title, m.author, m.date, m.summary
        FROM captures c
        LEFT JOIN metadata m ON c.hash = m.hash
        WHERE c.hash = ?
        """,
        (hash_prefix,),
    ).fetchone()

    # If no exact match, try prefix
    if not row:
        rows = conn.execute(
            """
            SELECT c.hash, c.source_uri, c.captured_at, c.raw_path,
                   m.title, m.author, m.date, m.summary
            FROM captures c
            LEFT JOIN metadata m ON c.hash = m.hash
            WHERE c.hash LIKE ?
            """,
            (hash_prefix + "%",),
        ).fetchall()

        if len(rows) == 0:
            return None
        if len(rows) > 1:
            # Ambiguous prefix
            return None
        row = rows[0]

    h = row["hash"]

    handles = [
        r["text"]
        for r in conn.execute(
            "SELECT text FROM handles WHERE hash = ? ORDER BY handle_index", (h,)
        ).fetchall()
    ]

    chunks = [
        r["text"]
        for r in conn.execute(
            "SELECT text FROM chunks WHERE hash = ? ORDER BY chunk_index", (h,)
        ).fetchall()
    ]

    return CaptureDetail(
        hash=h,
        source_uri=row["source_uri"],
        captured_at=row["captured_at"],
        raw_path=row["raw_path"],
        title=row["title"],
        author=row["author"],
        date=row["date"],
        summary=row["summary"],
        handles=handles,
        chunks=chunks,
    )


def get_last_capture(conn: sqlite3.Connection) -> CaptureDetail | None:
    """Get the most recently captured item."""
    row = conn.execute(
        "SELECT hash FROM captures ORDER BY captured_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return get_capture(conn, row["hash"])


def delete_capture(conn: sqlite3.Connection, hash_prefix: str, storage_dir: Path | None = None) -> str | None:
    """Delete a capture by hash/prefix. Returns the deleted hash, or None if not found."""
    # Resolve prefix to full hash
    detail = get_capture(conn, hash_prefix)
    if not detail:
        return None

    h = detail.hash

    # Delete FTS entries first (external content tables)
    chunk_ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE hash = ?", (h,))]
    if chunk_ids:
        conn.execute(
            f"DELETE FROM chunks_fts WHERE rowid IN ({','.join(['?'] * len(chunk_ids))})",
            chunk_ids,
        )

    handle_ids = [r[0] for r in conn.execute("SELECT id FROM handles WHERE hash = ?", (h,))]
    if handle_ids:
        conn.execute(
            f"DELETE FROM handles_fts WHERE rowid IN ({','.join(['?'] * len(handle_ids))})",
            handle_ids,
        )

    # Delete from tables
    conn.execute("DELETE FROM embeddings WHERE hash = ?", (h,))
    conn.execute("DELETE FROM chunks WHERE hash = ?", (h,))
    conn.execute("DELETE FROM handles WHERE hash = ?", (h,))
    conn.execute("DELETE FROM metadata WHERE hash = ?", (h,))
    conn.execute("DELETE FROM captures WHERE hash = ?", (h,))
    conn.commit()

    # Delete raw file
    if detail.raw_path:
        raw = Path(detail.raw_path)
        if raw.exists():
            raw.unlink()
            # Try to remove parent dir if empty
            try:
                raw.parent.rmdir()
            except OSError:
                pass

    return h


# -----------------------------------------------------------------------------
# Search
# -----------------------------------------------------------------------------


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
    """Brute-force KNN over embeddings."""
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
    kind: str
    index: int


def rrf_fuse(
    ranked_lists: dict[str, list[HitKey]],
    *,
    k: int = 60,
) -> dict[HitKey, float]:
    """Reciprocal Rank Fusion over several ranked lists.

    Score = sum(1 / (k + rank)).

    With k=60 and 4 channels:
    - #1 in all channels: ~0.066
    - #1 in one channel only: ~0.016
    - Scores >0.05 are strong, 0.02-0.05 moderate, <0.02 weak
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

    fts_chunks = search_fts_chunks(conn, query, k=k_each)
    ranked["fts_chunks"] = [HitKey(r["hash"], "chunk", int(r["chunk_index"])) for r in fts_chunks]

    fts_handles = search_fts_handles(conn, query, k=k_each)
    ranked["fts_handles"] = [HitKey(r["hash"], "handle", int(r["handle_index"])) for r in fts_handles]

    if query_vec is not None:
        vec_handles = knn_embeddings(conn, query_vec, kind="handle", k=k_each, distance="cosine")
        ranked["vec_handles"] = [HitKey(r["hash"], "handle", int(r["chunk_index"])) for r in vec_handles]

        vec_chunks = knn_embeddings(conn, query_vec, kind="chunk", k=k_each, distance="cosine")
        ranked["vec_chunks"] = [HitKey(r["hash"], "chunk", int(r["chunk_index"])) for r in vec_chunks]

    fused = rrf_fuse(ranked, k=rrf_k)

    top_keys = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k_each]

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
