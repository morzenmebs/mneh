"""SQLite storage layer."""

import hashlib
import sqlite3
from pathlib import Path


def connect(db_path: Path) -> sqlite3.Connection:
    """Open database, creating tables if needed."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
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
            extracted_at TEXT DEFAULT CURRENT_TIMESTAMP
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
    """)
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
) -> None:
    """Store extracted metadata for a capture."""
    conn.execute(
        "INSERT OR REPLACE INTO metadata (hash, title, author, date) VALUES (?, ?, ?, ?)",
        (hash, title, author, date),
    )
    conn.commit()


def search_fts(conn: sqlite3.Connection, query: str) -> list[dict]:
    """Full-text search over chunks."""
    rows = conn.execute(
        """
        SELECT chunks.*, captures.source_uri, metadata.title
        FROM chunks
        JOIN chunks_fts ON chunks.id = chunks_fts.rowid
        JOIN captures ON chunks.hash = captures.hash
        LEFT JOIN metadata ON chunks.hash = metadata.hash
        WHERE chunks_fts MATCH ?
        ORDER BY rank
        """,
        (query,),
    ).fetchall()
    return [dict(row) for row in rows]
