"""SQLite storage layer."""

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
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            source_type TEXT,
            source_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(
            content,
            content='items',
            content_rowid='rowid'
        );
    """)
    conn.commit()


def insert_item(
    conn: sqlite3.Connection,
    item_id: str,
    content: str,
    source_type: str | None = None,
    source_path: str | None = None,
) -> None:
    """Insert an item into the database."""
    conn.execute(
        "INSERT INTO items (id, content, source_type, source_path) VALUES (?, ?, ?, ?)",
        (item_id, content, source_type, source_path),
    )
    conn.execute(
        "INSERT INTO items_fts (rowid, content) SELECT rowid, content FROM items WHERE id = ?",
        (item_id,),
    )
    conn.commit()


def search_fts(conn: sqlite3.Connection, query: str) -> list[dict]:
    """Full-text search over items."""
    rows = conn.execute(
        """
        SELECT items.* FROM items
        JOIN items_fts ON items.rowid = items_fts.rowid
        WHERE items_fts MATCH ?
        ORDER BY rank
        """,
        (query,),
    ).fetchall()
    return [dict(row) for row in rows]
