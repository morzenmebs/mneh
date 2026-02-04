"""Capture URLs and extract content."""

import json
import trafilatura
from pathlib import Path

from .storage import (
    connect,
    store_capture,
    store_chunks,
    store_embedding,
    store_handles,
    store_metadata,
)
from .extract import extract_metadata
from .embed import embed_texts
from .chunk import chunk_text


def fetch_text(url: str) -> tuple[str, dict]:
    """Fetch a URL and return (extracted_text, trafilatura_metadata_hints)."""
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch {url}")

    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract content from {url}")

    traf_meta: dict = {}
    try:
        traf_json = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
        if traf_json:
            traf_meta = json.loads(traf_json)
    except Exception:
        pass

    return text, traf_meta


def capture_url(
    db_path: Path,
    url: str,
    storage_dir: Path,
    *,
    system_prompt: str | None = None,
    model: str = "gpt-5.2",
) -> str:
    """Fetch URL, extract content, store with embeddings, return hash."""
    text, traf_meta = fetch_text(url)
    
    # Store raw
    conn = connect(db_path)
    h = store_capture(conn, text, url, content_type="text/plain", storage_dir=storage_dir)
    
    # LLM extraction with hints
    meta = extract_metadata(
        text,
        url=url,
        hints={
            "title": traf_meta.get("title"),
            "author": traf_meta.get("author"),
            "date": traf_meta.get("date"),
        },
        system_prompt=system_prompt,
        model=model,
    )
    
    store_metadata(
        conn,
        h,
        title=meta.get("title"),
        author=meta.get("author"),
        date=meta.get("date"),
        topics=None,
        summary=meta.get("summary"),
    )
    
    # Chunk
    chunks = chunk_text(text)
    store_chunks(conn, h, chunks)

    handles = meta.get("handles") or []
    store_handles(conn, h, handles, style="llm")
    
    # Embed everything in one batch
    texts_to_embed = handles + chunks
    
    if texts_to_embed:
        vectors = embed_texts(texts_to_embed)
        
        # Store handle embeddings
        for i, vec in enumerate(vectors[: len(handles)]):
            store_embedding(conn, h, "handle", vec, chunk_index=i)

        # Store chunk embeddings
        for i, vec in enumerate(vectors[len(handles) :]):
            store_embedding(conn, h, "chunk", vec, chunk_index=i)
    
    return h
