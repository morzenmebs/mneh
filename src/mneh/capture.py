"""Capture URLs and extract content."""

import json
import trafilatura
from pathlib import Path

from .storage import connect, store_capture, store_metadata, store_embedding
from .extract import extract_metadata
from .embed import embed_texts
from .chunk import chunk_text


def capture_url(db_path: Path, url: str, storage_dir: Path) -> str:
    """Fetch URL, extract content, store with embeddings, return hash."""
    # Fetch and extract
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch {url}")
    
    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract content from {url}")
    
    # Get trafilatura's metadata as hints
    traf_meta = {}
    try:
        traf_json = trafilatura.extract(downloaded, output_format='json', with_metadata=True)
        if traf_json:
            traf_meta = json.loads(traf_json)
    except:
        pass
    
    # Store raw
    conn = connect(db_path)
    h = store_capture(conn, text, url, content_type="text/plain", storage_dir=storage_dir)
    
    # LLM extraction with hints
    meta = extract_metadata(text, url=url, hints={
        'title': traf_meta.get('title'),
        'author': traf_meta.get('author'),
        'date': traf_meta.get('date'),
    })
    
    store_metadata(
        conn, h,
        title=meta.get("title"),
        author=meta.get("author"),
        date=meta.get("date"),
        topics=meta.get("topics"),
    )
    
    # Chunk
    chunks = chunk_text(text)
    
    # Embed everything in one batch
    topics = meta.get("topics", [])
    texts_to_embed = topics + chunks
    
    if texts_to_embed:
        vectors = embed_texts(texts_to_embed)
        
        # Store topic embeddings
        for i, vec in enumerate(vectors[:len(topics)]):
            store_embedding(conn, h, "topic", vec)
        
        # Store chunk embeddings
        for i, vec in enumerate(vectors[len(topics):]):
            store_embedding(conn, h, "chunk", vec, chunk_index=i)
    
    return h
