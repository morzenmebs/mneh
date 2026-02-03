"""Capture URLs and extract content."""

import trafilatura
from pathlib import Path

from .storage import connect, store_capture, store_metadata


def capture_url(db_path: Path, url: str, storage_dir: Path) -> str:
    """Fetch URL, extract content, store capture, return hash."""
    # Fetch and extract
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        raise ValueError(f"Failed to fetch {url}")
    
    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract content from {url}")
    
    # Store
    conn = connect(db_path)
    h = store_capture(conn, text, url, content_type="text/plain", storage_dir=storage_dir)
    
    # TODO: LLM metadata extraction goes here
    # For now, try to get title from trafilatura
    metadata = trafilatura.extract(downloaded, output_format="json", include_comments=False)
    if metadata:
        import json
        meta = json.loads(metadata)
        store_metadata(conn, h, title=meta.get("title"), author=meta.get("author"), date=meta.get("date"))
    
    return h
