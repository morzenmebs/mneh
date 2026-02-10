"""Capture URLs and extract content."""

from __future__ import annotations

import io
import json
import re
import tarfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import httpx
import trafilatura

from .storage import (
    connect,
    delete_embeddings,
    store_capture,
    store_chunks,
    store_embedding,
    store_handles,
    store_metadata,
)

ARXIV_NEW_ID = re.compile(r"^(?P<core>\d{4}\.\d{4,5})(?P<version>v\d+)?$", re.IGNORECASE)
ARXIV_OLD_ID = re.compile(
    r"^(?P<core>[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?P<version>v\d+)?$", re.IGNORECASE
)
ARXIV_PREFIXED = re.compile(r"^arxiv:(?P<id>.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ArxivRef:
    """Parsed arXiv identifier."""

    identifier: str
    version: str | None = None

    @property
    def with_version(self) -> str:
        return f"{self.identifier}{self.version or ''}"

    @property
    def canonical_url(self) -> str:
        return f"https://arxiv.org/abs/{self.with_version}"


def parse_arxiv_reference(value: str) -> ArxivRef | None:
    """Parse arXiv IDs and arxiv.org URLs to a normalized reference."""
    raw = value.strip()
    if not raw:
        return None

    prefixed = ARXIV_PREFIXED.match(raw)
    if prefixed:
        raw = prefixed.group("id").strip()

    parsed = urlparse(raw)
    if parsed.scheme in {"http", "https"} and parsed.netloc.lower().endswith("arxiv.org"):
        path = parsed.path.strip("/")
        parts = [p for p in path.split("/") if p]
        if len(parts) >= 2 and parts[0] in {"abs", "pdf", "html", "src"}:
            raw = parts[1]
        elif parts:
            raw = parts[-1]

        if raw.endswith(".pdf"):
            raw = raw[:-4]

    for pattern in (ARXIV_NEW_ID, ARXIV_OLD_ID):
        match = pattern.match(raw)
        if match:
            return ArxivRef(identifier=match.group("core"), version=match.group("version"))

    return None


def _read_tex_from_tar(data: bytes) -> list[str]:
    """Extract .tex files from a tar/tar.gz payload."""
    texts: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".tex"):
                continue
            extracted = tf.extractfile(member)
            if not extracted:
                continue
            raw = extracted.read()
            texts.append(raw.decode("utf-8", errors="ignore") or raw.decode("latin-1", errors="ignore"))
    return texts


def _looks_like_tex(text: str) -> bool:
    return "\\begin{" in text or "\\documentclass" in text or "\\section{" in text


def latex_to_plaintext(tex: str) -> str:
    """Best-effort conversion from LaTeX source to readable plain text."""
    text = tex
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(r"\\begin\{(figure|table|equation|align\*?|tikzpicture)\}.*?\\end\{\1\}", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"\\\[.*?\\\]", " ", text, flags=re.DOTALL)
    text = re.sub(r"\\\(.*?\\\)", " ", text, flags=re.DOTALL)

    for _ in range(3):
        replaced = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
        if replaced == text:
            break
        text = replaced

    text = re.sub(r"\\[a-zA-Z@]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", " ").replace("}", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_arxiv_text(ref: ArxivRef) -> tuple[str, dict]:
    """Download arXiv source and convert to plaintext, plus metadata hints."""
    source_url = f"https://arxiv.org/e-print/{ref.with_version}"
    response = httpx.get(source_url, follow_redirects=True, timeout=60.0)
    response.raise_for_status()

    payload = response.content
    tex_sources: list[str] = []

    try:
        tex_sources = _read_tex_from_tar(payload)
    except tarfile.TarError:
        decoded = payload.decode("utf-8", errors="ignore")
        if _looks_like_tex(decoded):
            tex_sources = [decoded]

    if not tex_sources:
        raise ValueError(f"Could not extract TeX source from {source_url}")

    combined_tex = "\n\n".join(tex_sources)
    text = latex_to_plaintext(combined_tex)
    if not text:
        raise ValueError(f"TeX extraction produced empty text for {ref.with_version}")

    meta: dict = {}
    try:
        api_url = f"https://export.arxiv.org/api/query?id_list={ref.identifier}"
        api_response = httpx.get(api_url, timeout=30.0)
        api_response.raise_for_status()
        root = ET.fromstring(api_response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entry = root.find("atom:entry", ns)
        if entry is not None:
            authors = [a.findtext("atom:name", default="", namespaces=ns).strip() for a in entry.findall("atom:author", ns)]
            meta = {
                "title": (entry.findtext("atom:title", default="", namespaces=ns) or "").strip() or None,
                "author": ", ".join([a for a in authors if a]) or None,
                "date": (entry.findtext("atom:published", default="", namespaces=ns) or "").strip() or None,
            }
    except Exception:
        pass

    return text, meta


def fetch_text(source: str) -> tuple[str, dict, str]:
    """Fetch a source and return (text, metadata_hints, canonical_source_uri)."""
    arxiv_ref = parse_arxiv_reference(source)
    if arxiv_ref:
        text, meta = fetch_arxiv_text(arxiv_ref)
        return text, meta, arxiv_ref.canonical_url

    downloaded = trafilatura.fetch_url(source)
    if not downloaded:
        raise ValueError(f"Failed to fetch {source}")

    text = trafilatura.extract(downloaded)
    if not text:
        raise ValueError(f"Failed to extract content from {source}")

    traf_meta: dict = {}
    try:
        traf_json = trafilatura.extract(downloaded, output_format="json", with_metadata=True)
        if traf_json:
            traf_meta = json.loads(traf_json)
    except Exception:
        pass

    return text, traf_meta, source


def capture_url(db_path: Path, source: str, storage_dir: Path) -> str:
    """Fetch source text, extract content, store with embeddings, return hash."""
    from .chunk import chunk_text
    from .embed import embed_texts
    from .extract import extract_metadata

    text, hints, source_uri = fetch_text(source)

    conn = connect(db_path)
    h = store_capture(conn, text, source_uri, content_type="text/plain", storage_dir=storage_dir)

    # LLM extraction with hints
    meta = extract_metadata(
        text,
        url=source_uri,
        hints={
            "title": hints.get("title"),
            "author": hints.get("author"),
            "date": hints.get("date"),
        },
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

    # Clear any existing embeddings (for re-capture case)
    delete_embeddings(conn, h)

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
