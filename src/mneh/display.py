"""Display formatting for CLI output.

Produces clean, columnar text output. No ANSI colors (keep it simple).
"""

from __future__ import annotations

import json
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Iterable

from .storage import CaptureInfo, CaptureDetail, HitKey


# -----------------------------------------------------------------------------
# Text utilities
# -----------------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "without", "as", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "i", "you", "we", "they", "he", "she", "them", "his",
    "her", "our", "your", "their", "my", "me",
}


def extract_query_terms(query: str) -> list[str]:
    """Extract meaningful terms from a query for highlighting."""
    words = re.findall(r"\w+", query.lower())
    terms = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    terms.sort(key=len, reverse=True)
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def highlight_bracket(text: str, terms: Iterable[str]) -> str:
    """Bracket-and-uppercase matches: foo -> [FOO]."""
    out = text
    for t in terms:
        rx = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        out = rx.sub(lambda m: f"[{m.group(0).upper()}]", out)
    return out


def excerpt_around_terms(text: str, terms: Iterable[str], *, max_chars: int = 200) -> str:
    """Extract a snippet around the first query-term match."""
    s = " ".join((text or "").split())
    if not s:
        return ""

    # Find first match
    span = None
    for t in terms:
        m = re.search(rf"\b{re.escape(t)}\b", s, flags=re.IGNORECASE)
        if m:
            span = (m.start(), m.end())
            break

    if span is None or len(s) <= max_chars:
        return s[:max_chars].strip()

    start, end = span
    half = max_chars // 2
    lo = max(0, start - half)
    hi = min(len(s), end + half)

    # Nudge to word boundaries
    if lo > 0:
        lo = s.rfind(" ", 0, lo) + 1 or 0
    if hi < len(s):
        nxt = s.find(" ", hi)
        if nxt != -1:
            hi = nxt

    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(s) else ""
    return (prefix + s[lo:hi].strip() + suffix).strip()


def truncate(s: str, max_len: int) -> str:
    """Truncate string with ellipsis if needed."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# -----------------------------------------------------------------------------
# List formatting
# -----------------------------------------------------------------------------


def format_list(
    captures: list[CaptureInfo],
    *,
    show_urls: bool = False,
    verbose: bool = False,
) -> str:
    """Format capture list for terminal output.

    Default:     hash  date  title
    -u:          hash  date  title + url on second line
    -v:          hash  date  chunks  handles  title
    -uv:         hash  date  chunks  handles  title + url on second line
    """
    if not captures:
        return "No captures."

    lines = []
    for c in captures:
        h = c.hash[:8]
        date = (c.captured_at or "")[:10]  # YYYY-MM-DD
        title = c.title or "Untitled"

        if verbose:
            # hash  date  ch  hd  title
            line = f"{h}  {date}  {c.chunk_count:3d} ch  {c.handle_count:3d} hd  {title}"
        else:
            # hash  date  title
            line = f"{h}  {date}  {title}"

        lines.append(line)

        if show_urls:
            # URL on second line, aligned with date column
            indent = " " * 10  # 8 (hash) + 2 (spaces)
            lines.append(f"{indent}{c.source_uri}")

    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Show formatting
# -----------------------------------------------------------------------------


def format_show(
    detail: CaptureDetail,
    *,
    verbose: int = 0,  # 0=normal, 1=show handles, 2=show handles+chunks
) -> str:
    """Format a single capture for display.

    verbose=0: metadata only
    verbose=1: metadata + handles list
    verbose=2: metadata + handles + chunk previews
    """
    lines = []

    lines.append(f"hash:     {detail.hash}")
    lines.append(f"title:    {detail.title or '(none)'}")
    if detail.author:
        lines.append(f"author:   {detail.author}")
    if detail.date:
        lines.append(f"date:     {detail.date}")
    lines.append(f"url:      {detail.source_uri}")
    lines.append(f"captured: {detail.captured_at}")
    if detail.summary:
        wrapped = textwrap.fill(detail.summary, width=70, initial_indent="summary:  ", subsequent_indent="          ")
        lines.append(wrapped)

    lines.append("")
    lines.append(f"chunks:   {len(detail.chunks)}")
    lines.append(f"handles:  {len(detail.handles)}")

    if verbose >= 1 and detail.handles:
        lines.append("")
        lines.append(f"handles ({len(detail.handles)}):")
        for i, h in enumerate(detail.handles, start=1):
            lines.append(f"  {i:02d}  {h}")

    if verbose >= 2 and detail.chunks:
        lines.append("")
        lines.append(f"chunks ({len(detail.chunks)}):")
        for i, chunk in enumerate(detail.chunks):
            preview = " ".join(chunk.split())[:100]
            lines.append(f"  [{i}] {preview}...")

    return "\n".join(lines)


def format_show_json(detail: CaptureDetail) -> str:
    """Format capture as JSON."""
    return json.dumps(asdict(detail), indent=2, ensure_ascii=False)


# -----------------------------------------------------------------------------
# Search result formatting
# -----------------------------------------------------------------------------


@dataclass
class DocResult:
    """A document-level search result with aggregated hits."""

    hash: str
    title: str
    source_uri: str
    summary: str
    score: float
    handle_hits: list[tuple[float, int, str]]  # (score, index, text)
    chunk_hits: list[tuple[float, int, str]]   # (score, index, text)


def aggregate_search_results(
    results: list[dict[str, Any]],
    hit_scores: dict[HitKey, float],
) -> list[DocResult]:
    """Aggregate raw search hits into per-document results."""
    by_hash: dict[str, dict[str, Any]] = {}
    handle_hits: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    chunk_hits: dict[str, list[tuple[float, int, str]]] = defaultdict(list)

    for r in results:
        h = r.get("hash")
        if not h:
            continue

        if h not in by_hash:
            by_hash[h] = r

        kind = r.get("hit_kind")
        idx = r.get("hit_index", 0)
        text = r.get("text", "")
        score = r.get("rrf_score", 0.0)

        if kind == "handle":
            handle_hits[h].append((score, idx, text))
        elif kind == "chunk":
            chunk_hits[h].append((score, idx, text))

    docs = []
    for h, r in by_hash.items():
        hh = sorted(handle_hits.get(h, []), key=lambda x: x[0], reverse=True)
        ch = sorted(chunk_hits.get(h, []), key=lambda x: x[0], reverse=True)

        # Doc score = max of all hits
        all_scores = [s for s, _, _ in hh] + [s for s, _, _ in ch]
        doc_score = max(all_scores) if all_scores else 0.0

        docs.append(
            DocResult(
                hash=h,
                title=r.get("title") or "Untitled",
                source_uri=r.get("source_uri") or "",
                summary=r.get("summary") or "",
                score=doc_score,
                handle_hits=hh,
                chunk_hits=ch,
            )
        )

    docs.sort(key=lambda d: d.score, reverse=True)
    return docs


def filter_noise(docs: list[DocResult], min_score: float = 0.015) -> list[DocResult]:
    """Filter out results below noise threshold."""
    return [d for d in docs if d.score >= min_score]


def format_search(
    docs: list[DocResult],
    query: str,
    *,
    verbose: bool = False,
    limit: int = 20,
) -> str:
    """Format search results.

    Default: rank, title, url, summary, best chunk snippet
    Verbose: adds handle hits and chunk hits with scores
    """
    if not docs:
        return "No results."

    terms = extract_query_terms(query)
    lines = []

    for i, doc in enumerate(docs[:limit], start=1):
        if verbose:
            # Verbose format with score in header
            lines.append(f"[{i}] {doc.title:<50} score: {doc.score:.4f}")
        else:
            lines.append(f"[{i}] {doc.title}")

        lines.append(f"    {doc.source_uri}")

        if doc.summary:
            wrapped = textwrap.fill(doc.summary, width=76, initial_indent="    ", subsequent_indent="    ")
            lines.append(wrapped)

        if verbose:
            # Show handle hits
            if doc.handle_hits:
                lines.append("")
                lines.append(f"    handles ({len(doc.handle_hits)} hits):")
                for score, idx, text in doc.handle_hits[:5]:
                    lines.append(f"      {score:.4f}  {text}")

            # Show chunk hits
            if doc.chunk_hits:
                lines.append("")
                lines.append(f"    chunks ({len(doc.chunk_hits)} hits):")
                for score, idx, text in doc.chunk_hits[:3]:
                    snippet = excerpt_around_terms(text, terms, max_chars=120)
                    snippet = highlight_bracket(snippet, terms)
                    lines.append(f"      {score:.4f}  [{idx}] \"{snippet}\"")
        else:
            # Just show best chunk snippet
            if doc.chunk_hits:
                _, idx, text = doc.chunk_hits[0]
                snippet = excerpt_around_terms(text, terms, max_chars=200)
                snippet = highlight_bracket(snippet, terms)
                wrapped = textwrap.fill(f'"{snippet}"', width=76, initial_indent="    ", subsequent_indent="    ")
                lines.append(wrapped)

        lines.append("")

    return "\n".join(lines)


def format_search_json(docs: list[DocResult]) -> str:
    """Format search results as JSON."""
    out = []
    for d in docs:
        out.append({
            "hash": d.hash,
            "title": d.title,
            "source_uri": d.source_uri,
            "summary": d.summary,
            "score": d.score,
            "handle_hits": [{"score": s, "index": i, "text": t} for s, i, t in d.handle_hits],
            "chunk_hits": [{"score": s, "index": i, "text": t} for s, i, t in d.chunk_hits],
        })
    return json.dumps(out, indent=2, ensure_ascii=False)
