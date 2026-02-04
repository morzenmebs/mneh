"""CLI-facing display helpers.

This module is intentionally ANSI-free: it formats readable, wrapped text for
terminal output.
"""

from __future__ import annotations

import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable


_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "to", "of", "in", "on", "for", "with", "without", "as", "at", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that", "these",
    "those", "i", "you", "we", "they", "he", "she", "them", "his", "her", "our",
    "your", "their", "my", "me",
}


def extract_query_terms(query: str) -> list[str]:
    r"""Cheap keywordization for highlighting.

    Keeps \w+ tokens, lowercases, drops stopwords and very short tokens.
    """
    words = re.findall(r"\w+", query.lower())
    terms = [w for w in words if len(w) >= 3 and w not in _STOPWORDS]
    # Prefer longer terms first to reduce partial overlap.
    terms.sort(key=len, reverse=True)
    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for t in terms:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _first_match_span(text: str, terms: Iterable[str]) -> tuple[int, int] | None:
    for t in terms:
        m = re.search(rf"\b{re.escape(t)}\b", text, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def highlight_bracket(text: str, terms: Iterable[str]) -> str:
    """Bracket-and-uppercase matches: foo -> [FOO]."""
    out = text
    for t in terms:
        rx = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
        out = rx.sub(lambda m: f"[{m.group(0).upper()}]", out)
    return out


def excerpt_around_terms(text: str, terms: Iterable[str], *, max_chars: int) -> str:
    """Take a window around the first query-term match (or leading text)."""
    s = " ".join((text or "").split())
    if not s:
        return ""

    span = _first_match_span(s, terms)
    if span is None or len(s) <= max_chars:
        return s[:max_chars].strip()

    start, end = span
    half = max_chars // 2
    lo = max(0, start - half)
    hi = min(len(s), end + half)

    # Nudge to word boundaries.
    if lo > 0:
        lo = s.rfind(" ", 0, lo) + 1 or 0
    if hi < len(s):
        nxt = s.find(" ", hi)
        if nxt != -1:
            hi = nxt

    prefix = "…" if lo > 0 else ""
    suffix = "…" if hi < len(s) else ""
    return (prefix + s[lo:hi].strip() + suffix).strip()


@dataclass(frozen=True)
class DisplayResult:
    hash: str
    title: str
    source_uri: str
    summary: str
    score: float
    best_hit_kind: str
    best_hit_index: int
    snippet_kind: str
    snippet_index: int
    snippet_text: str
    more_chunks: int
    more_handles: int


def collapse_hits_by_document(results: list[dict[str, Any]]) -> list[DisplayResult]:
    """Collapse multiple hit rows into one per document.

    Selection rules:
      - score = best rrf_score across hits
      - snippet prefers the best *chunk* hit if available, else the best overall hit
      - counts of suppressed hits are tracked for display
    """
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("hash"):
            by_hash[str(r["hash"])].append(r)

    docs: list[DisplayResult] = []
    for h, hits in by_hash.items():
        hits_sorted = sorted(hits, key=lambda x: float(x.get("rrf_score") or 0.0), reverse=True)
        best = hits_sorted[0]

        best_chunk = next((x for x in hits_sorted if x.get("hit_kind") == "chunk"), None)
        snippet = best_chunk or best

        chunk_hits = [x for x in hits_sorted if x.get("hit_kind") == "chunk"]
        handle_hits = [x for x in hits_sorted if x.get("hit_kind") == "handle"]

        more_chunks = max(0, len(chunk_hits) - (1 if snippet.get("hit_kind") == "chunk" else 0))
        more_handles = max(0, len(handle_hits) - (1 if snippet.get("hit_kind") == "handle" else 0))

        docs.append(
            DisplayResult(
                hash=h,
                title=(best.get("title") or "Untitled"),
                source_uri=(best.get("source_uri") or ""),
                summary=(best.get("summary") or ""),
                score=float(best.get("rrf_score") or 0.0),
                best_hit_kind=str(best.get("hit_kind")),
                best_hit_index=int(best.get("hit_index") or 0),
                snippet_kind=str(snippet.get("hit_kind")),
                snippet_index=int(snippet.get("hit_index") or 0),
                snippet_text=str(snippet.get("text") or ""),
                more_chunks=more_chunks,
                more_handles=more_handles,
            )
        )

    docs.sort(key=lambda d: d.score, reverse=True)
    return docs


def filter_by_score(
    docs: list[DisplayResult],
    *,
    drop_bottom_frac: float = 0.10,
    min_score: float | None = None,
) -> tuple[list[DisplayResult], float | None]:
    """Filter docs by a score floor.

    If drop_bottom_frac is set and we have >=10 docs, we drop the bottom fraction
    by score (doc-level).

    Returns (filtered_docs, applied_threshold).
    """
    if not docs:
        return [], None

    threshold: float | None = None

    if drop_bottom_frac and len(docs) >= 10:
        frac = min(max(drop_bottom_frac, 0.0), 0.99)
        scores = sorted(d.score for d in docs)
        cut = int(len(scores) * frac)
        cut = min(max(cut, 0), len(scores) - 1)
        threshold = scores[cut]

    if min_score is not None:
        threshold = max(threshold or min_score, min_score)

    if threshold is None:
        return docs, None

    return [d for d in docs if d.score >= threshold], threshold


def format_result(
    doc: DisplayResult,
    *,
    query: str,
    snippet_chars: int = 360,
    width: int = 92,
) -> str:
    terms = extract_query_terms(query)
    snippet = excerpt_around_terms(doc.snippet_text, terms, max_chars=snippet_chars)
    snippet = highlight_bracket(snippet, terms)

    more = []
    if doc.more_chunks:
        more.append(f"+{doc.more_chunks} more chunks")
    if doc.more_handles:
        more.append(f"+{doc.more_handles} more handles")
    more_s = f"  ({', '.join(more)})" if more else ""

    header = (
        f"{doc.title} [{doc.hash[:8]}]  score={doc.score:.6f}"
        f"  (best={doc.best_hit_kind}:{doc.best_hit_index}, snippet={doc.snippet_kind}:{doc.snippet_index})"
        f"{more_s}"
    )

    lines = [header]
    if doc.summary:
        lines.append("  " + textwrap.fill(" ".join(doc.summary.split()), width=width, subsequent_indent="  "))
    if doc.source_uri:
        lines.append(f"  {doc.source_uri}")
    if snippet:
        lines.append("  " + textwrap.fill(snippet, width=width, subsequent_indent="  "))
    return "\n".join(lines)
