"""Token-aware text chunking.

Default strategy is "semantic-ish": split on paragraph boundaries ("\n\n" or more
blank lines), then merge adjacent paragraphs up to a token budget.

If any single paragraph exceeds the budget, we fall back to the token-based
splitter for that paragraph.
"""

from __future__ import annotations

import re
import tiktoken

ENCODER = tiktoken.get_encoding("cl100k_base")
PARA_SEP_TOKENS = len(ENCODER.encode("\n\n"))

MAX_CHUNK_TOKENS = 500  # target size
OVERLAP_TOKENS = 50  # overlap between chunks (token-based fallback)
HARD_LIMIT_TOKENS = 7500  # safety


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))


def decode_tokens(tokens: list[int]) -> str:
    return ENCODER.decode(tokens)


def encode_text(text: str) -> list[int]:
    return ENCODER.encode(text)


def _chunk_text_tokenwise(text: str, target_tokens: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by token count."""
    tokens = encode_text(text)

    if len(tokens) <= target_tokens:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(tokens):
        end = start + target_tokens
        chunk_tokens = tokens[start:end]
        chunks.append(decode_tokens(chunk_tokens))

        # Move forward by (target - overlap) to create overlap
        start += max(1, target_tokens - overlap)

        # Avoid tiny final chunk
        if len(tokens) - start < overlap * 2:
            if start < len(tokens):
                chunks[-1] = decode_tokens(tokens[start - (target_tokens - overlap) :])
            break

    return chunks


_PARABREAK_RE = re.compile(r"\n\s*\n+")


def _split_paragraphs(text: str) -> list[str]:
    """Split on blank-line paragraph breaks, preserving intra-paragraph newlines."""
    parts = [p.strip() for p in _PARABREAK_RE.split(text) if p.strip()]
    return parts


def chunk_text(
    text: str,
    target_tokens: int = MAX_CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
    *,
    mode: str = "semantic",
) -> list[str]:
    """Chunk text for embedding / retrieval.

    mode:
      - "semantic": split on paragraphs, then merge to token budget. Long paragraphs
        fall back to token-based splitting.
      - "token": pure token-based splitting with overlap.
    """
    if mode not in {"semantic", "token"}:
        raise ValueError(f"Unknown chunking mode: {mode}")

    if mode == "token":
        return _chunk_text_tokenwise(text, target_tokens=target_tokens, overlap=overlap)

    # Semantic-ish: paragraph boundaries first.
    paras = _split_paragraphs(text)
    if not paras:
        return []

    out: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    def flush_buf() -> None:
        nonlocal buf, buf_tokens
        if buf:
            out.append("\n\n".join(buf).strip())
            buf = []
            buf_tokens = 0

    for p in paras:
        ptoks = count_tokens(p)

        # Paragraph too large: flush buffer, then token-split this paragraph.
        if ptoks > target_tokens:
            flush_buf()
            out.extend(_chunk_text_tokenwise(p, target_tokens=target_tokens, overlap=overlap))
            continue

        # Otherwise, try to merge into the current buffer.
        sep_toks = 0 if not buf else PARA_SEP_TOKENS
        if buf_tokens + sep_toks + ptoks <= target_tokens:
            if buf:
                buf_tokens += sep_toks
            buf.append(p)
            buf_tokens += ptoks
        else:
            flush_buf()
            buf.append(p)
            buf_tokens = ptoks

    flush_buf()
    return out
