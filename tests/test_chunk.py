import pytest

tiktoken = pytest.importorskip("tiktoken")

from mneh.chunk import chunk_text, count_tokens, PARA_SEP_TOKENS


def test_semantic_merges_paragraphs_up_to_budget():
    p1 = "alpha beta gamma."
    p2 = "delta epsilon zeta."
    p3 = "eta theta iota."
    text = f"{p1}\n\n{p2}\n\n{p3}"

    t1 = count_tokens(p1)
    t2 = count_tokens(p2)
    t3 = count_tokens(p3)

    target = t1 + PARA_SEP_TOKENS + t2
    assert target < t1 + PARA_SEP_TOKENS + t2 + PARA_SEP_TOKENS + t3

    chunks = chunk_text(text, target_tokens=target, mode="semantic")
    assert chunks == [f"{p1}\n\n{p2}", p3]

    for c in chunks:
        assert count_tokens(c) <= target


def test_long_paragraph_falls_back_to_token_split():
    p_short = "one two three."
    p_long = ("x " * 500).strip()
    text = f"{p_short}\n\n{p_long}\n\n{p_short}"

    target = 80
    chunks = chunk_text(text, target_tokens=target, overlap=10, mode="semantic")

    # First chunk should be just the first short paragraph.
    assert chunks[0].strip() == p_short

    # Long paragraph must be split into >1 chunks.
    long_chunks = [c for c in chunks[1:-1]]
    assert len(long_chunks) >= 2

    # Final chunk should be last short paragraph.
    assert chunks[-1].strip() == p_short

    for c in chunks:
        assert count_tokens(c) <= target
