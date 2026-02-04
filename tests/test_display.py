from mneh.display import collapse_hits_by_document, filter_by_score


def test_collapse_prefers_chunk_for_snippet_even_if_handle_scores_higher():
    h = "deadbeef" * 8
    results = [
        {
            "hash": h,
            "title": "T",
            "source_uri": "u",
            "summary": "s",
            "text": "HANDLE TEXT",
            "rrf_score": 0.9,
            "hit_kind": "handle",
            "hit_index": 2,
        },
        {
            "hash": h,
            "title": "T",
            "source_uri": "u",
            "summary": "s",
            "text": "this is a chunk about foo bar baz",
            "rrf_score": 0.8,
            "hit_kind": "chunk",
            "hit_index": 7,
        },
    ]

    docs = collapse_hits_by_document(results)
    assert len(docs) == 1
    d = docs[0]
    assert d.score == 0.9
    assert d.best_hit_kind == "handle"
    assert d.best_hit_index == 2
    assert d.snippet_kind == "chunk"
    assert d.snippet_index == 7


def test_filter_by_score_drops_bottom_fraction():
    docs = [
        type("D", (), {"score": float(i)})()  # lightweight stub
        for i in range(1, 11)
    ]
    filtered, thresh = filter_by_score(docs, drop_bottom_frac=0.1, min_score=None)
    assert thresh is not None
    assert len(filtered) == 9
