from mneh.extract import normalize_extracted_metadata


def test_normalize_extracted_metadata_defaults_for_invalid_payload():
    out = normalize_extracted_metadata("bad")
    assert out == {
        "title": None,
        "author": None,
        "date": None,
        "summary": None,
        "handles": [],
    }


def test_normalize_extracted_metadata_cleans_handles_and_fields():
    out = normalize_extracted_metadata(
        {
            "title": "  My Title  ",
            "author": " Ada ",
            "date": " 2024-01-01 ",
            "summary": "  short summary ",
            "handles": [" one ", "", None, "two"],
        }
    )

    assert out["title"] == "My Title"
    assert out["author"] == "Ada"
    assert out["date"] == "2024-01-01"
    assert out["summary"] == "short summary"
    assert out["handles"] == ["one", "two"]
