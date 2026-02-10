from mneh.capture import ArxivRef, latex_to_plaintext, normalize_handles, parse_arxiv_reference


def test_parse_prefixed_new_style_id_with_version():
    assert parse_arxiv_reference("arXiv:2002.12327v3") == ArxivRef("2002.12327", "v3")


def test_parse_arxiv_abs_url():
    assert parse_arxiv_reference("https://arxiv.org/abs/1706.03762") == ArxivRef("1706.03762")


def test_parse_arxiv_pdf_url_with_version():
    assert parse_arxiv_reference("https://arxiv.org/pdf/2002.12327v3.pdf") == ArxivRef(
        "2002.12327", "v3"
    )


def test_parse_old_style_id():
    assert parse_arxiv_reference("hep-th/9901001v2") == ArxivRef("hep-th/9901001", "v2")


def test_parse_non_arxiv_value():
    assert parse_arxiv_reference("https://example.com/article") is None


def test_latex_to_plaintext_keeps_readable_content():
    tex = r"""
    \documentclass{article}
    \title{A Great Paper}
    \author{Ada Lovelace}
    \begin{document}
    \maketitle
    This is the abstract with \textbf{bold words}.
    \section{Introduction}
    We show that $E=mc^2$ in simple settings.
    \end{document}
    """

    text = latex_to_plaintext(tex)
    assert "A Great Paper" in text
    assert "This is the abstract with bold words." in text
    assert "Introduction" in text
    assert "E=mc^2" not in text


def test_normalize_handles_uses_model_handles_when_present():
    handles = normalize_handles([" alpha beta ", "Alpha Beta", "gamma delta"], "unused")
    assert handles == ["alpha beta", "gamma delta"]


def test_normalize_handles_falls_back_when_model_omits_handles():
    text = """
    Graph neural networks for molecules
    Message passing architecture and atom featurization
    Benchmarking on quantum chemistry datasets
    Error analysis and ablation studies
    """
    handles = normalize_handles([], text)
    assert handles
    assert any("graph neural networks" in h.lower() for h in handles)
