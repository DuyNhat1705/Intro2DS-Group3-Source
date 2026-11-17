"""Khử trùng element theo khoảng trắng và cache SaT."""

from unittest.mock import patch

from conftest import fake_split
from src.hierarchy import Node, build_element_table, construct_hierarchy, normalize_whitespace


def test_normalize_whitespace_collapses_outside_verbatim():
    assert normalize_whitespace("  a \n b\t\tc  ") == "a b c"
    verbatim = "\\begin{verbatim}x  =\n  1\\end{verbatim}"
    assert normalize_whitespace(f"see\n{verbatim}  and") == f"see {verbatim} and"
    assert normalize_whitespace("use \\verb|a  b| here") == "use \\verb|a  b| here"


def test_sentences_differing_only_in_line_wrapping_share_one_element():
    v1 = [Node("n0", None, "document", ""), Node("n1", "n0", "sentence", "We study attention\nin transformers.")]
    v2 = [Node("n0", None, "document", ""), Node("n1", "n0", "sentence", "We study attention in  transformers.")]
    elements, hierarchy = build_element_table("2609.00001", {1: v1, 2: v2})
    sentences = [row for row in elements if row["kind"] == "sentence"]
    assert len(sentences) == 1
    assert sentences[0]["content"] == "We study attention in transformers."
    edges = [row for row in hierarchy if row["element_id"] == sentences[0]["element_id"]]
    assert [row["version"] for row in edges] == [1, 2]


def test_containers_with_same_content_stay_distinct_within_a_version():
    r"""Hai list (nội dung rỗng) và hai heading "Proof" trong cùng phiên bản không gộp thành một nút."""
    tex = (
        r"\documentclass{article}\begin{document}\section{Proof}First proof."
        r"\begin{itemize}\item Alpha one.\item Beta two.\end{itemize}"
        r"\section{Proof}Second proof.\begin{itemize}\item Gamma three.\end{itemize}\end{document}"
    )
    v1 = _build(tex)
    elements, hierarchy = build_element_table("2609.00001", {1: v1, 2: _build(tex)})
    kinds = {e["element_id"]: e["kind"] for e in elements}
    content = {e["element_id"]: e["content"] for e in elements}
    assert sum(k == "list" for k in kinds.values()) == 2
    assert sum(k == "item" for k in kinds.values()) == 3
    assert sum(k == "heading" for k in kinds.values()) == 2
    # Mỗi câu nằm dưới đúng item của nó; mỗi item có đúng một con trong mỗi phiên bản.
    parent = {(h["version"], content[h["element_id"]]): h["parent_id"] for h in hierarchy}
    items = {parent[(1, s)] for s in ("Alpha one.", "Beta two.", "Gamma three.")}
    assert len(items) == 3 and all(kinds[i] == "item" for i in items)
    # Phiên bản giống hệt nhau vẫn dùng chung toàn bộ ID (khử trùng giữa các phiên bản không đổi).
    assert all(parent[(1, s)] == parent[(2, s)] for s in ("Alpha one.", "Beta two.", "Gamma three."))
    assert len(v1) == len(elements)


def _build(tex):
    with patch("src.hierarchy._sentence_model") as model:
        model.return_value.split.side_effect = fake_split
        return construct_hierarchy(tex)


def test_title_becomes_document_root_content():
    nodes = _build("\\title{Deep \\\\ Models\\thanks{Funded by X.}}\n\n\\section{Intro}\nWe study it.\n")
    assert nodes[0].kind == "document" and nodes[0].content == "Deep Models"
    assert not any("\\title" in node.content for node in nodes)


def test_unknown_environment_tags_do_not_leak_into_sentences():
    nodes = _build(
        "\\begin{openproblem}[Fellows et al.]\nIs it tractable?\n\\end{openproblem}\n\n"
        "\\begin{tcolorbox}{red}\nBoxed text.\n\\end{tcolorbox}\n"
    )
    sentences = [node.content for node in nodes if node.kind == "sentence"]
    assert not any("\\begin" in s or "\\end" in s or "{red}" in s for s in sentences)
    assert "Is it tractable?" in " ".join(sentences) and sentences[-1] == "Boxed text."


def test_repeated_fragments_give_one_warning_per_version():
    import warnings

    nodes = [
        Node("n0", None, "document", ""),
        Node("n1", "n0", "heading", "A"),
        Node("n2", "n1", "sentence", "Then"),
        Node("n3", "n0", "heading", "B"),
        Node("n4", "n3", "sentence", "Then"),
        Node("n5", "n0", "heading", "C"),
        Node("n6", "n5", "sentence", "Then"),
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_element_table("2609.00001", {1: nodes})
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1 and messages[0].startswith("1 element(s) have multiple parents in version 1")


def test_citation_keys():
    from src.hierarchy import citation_keys

    assert citation_keys("See \\cite{a, b} and \\citep[p.~2]{c}; \\url{x\\cite{no}}.") == {"a", "b", "c"}


def test_sat_runs_once_per_distinct_paragraph_across_versions():
    tex = "\\section{Intro}\nFirst paragraph here.\n\nSecond paragraph here.\n\n"
    with patch("src.hierarchy._sentence_model") as model:
        model.return_value.split.side_effect = fake_split
        first = construct_hierarchy(tex)
        calls = model.return_value.split.call_count
        second = construct_hierarchy(tex + "New paragraph in v2.\n")
    assert calls == 2
    # Phiên bản thứ hai chỉ gọi mô hình cho đoạn mới.
    assert model.return_value.split.call_count == 3
    assert [n.content for n in second[: len(first)]] == [n.content for n in first]
