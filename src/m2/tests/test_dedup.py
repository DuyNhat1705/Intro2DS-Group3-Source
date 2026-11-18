r"""Tests of cross-version element deduplication (prompt §5.8) and the statement's example."""

from __future__ import annotations

from src.m2.dedup import build_tables
from src.m2.hierarchy import Node
from src.m2.pipeline import parse_version_files


def N(kind, content="", children=()):
    n = Node(kind=kind, content=content)
    n.children.extend(children)
    return n


def test_statement_example():
    # v1: root1 -> h(Introduction) -> s ; v2: root2 -> h(Background and model) -> s (same sentence)
    s = "We study the citation matching problem"
    v1 = N("document", "Version 1 root", [N("section", "Introduction", [N("sentence", s)])])
    v2 = N("document", "Version 2 root", [N("section", "Background and model", [N("sentence", s)])])
    elements, hierarchy = build_tables("2310-12345", "2310.12345", {1: v1, 2: v2})
    el = {e["element_id"]: (e["kind"], e["content"]) for e in elements}
    assert el["2310-12345#root1"] == ("document", "Version 1 root")
    assert el["2310-12345#root2"] == ("document", "Version 2 root")
    assert el["2310-12345#h1"] == ("section", "Introduction")
    assert el["2310-12345#h2"] == ("section", "Background and model")
    assert el["2310-12345#s1"] == ("sentence", s)
    assert len(elements) == 5
    edges = [(h["version"], h["element_id"], h["parent_id"]) for h in hierarchy]
    assert edges == [
        (1, "2310-12345#h1", "2310-12345#root1"),
        (1, "2310-12345#s1", "2310-12345#h1"),
        (2, "2310-12345#h2", "2310-12345#root2"),
        (2, "2310-12345#s1", "2310-12345#h2"),
    ]
    assert all(e["arxiv_id"] == "2310.12345" for e in elements)


def test_occurrence_index_keeps_tree_property():
    # The same sentence twice in one version -> two elements, each with one parent.
    v1 = N("document", "T", [N("section", "A", [N("sentence", "Same.")]),
                             N("section", "B", [N("sentence", "Same.")])])
    v2 = N("document", "T", [N("section", "A", [N("sentence", "Same.")])])
    elements, hierarchy = build_tables("p", "p", {1: v1, 2: v2})
    sents = [e for e in elements if e["kind"] == "sentence"]
    assert len(sents) == 2
    v1_edges = [(h["element_id"], h["parent_id"]) for h in hierarchy if h["version"] == 1]
    ids = [e for e, _p in v1_edges]
    assert len(ids) == len(set(ids))  # one parent per node per version
    v2_edges = [(h["element_id"], h["parent_id"]) for h in hierarchy if h["version"] == 2]
    assert ("p#s1", "p#h1") in v2_edges  # first occurrence shared with v1


def test_container_content_is_subtree_text_and_ids_by_prefix():
    lst = N("list", "", [N("item", "", [N("sentence", "One.")]), N("item", "", [N("sentence", "Two.")])])
    v1 = N("document", "T", [N("section", "S", [lst, N("formula", "\\begin{equation}x\\end{equation}"),
                                                 N("figure", "Cap."), N("table", "Tab.")])])
    elements, _h = build_tables("p", "p", {1: v1})
    by = {e["element_id"]: e for e in elements}
    assert by["p#l1"]["content"] == "One. Two."
    assert by["p#i1"]["content"] == "One."
    assert by["p#i2"]["content"] == "Two."
    assert {"p#f1", "p#fig1", "p#tab1", "p#h1", "p#s1", "p#s2"} <= set(by)


def test_roots_never_deduplicated_and_title_fallback():
    files = {"main.tex": "\\documentclass{article}\\begin{document}Hello world.\\end{document}"}
    t1 = parse_version_files(files, meta_title="Meta Title").tree
    t2 = parse_version_files(files, meta_title="Meta Title").tree
    elements, hierarchy = build_tables("p", "p", {1: t1, 2: t2})
    roots = [e for e in elements if e["kind"] == "document"]
    assert [r["element_id"] for r in roots] == ["p#root1", "p#root2"]
    assert roots[0]["content"] == "Meta Title"
    assert len([e for e in elements if e["kind"] == "sentence"]) == 1


def test_hierarchy_preorder():
    v1 = N("document", "T", [N("section", "A", [N("sentence", "a1."), N("subsection", "A1", [N("sentence", "x.")])]),
                             N("section", "B", [N("sentence", "b1.")])])
    _e, h = build_tables("p", "p", {1: v1})
    assert [r["element_id"] for r in h] == ["p#h1", "p#s1", "p#h2", "p#s2", "p#h3", "p#s3"]
