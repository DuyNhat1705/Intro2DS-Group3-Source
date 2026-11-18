r"""Cross-version element deduplication and deterministic ``element_id`` assignment.

* each version's root is its own element ``<paper_key>#root<v>`` (content = cleaned title);
  roots are never deduplicated;
* every other element is identified by ``(kind, cleaned_content, occurrence_index)`` where the
  occurrence index counts earlier nodes of the same ``(kind, content)`` *within the version*;
  identical content across versions therefore shares one ``element_id`` (full-text dedup after
  cleanup) while each version stays a true tree (one parent per node per version);
* container content: headings -> title; ``abstract``/``list``/``item``/``block`` -> the cleaned
  text of their subtree's leaves;
* ``element_id = <paper_key>#<prefix><n>``, ``n`` in order of first appearance (v1 -> vN, pre-order);
* hierarchy rows are written in pre-order, which preserves sibling order without a column.
"""

from __future__ import annotations

from . import config as C
from .hierarchy import Node

_LEAVES = ("sentence", "formula", "figure", "table")
_SUBTREE_KINDS = ("abstract", "list", "item", "block")


def _subtree_text(node: Node) -> str:
    parts = []
    stack = list(reversed(node.children))
    while stack:
        n = stack.pop()
        if n.kind in _LEAVES:
            if n.content:
                parts.append(n.content)
        else:
            stack.extend(reversed(n.children))
    return " ".join(parts)


def build_tables(paper_key: str, arxiv_id: str, trees: dict[int, Node]) -> tuple[list[dict], list[dict]]:
    ids: dict[tuple[str, str, int], str] = {}
    counters: dict[str, int] = {}
    elements: list[dict] = []
    hierarchy: list[dict] = []
    for v in sorted(trees):
        root = trees[v]
        root_id = f"{paper_key}#root{v}"
        elements.append({"element_id": root_id, "arxiv_id": arxiv_id, "kind": "document",
                         "content": root.content or ""})
        occ: dict[tuple[str, str], int] = {}
        stack: list[tuple[Node, str]] = [(c, root_id) for c in reversed(root.children)]
        while stack:
            node, parent = stack.pop()
            kind = node.kind
            if kind in _SUBTREE_KINDS:
                content = _subtree_text(node)
            else:
                content = node.content or ""
            ck = (kind, content)
            k = occ.get(ck, 0)
            occ[ck] = k + 1
            key = (kind, content, k)
            eid = ids.get(key)
            if eid is None:
                prefix = C.ID_PREFIX.get(kind, "x")
                n = counters.get(prefix, 0) + 1
                counters[prefix] = n
                eid = f"{paper_key}#{prefix}{n}"
                ids[key] = eid
                elements.append({"element_id": eid, "arxiv_id": arxiv_id, "kind": kind,
                                 "content": content})
            hierarchy.append({"arxiv_id": arxiv_id, "version": v, "element_id": eid,
                              "parent_id": parent})
            for c in reversed(node.children):
                stack.append((c, eid))
    return elements, hierarchy
