"""Interfaces for per-version parsing and publication-wide element tables."""

from dataclasses import dataclass
from typing import TypedDict


@dataclass
class Node:
    local_id: str
    parent_local_id: str | None
    kind: str
    content: str


class ElementRow(TypedDict):
    element_id: str
    arxiv_id: str
    kind: str
    content: str


class HierarchyRow(TypedDict):
    arxiv_id: str
    version: int
    element_id: str
    parent_id: str


def construct_hierarchy(cleaned_tex: str) -> list[Node]:
    """Build one version's tree, including a root with parent_local_id=None."""
    raise NotImplementedError("MĐạt implements construct_hierarchy")


def rewrite_citations(nodes: list[Node], key_map: dict[str, str]) -> list[Node]:
    """Apply one version's cite-key mapping while preserving local IDs and edges."""
    raise NotImplementedError("Nhất implements rewrite_citations")


def build_element_tables(
    arxiv_id: str, nodes_by_version: dict[int, list[Node]]
) -> tuple[list[ElementRow], list[HierarchyRow]]:
    """Share element IDs across versions and retain version-specific parent edges."""
    raise NotImplementedError("Nhất implements build_element_tables")
