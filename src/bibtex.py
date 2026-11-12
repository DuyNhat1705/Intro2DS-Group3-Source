"""Interfaces for extracting and unifying bibliography entries."""

from dataclasses import dataclass
from typing import TypedDict

from utils import GatherTex


@dataclass
class BibEntry:
    citation_key: str
    entry_type: str
    fields: dict[str, str]


class BibtexRow(TypedDict):
    arxiv_id: str
    citation_key: str
    entry_type: str
    title: str
    authors: list[str]
    year: int | None
    doi: str
    journal: str
    url: str


def extract_bibtex(tex: GatherTex) -> list[BibEntry]:
    """Read the referenced .bib files and convert reachable LaTeX bibitems."""
    pass


def merge_bibtex(
    entries_by_version: dict[int, list[BibEntry]],
) -> tuple[list[BibEntry], dict[int, dict[str, str]]]:
    """Return unified entries and {version: {original_key: canonical_key}}."""
    pass


def build_bibtex_rows(arxiv_id: str, entries: list[BibEntry]) -> list[BibtexRow]:
    """Project unified entries into the final bibtex columns."""
    pass