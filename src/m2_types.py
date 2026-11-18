"""
Shared Types for BibTeX Extraction
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BibtexEntry:
    """One normalized bibliography entry."""
    arxiv_id: str
    citation_key: str         # vd: "vaswani2017attention"
    entry_type: str           # "article" | "inproceedings" | "book" | "misc"
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    arxiv_id_hint: Optional[str] = None     # Extracted original arXiv ID (for Milestone 3)
    confidence_score: float = 1.0           # Extraction quality score (0.0 - 1.0)
    source_method: str = "regex"            # "bib_file", "regex", "crossref", "llm"
    # \bibitem text with collapsed whitespace (None for .bib entries), for post-processing.
    raw_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arxiv_id": self.arxiv_id,
            "citation_key": self.citation_key,
            "entry_type": self.entry_type,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": self.doi,
            "journal": self.journal,
            "url": self.url,
            "arxiv_id_hint": self.arxiv_id_hint,
            "confidence_score": self.confidence_score,
            "source_method": self.source_method,
            "raw_text": self.raw_text,
        }
