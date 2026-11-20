"""
Shared Types for BibTeX Extraction
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BibtexEntry:
    """Một mục tài liệu tham khảo đã chuẩn hóa."""
    arxiv_id: str
    citation_key: str         # vd: "vaswani2017attention"
    entry_type: str           # "article" | "inproceedings" | "book" | "misc"
    title: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: Optional[str] = None
    journal: Optional[str] = None
    url: Optional[str] = None
    arxiv_id_hint: Optional[str] = None     # Bóc được arXiv ID gốc (phục vụ Milestone 3)
    confidence_score: float = 1.0           # Điểm chất lượng bóc tách (0.0 - 1.0)
    source_method: str = "regex"            # "bib_file", "regex", "crossref", "llm"

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
        }
