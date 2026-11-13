"""
Milestone 2 — BibTeX Extraction Module
=======================================
Trích xuất và chuẩn hóa toàn bộ tài liệu tham khảo từ LaTeX sources (.tex và .bib).
Tích hợp:
  - Regex Engine bóc tách trường: Title, Authors, Year, DOI, Journal, Entry Type
  - CrossRef API Auto-Enrichment: Tự động đối soát và sửa lỗi tiêu đề / bù đắp tác giả
  - arXiv ID Hint Extractor: Bắt trước arXiv ID làm đòn bẩy cho Milestone 3
  - Confidence Scoring: Chấm điểm chất lượng dữ liệu bóc tách
  - LLM Fallback: Cơ chế dự phòng cho các cấu trúc LaTeX dị dạng
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from m2_types import BibtexEntry
except ImportError:
    from src.m2_types import BibtexEntry


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_BIBITEM_PATTERN = re.compile(
    r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})",
    re.DOTALL,
)

_BIBLIOGRAPHY_ENV = re.compile(
    r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def extract_bibtex(
    compilation_path: List[Path],
    arxiv_id: str,
) -> List[BibtexEntry]:
    """
    Trích xuất và chuẩn hóa toàn bộ tài liệu tham khảo.

    Nguồn tìm kiếm (theo thứ tự ưu tiên):
      1. File .bib riêng trong cùng thư mục
      2. \\bibitem trong \\begin{thebibliography}...\\end{thebibliography}

    Args:
        compilation_path : Danh sách .tex files theo compilation path
        arxiv_id         : Mã bài báo (ví dụ: "2310.00001")

    Returns:
        List[BibtexEntry]: Danh sách entries đã chuẩn hóa.
    """
    if not compilation_path:
        return []

    version_dir = compilation_path[0].parent

    # Ưu tiên 1: file .bib riêng
    bib_files = list(version_dir.rglob("*.bib"))
    if bib_files:
        return _parse_bib_file(bib_files[0], arxiv_id)

    # Ưu tiên 2: \bibitem trong .tex
    return _parse_bibitem_from_tex(compilation_path, arxiv_id)


def _parse_bib_file(bib_path: Path, arxiv_id: str) -> List[BibtexEntry]:
    """Parse file .bib bằng thư viện bibtexparser."""
    entries: List[BibtexEntry] = []
    try:
        import bibtexparser
        with open(bib_path, encoding="utf-8", errors="ignore") as f:
            db = bibtexparser.load(f)
        for e in db.entries:
            entries.append(BibtexEntry(
                arxiv_id=arxiv_id,
                citation_key=e.get("ID", ""),
                entry_type=e.get("ENTRYTYPE", "misc"),
                title=e.get("title"),
                authors=_split_authors(e.get("author", "")),
                year=int(e["year"]) if e.get("year", "").isdigit() else None,
                doi=e.get("doi"),
                journal=e.get("journal") or e.get("booktitle"),
                url=e.get("url"),
                confidence_score=1.0,
                source_method="bib_file",
            ))
    except Exception:
        pass
    return entries


def _parse_bibitem_from_tex(
    compilation_path: List[Path],
    arxiv_id: str,
) -> List[BibtexEntry]:
    """Tìm \\begin{thebibliography}...\\end{...} trong .tex, extract từng \\bibitem."""
    combined = ""
    for tex_path in compilation_path:
        try:
            combined += tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

    bib_match = _BIBLIOGRAPHY_ENV.search(combined)
    if not bib_match:
        return []

    bib_block = bib_match.group(0)

    entries = []
    for m in _BIBITEM_PATTERN.finditer(bib_block):
        key = m.group(1).strip()
        raw = m.group(2).strip()
        entry = _parse_bibitem_block(key, raw, arxiv_id)
        entries.append(entry)

    return entries


def _parse_bibitem_block(
    citation_key: str,
    raw_text: str,
    arxiv_id: str,
) -> BibtexEntry:
    """Parse một block \\bibitem thô thành BibtexEntry chuẩn."""
    text = re.sub(r"\s+", " ", raw_text).strip()

    # 1. Year
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", text)
    year = int(year_match.group(1)) if year_match else None

    # 2. DOI
    doi_match = re.search(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", text)
    doi = doi_match.group(1).rstrip(".,;)") if doi_match else None

    # 3. URL & arXiv ID Hint (Đòn bẩy Milestone 3)
    url_match = re.search(r"https?://[^\s{}]+", text)
    url = url_match.group(0).rstrip(".,;)") if url_match else None

    arxiv_hint_match = re.search(
        r"(?:arxiv(?:\.org)?(?:/(?:abs|pdf))?[:\/\s]+)(\d{4}\.\d{4,5}(?:v\d+)?)",
        text,
        re.IGNORECASE,
    )
    arxiv_id_hint = arxiv_hint_match.group(1) if arxiv_hint_match else None

    # 4. Title & Authors & Journal/Venue
    title: Optional[str] = None
    authors: List[str] = []
    journal: Optional[str] = None

    quote_match = re.search(r"(?:``|\")([^\"']+?)(?:''|\")", text)
    style_match = re.search(r"\\(?:textit|emph|textbf|em)\{([^}]+)\}", text)

    title_start = -1
    title_end = -1

    if quote_match and len(quote_match.group(1).strip()) > 5:
        title = quote_match.group(1).strip()
        title_start, title_end = quote_match.span()
    elif style_match and len(style_match.group(1).strip()) > 5:
        title = style_match.group(1).strip()
        title_start, title_end = style_match.span()

    if title and title_start != -1:
        author_part = text[:title_start].strip().rstrip(",.")
        tail_part = text[title_end:].strip().lstrip(",.")

        authors = _parse_authors_from_text(author_part)

        j_match = re.search(r"(?:in\s+)?\\(?:textit|emph|em)\{([^}]+)\}", tail_part, re.IGNORECASE)
        if not j_match:
            j_match = re.search(r"\bin\s+([A-Z][A-Za-z0-9\s,\.:&-]+?)(?:,|\.|\b\d{4}\b|$)", tail_part, re.IGNORECASE)
        if j_match:
            journal = j_match.group(1).strip()
    else:
        parts = [p.strip() for p in text.split(".") if p.strip()]
        if len(parts) >= 2:
            authors = _parse_authors_from_text(parts[0])
            title = parts[1]
            if len(parts) >= 3:
                journal = parts[2]
        elif len(parts) == 1:
            title = parts[0]

    if title:
        title = re.sub(r"\\[a-zA-Z]+|\{|\}", "", title).strip().rstrip(",.")

    # 5. Phân loại Entry Type
    text_lower = text.lower()
    if journal and any(kw in journal.lower() for kw in ("proc", "conference", "symposium", "inproceedings")):
        entry_type = "inproceedings"
    elif "arxiv" in text_lower or arxiv_id_hint:
        entry_type = "article"
    elif any(kw in text_lower for kw in ("journal", "transactions", "ieee", "acm", "vol")):
        entry_type = "article"
    elif "thesis" in text_lower or "dissertation" in text_lower:
        entry_type = "phdthesis"
    elif "book" in text_lower:
        entry_type = "book"
    else:
        entry_type = "misc"

    source_method = "regex"

    # CrossRef API Auto-Enrichment
    if title and (not authors or year is None or not doi):
        enriched = _enrich_with_crossref(title, doi)
        if enriched:
            title = enriched.get("title") or title
            authors = enriched.get("authors") or authors
            year = enriched.get("year") or year
            doi = enriched.get("doi") or doi
            journal = enriched.get("journal") or journal
            source_method = "crossref"

    # LLM Fallback (khi regex thiếu trường nghiêm trọng)
    if not title or not authors:
        llm_entry = _fallback_parse_with_llm(citation_key, raw_text, arxiv_id)
        if llm_entry:
            llm_entry.arxiv_id_hint = arxiv_id_hint
            llm_entry.source_method = "llm"
            llm_entry.confidence_score = _calculate_confidence_score(llm_entry)
            return llm_entry

    res_entry = BibtexEntry(
        arxiv_id=arxiv_id,
        citation_key=citation_key,
        entry_type=entry_type,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        journal=journal,
        url=url,
        arxiv_id_hint=arxiv_id_hint,
        source_method=source_method,
    )
    res_entry.confidence_score = _calculate_confidence_score(res_entry)
    return res_entry


def _parse_authors_from_text(raw_authors: str) -> List[str]:
    """Tách danh sách tác giả từ văn bản thô."""
    clean_str = re.sub(r"\\[a-zA-Z]+|\{|\}|~", " ", raw_authors)
    clean_str = re.sub(r"\bet\s+al\.?\b", "", clean_str, flags=re.IGNORECASE).strip()
    clean_str = re.sub(r"\s+and\s+", ", ", clean_str, flags=re.IGNORECASE)
    tokens = [p.strip() for p in clean_str.split(",") if p.strip()]

    author_list = []
    for token in tokens:
        name = token.strip().rstrip(".,")
        if 2 <= len(name) <= 60 and not re.search(r"^(in|pp|vol|no|page)\b", name, re.IGNORECASE):
            author_list.append(name)
    return author_list


def _split_authors(author_str: str) -> List[str]:
    """Tách chuỗi authors từ bibtexparser format."""
    if not author_str:
        return []
    return [a.strip() for a in re.split(r"\s+and\s+", author_str, flags=re.IGNORECASE)]


def _calculate_confidence_score(entry: BibtexEntry) -> float:
    """Tính điểm chất lượng bóc tách từ 0.0 đến 1.0"""
    score = 0.0
    if entry.title and len(entry.title.strip()) >= 5:
        score += 0.45
    if entry.authors and len(entry.authors) > 0:
        score += 0.25
    if entry.year and 1900 <= entry.year <= 2026:
        score += 0.15
    if entry.doi or entry.arxiv_id_hint:
        score += 0.15
    return round(min(score, 1.0), 2)


def _enrich_with_crossref(title: str, doi: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Tự động đối soát và bù đắp thông tin qua CrossRef Open API."""
    try:
        if doi:
            url = f"https://api.crossref.org/works/{doi}"
        else:
            query = urllib.parse.quote_plus(title)
            url = f"https://api.crossref.org/works?query.title={query}&rows=1"

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Intro2DS-Group3 (student project; mailto:contact@group3.ds)"}
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if doi:
                item = data.get("message", {})
            else:
                items = data.get("message", {}).get("items", [])
                if not items:
                    return None
                item = items[0]

            ext_title = item.get("title", [None])[0]
            ext_doi = item.get("DOI")
            ext_journal = item.get("container-title", [None])[0]

            ext_authors = []
            for a in item.get("author", []):
                name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                if name:
                    ext_authors.append(name)

            ext_year = None
            date_parts = item.get("published", {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                ext_year = int(date_parts[0][0])

            return {
                "title": ext_title,
                "authors": ext_authors,
                "year": ext_year,
                "doi": ext_doi,
                "journal": ext_journal,
            }
    except Exception:
        return None


def _fallback_parse_with_llm(
    citation_key: str,
    raw_text: str,
    arxiv_id: str,
) -> Optional[BibtexEntry]:
    """Cơ chế dự phòng bằng mô hình ngôn ngữ nhỏ khi cú pháp lỗi nghiêm trọng."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None

    try:
        prompt = (
            "Extract structured bibliographic metadata from this LaTeX bibitem. "
            "Return ONLY a JSON object with keys: title (string), authors (list of strings), "
            "year (integer or null), doi (string or null), journal (string or null), entry_type (string).\n\n"
            f"Bibitem text:\n{raw_text}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"response_mime_type": "application/json"}
        }).encode("utf-8")

        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            text_out = res_data["candidates"][0]["content"]["parts"][0]["text"]
            meta = json.loads(text_out)

            return BibtexEntry(
                arxiv_id=arxiv_id,
                citation_key=citation_key,
                entry_type=meta.get("entry_type", "misc"),
                title=meta.get("title"),
                authors=meta.get("authors", []),
                year=meta.get("year"),
                doi=meta.get("doi"),
                journal=meta.get("journal"),
            )
    except Exception:
        return None


# ===== PHẦN MỚI: UNIFY ACROSS VERSIONS + DEDUPLICATION =====

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import argparse
import itertools
import threading
import time
import unicodedata
from typing import Callable, Iterable, Sequence, Tuple


_SOURCE_PRIORITY = {
    "bib_file": 4,
    "regex": 3,
    "llm": 2,
    "crossref": 1,
}
_CITATION_COMMAND_PATTERN = re.compile(
    r"\\(?P<command>cite|citep|citet|citealt|citealp|citeauthor|citeyear|"
    r"citeyearpar|citenum|parencite|textcite|autocite|cites|nocite)"
    r"(?P<star>\*)?"
    r"(?P<body>(?:\s*\[[^\]]*\]){0,2}\s*\{[^{}]*\}"
    r"(?:\s*(?:\[[^\]]*\]){0,2}\s*\{[^{}]*\})*)"
)
_LATEX_COMMAND_PATTERN = re.compile(r"\\[a-zA-Z@]+\*?(?:\s*\[[^\]]*\])?")
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)


@dataclass
class UnifiedBibEntry:
    """Một entry đã hợp nhất qua các version của cùng một publication."""

    arxiv_id: str
    citation_key: str
    entry_type: str
    title: Optional[str]
    authors: List[str]
    year: Optional[int]
    doi: Optional[str]
    journal: Optional[str]
    url: Optional[str]
    aliases: List[str] = field(default_factory=list)
    versions_present: List[int] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    arxiv_id_hint: Optional[str] = None
    conflicts: List[Dict[str, Any]] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        """Trả về đúng các cột của bảng chuẩn bibtex."""
        return {
            "arxiv_id": self.arxiv_id,
            "citation_key": self.citation_key,
            "entry_type": self.entry_type,
            "title": self.title,
            "authors": list(self.authors),
            "year": self.year,
            "doi": _norm_doi(self.doi) or None,
            "journal": self.journal,
            "url": self.url,
        }

    def to_debug_row(self) -> Dict[str, Any]:
        """Trả về bản ghi đầy đủ phục vụ kiểm tra quá trình hợp nhất."""
        return {**self.to_row(),
                "aliases": list(self.aliases),
                "versions_present": list(self.versions_present),
                "sources": list(self.sources),
                "confidence": self.confidence,
                "arxiv_id_hint": self.arxiv_id_hint,
                "conflicts": list(self.conflicts)}


@dataclass
class UnifiedBibliography:
    """Kết quả hợp nhất một publication, gồm dữ liệu, ánh xạ key và thống kê."""

    entries: List[UnifiedBibEntry]
    key_map: Dict[int, Dict[str, str]]
    stats: Dict[str, Any]


def _norm_text(value: Optional[str]) -> str:
    """Bỏ dấu LaTeX, chuẩn hóa accent và dấu câu để so khớp."""
    if not value:
        return ""
    text = _LATEX_COMMAND_PATTERN.sub(" ", value)
    text = text.replace("{", " ").replace("}", " ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _PUNCTUATION_PATTERN.sub(" ", text.lower())
    return " ".join(text.split())


def _norm_title(value: Optional[str]) -> str:
    """Chuẩn hóa title riêng cho các bước nhận diện cùng tài liệu."""
    return _norm_text(value)


def _norm_doi(value: Optional[str]) -> str:
    """Chuẩn hóa DOI nhưng không thay đổi giá trị lưu trong entry."""
    if not value:
        return ""
    doi = value.strip().lower()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi)
    doi = re.sub(r"^dx\.doi\.org/", "", doi)
    return doi.rstrip(" \t\r\n.,;:)]}")


def _norm_arxiv(value: Optional[str]) -> str:
    """Bỏ hậu tố version khỏi arXiv ID hint."""
    if not value:
        return ""
    return re.sub(r"v\d+$", "", value.strip().lower())


def _author_surname(author: str) -> str:
    """Rút họ từ tên dạng Last, First hoặc First Last."""
    clean = _norm_text(re.sub(r"\bet\s+al\.?\b", "", author, flags=re.IGNORECASE))
    if not clean:
        return ""
    if "," in author:
        left = _norm_text(author.split(",", 1)[0])
        return left.split()[-1] if left else ""
    parts = clean.split()
    if len(parts) == 1:
        return parts[0]
    # Họ compound phổ biến được giữ nguyên theo token cuối để ổn định giữa định dạng.
    return parts[-1]


def _author_surnames(authors: Sequence[str]) -> frozenset[str]:
    """Trả về tập họ đã chuẩn hóa, không tính et al."""
    names = set()
    for author in authors or []:
        if re.search(r"\bet\s+al\.?\b", author, flags=re.IGNORECASE):
            continue
        surname = _author_surname(author)
        if surname:
            names.add(surname)
    return frozenset(names)


def _entry_identity(entry: BibtexEntry) -> Tuple[Any, ...]:
    """Khóa xác định để sắp xếp đầu vào ổn định, không dùng thứ tự list."""
    return (
        entry.citation_key or "",
        _norm_title(entry.title),
        _norm_doi(entry.doi),
        entry.year if entry.year is not None else -1,
        tuple(sorted(_author_surnames(entry.authors or []))),
        entry.entry_type or "",
        entry.source_method or "",
        tuple(entry.authors or []),
        entry.title or "",
        entry.doi or "",
        entry.journal or "",
        entry.url or "",
        entry.confidence_score or 0.0,
    )


def _years_compatible(left: Optional[int], right: Optional[int]) -> bool:
    return left is None or right is None or abs(left - right) <= 1


def _authors_compatible(left: Sequence[str], right: Sequence[str]) -> bool:
    left_names = _author_surnames(left)
    right_names = _author_surnames(right)
    return not left_names or not right_names or bool(left_names & right_names)


def _title_similarity(left: str, right: str) -> float:
    """So khớp bằng difflib cố định, không phụ thuộc package tùy chọn."""
    return SequenceMatcher(None, left, right).ratio()


class _UnionFind:
    """Union-Find có chọn root ổn định theo chỉ số đầu vào."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> bool:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return False
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        elif self.rank[left_root] == self.rank[right_root] and left_root > right_root:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1
        return True


def _clusters(uf: _UnionFind, size: int) -> List[List[int]]:
    groups: Dict[int, List[int]] = defaultdict(list)
    for index in range(size):
        groups[uf.find(index)].append(index)
    return [groups[root] for root in sorted(groups)]


def _can_merge(
    left: Sequence[BibtexEntry],
    right: Sequence[BibtexEntry],
    *,
    fuzzy: bool,
    same_key_requires_title: bool = False,
    same_key_threshold: float = 0.80,
) -> bool:
    """Áp dụng veto DOI, đồng thời kiểm tra title/năm/tác giả theo tầng."""
    left_dois = {_norm_doi(item.doi) for item in left if _norm_doi(item.doi)}
    right_dois = {_norm_doi(item.doi) for item in right if _norm_doi(item.doi)}
    if left_dois and right_dois and (
        left_dois != right_dois or len(left_dois) != 1
    ):
        return False
    if same_key_requires_title:
        left_titles = {_norm_title(item.title) for item in left if _norm_title(item.title)}
        right_titles = {_norm_title(item.title) for item in right if _norm_title(item.title)}
        if left_titles and right_titles and not any(
            _title_similarity(a, b) >= same_key_threshold
            for a in left_titles for b in right_titles
        ):
            return False
    for first in left:
        for second in right:
            title_a = _norm_title(first.title)
            title_b = _norm_title(second.title)
            if not fuzzy and not same_key_requires_title and title_a and title_b and title_a != title_b:
                continue
            if not _years_compatible(first.year, second.year):
                continue
            if not _authors_compatible(first.authors or [], second.authors or []):
                continue
            return True
    # DOI/arXiv chắc chắn được phép nối dù metadata phụ không khớp.
    return False


def _field_candidates(
    members: Sequence[Tuple[int, BibtexEntry]],
    field_name: str,
) -> List[Tuple[Any, int, BibtexEntry]]:
    candidates = []
    for version, entry in members:
        value = getattr(entry, field_name)
        if value is not None and value != "" and value != []:
            candidates.append((value, version, entry))
    return candidates


def _candidate_rank(candidate: Tuple[Any, int, BibtexEntry]) -> Tuple[Any, ...]:
    value, version, entry = candidate
    return (
        _SOURCE_PRIORITY.get(entry.source_method or "", 0),
        float(entry.confidence_score or 0.0),
        version,
        str(value).lower(),
    )


def _select_scalar(
    members: Sequence[Tuple[int, BibtexEntry]],
    field_name: str,
    conflicts: List[Dict[str, Any]],
) -> Any:
    candidates = _field_candidates(members, field_name)
    if not candidates:
        return None
    all_candidates = list(candidates)
    if field_name == "year":
        counts = Counter(value for value, _, _ in candidates)
        maximum = max(counts.values())
        candidates = [item for item in candidates if counts[item[0]] == maximum]
    if field_name == "entry_type":
        specific = [item for item in candidates if str(item[0]).lower() != "misc"]
        if specific:
            candidates = specific
    selected = max(candidates, key=_candidate_rank)[0]
    distinct: Dict[str, Any] = {}
    for value, _, _ in all_candidates:
        canonical = _norm_doi(value) if field_name == "doi" else value
        distinct.setdefault(
            json.dumps(canonical, ensure_ascii=False, sort_keys=True),
            canonical,
        )
    if len(distinct) > 1:
        conflicts.append({
            "field": field_name,
            "values": [distinct[key] for key in sorted(distinct)],
            "selected": _norm_doi(selected) if field_name == "doi" else selected,
        })
    return selected


def _author_completeness(author: str) -> Tuple[int, int, str]:
    tokens = _norm_text(author).split()
    non_initials = sum(1 for token in tokens if len(token) > 1)
    return non_initials, len(tokens), _norm_text(author)


def _select_authors(
    members: Sequence[Tuple[int, BibtexEntry]],
    conflicts: List[Dict[str, Any]],
) -> List[str]:
    candidates = []
    for version, entry in members:
        authors = list(entry.authors or [])
        if authors:
            normalized_count = len(_author_surnames(authors))
            completeness = sum(_author_completeness(name)[0] for name in authors)
            candidates.append((authors, normalized_count, completeness, version, entry))
    if not candidates:
        return []
    authors, _, _, _, _ = max(
        candidates,
        key=lambda item: (item[1], item[2], item[3], tuple(item[0])),
    )
    distinct = {}
    for item in candidates:
        key = json.dumps(item[0], ensure_ascii=False)
        distinct.setdefault(key, item[0])
    if len(distinct) > 1:
        conflicts.append({
            "field": "authors",
            "values": [distinct[key] for key in sorted(distinct)],
            "selected": list(authors),
        })
    return list(authors)


def _key_choice(
    members: Sequence[Tuple[int, BibtexEntry]],
) -> Tuple[str, List[str]]:
    """Chọn key theo số version, version mới nhất rồi thứ tự từ điển."""
    key_versions: Dict[str, set[int]] = defaultdict(set)
    key_latest: Dict[str, int] = {}
    for version, entry in members:
        key = (entry.citation_key or "").strip()
        if not key:
            continue
        key_versions[key].add(version)
        key_latest[key] = max(version, key_latest.get(key, version))
    if not key_versions:
        return "unknown", []
    keys = sorted(key_versions)
    selected = min(
        keys,
        key=lambda key: (-len(key_versions[key]), -key_latest[key], key),
    )
    return selected, [key for key in keys if key != selected]


def _build_unified_entry(
    arxiv_id: str,
    members: Sequence[Tuple[int, BibtexEntry]],
) -> UnifiedBibEntry:
    conflicts: List[Dict[str, Any]] = []
    key, aliases = _key_choice(members)
    title = _select_scalar(members, "title", conflicts)
    year = _select_scalar(members, "year", conflicts)
    doi = _norm_doi(_select_scalar(members, "doi", conflicts)) or None
    journal = _select_scalar(members, "journal", conflicts)
    url = _select_scalar(members, "url", conflicts)
    entry_type = _select_scalar(members, "entry_type", conflicts) or "misc"
    authors = _select_authors(members, conflicts)
    versions = sorted({version for version, _ in members})
    sources = sorted({entry.source_method for _, entry in members if entry.source_method})
    confidence = max(
        (float(entry.confidence_score or 0.0) for _, entry in members),
        default=0.0,
    )
    hints = sorted({
        entry.arxiv_id_hint.strip()
        for _, entry in members
        if entry.arxiv_id_hint and entry.arxiv_id_hint.strip()
    })
    if len(hints) > 1:
        conflicts.append({
            "field": "arxiv_id_hint",
            "values": hints,
            "selected": hints[0],
        })
    return UnifiedBibEntry(
        arxiv_id=arxiv_id,
        citation_key=key,
        entry_type=entry_type,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        journal=journal,
        url=url,
        aliases=aliases,
        versions_present=versions,
        sources=sources,
        confidence=confidence,
        arxiv_id_hint=hints[0] if hints else None,
        conflicts=conflicts,
    )


def unify_publication(
    arxiv_id: str,
    per_version: Dict[int, List[BibtexEntry]],
    *,
    fuzzy_threshold: float = 0.92,
) -> UnifiedBibliography:
    """Gộp BibTeX của các version, deduplicate theo các tầng có veto DOI.

    Thứ tự hợp nhất: DOI/arXiv chắc chắn, title chuẩn hóa, cùng citation key
    tương thích, sau đó so khớp fuzzy có blocking để giới hạn số cặp cần xét.
    """
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold phải nằm trong khoảng [0, 1].")
    versions = sorted(per_version)
    indexed = [
        (version, entry)
        for version in versions
        for entry in sorted(per_version[version] or [], key=_entry_identity)
    ]
    entries = [entry for _, entry in indexed]
    uf = _UnionFind(len(entries))
    merge_counts = {"doi": 0, "arxiv_hint": 0, "title": 0, "citation_key": 0, "fuzzy": 0}

    def connect_pairs(pairs: Iterable[Tuple[int, int]], stage: str,
                      *, same_key: bool = False, strong: bool = False) -> None:
        for left_index, right_index in sorted(set(pairs)):
            left_root = uf.find(left_index)
            right_root = uf.find(right_index)
            if left_root == right_root:
                continue
            left_members = [entries[i] for i in range(len(entries))
                            if uf.find(i) == left_root]
            right_members = [entries[i] for i in range(len(entries))
                             if uf.find(i) == right_root]
            if strong:
                left_dois = {_norm_doi(item.doi) for item in left_members if _norm_doi(item.doi)}
                right_dois = {_norm_doi(item.doi) for item in right_members if _norm_doi(item.doi)}
                if left_dois and right_dois and (
                    left_dois != right_dois or len(left_dois) != 1
                ):
                    continue
                uf.union(left_index, right_index)
                merge_counts[stage] += 1
            elif _can_merge(
                left_members,
                right_members,
                fuzzy=(stage == "fuzzy"),
                same_key_requires_title=same_key,
            ):
                uf.union(left_index, right_index)
                merge_counts[stage] += 1

    # Tầng 1: đồng nhất DOI hoặc arXiv ID hint. Khác DOI luôn veto.
    for identity_field, stage in (("doi", "doi"), ("arxiv_id_hint", "arxiv_hint")):
        groups: Dict[str, List[int]] = defaultdict(list)
        for index, entry in enumerate(entries):
            value = _norm_doi(entry.doi) if identity_field == "doi" else _norm_arxiv(entry.arxiv_id_hint)
            if value:
                groups[value].append(index)
        pairs = (pair for value in sorted(groups) for pair in itertools.combinations(groups[value], 2))
        connect_pairs(pairs, stage, strong=True)

    # Tầng 2: tiêu đề bằng nhau, đủ dài, không mâu thuẫn về năm hoặc tác giả.
    groups = defaultdict(list)
    for index, entry in enumerate(entries):
        title = _norm_title(entry.title)
        if len(title) >= 12:
            groups[title].append(index)
    pairs = (pair for title in sorted(groups) for pair in itertools.combinations(groups[title], 2))
    connect_pairs(pairs, "title")

    # Cùng key không tự động chứng minh cùng tài liệu: title khác nhau là tái sử dụng key.
    key_groups: Dict[str, List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        if entry.citation_key:
            key_groups[entry.citation_key.strip()].append(index)
    pairs = (pair for key in sorted(key_groups)
             for pair in itertools.combinations(key_groups[key], 2))
    connect_pairs(pairs, "citation_key", same_key=True)

    # Tầng 3: blocking theo họ đầu và prefix token tiêu đề; chỉ so cặp trong cùng block.
    blocks: Dict[Tuple[str, Tuple[str, ...]], List[int]] = defaultdict(list)
    for index, entry in enumerate(entries):
        title_tokens = _norm_title(entry.title).split()
        if len(title_tokens) < 3:
            continue
        first_surname = next(
            (
                surname
                for author in (entry.authors or [])
                if (surname := _author_surname(author))
            ),
            "",
        )
        blocks[(first_surname, tuple(title_tokens[:3]))].append(index)
    fuzzy_pairs = set()
    for block in sorted(blocks):
        fuzzy_pairs.update(itertools.combinations(blocks[block], 2))
    for left_index, right_index in sorted(fuzzy_pairs):
        if uf.find(left_index) == uf.find(right_index):
            continue
        left = entries[left_index]
        right = entries[right_index]
        title_a = _norm_title(left.title)
        title_b = _norm_title(right.title)
        if not title_a or not title_b:
            continue
        if _title_similarity(title_a, title_b) < fuzzy_threshold:
            continue
        names_a = _author_surnames(left.authors or [])
        names_b = _author_surnames(right.authors or [])
        overlap = (
            not names_a or not names_b
            or len(names_a & names_b) / max(1, len(names_a | names_b)) >= 0.5
        )
        if not overlap or not _years_compatible(left.year, right.year):
            continue
        connect_pairs([(left_index, right_index)], "fuzzy")

    component_members: List[List[Tuple[int, BibtexEntry]]] = []
    for indices in _clusters(uf, len(entries)):
        component_members.append([indexed[index] for index in indices])
    components = [
        (_build_unified_entry(arxiv_id, members), members)
        for members in component_members
    ]
    components.sort(key=lambda pair: (
        pair[0].citation_key,
        _norm_title(pair[0].title),
        pair[0].year or 0,
        tuple((version, _entry_identity(entry)) for version, entry in pair[1]),
    ))

    # Một key duy nhất trong publication; nếu key bị tái sử dụng, ghi xung đột.
    used_keys = set()
    original_key_targets: Dict[Tuple[int, str], set[str]] = defaultdict(set)
    for item, members in components:
        base = item.citation_key or "unknown"
        candidate = base
        suffix = 2
        while candidate in used_keys:
            candidate = f"{base}_{suffix}"
            suffix += 1
        item.citation_key = candidate
        used_keys.add(candidate)
        for version, entry in members:
            old_key = (entry.citation_key or "").strip()
            if old_key:
                original_key_targets[(version, old_key)].add(candidate)
        if candidate != base:
            item.conflicts.append({
                "field": "citation_key",
                "values": [base, candidate],
                "selected": candidate,
            })

    unified = [item for item, _ in components]
    key_map: Dict[int, Dict[str, str]] = {}
    for (version, old_key), targets in sorted(original_key_targets.items()):
        # Key bị tái sử dụng cùng version là mơ hồ; không tạo một ánh xạ sai.
        if len(targets) == 1:
            target = next(iter(targets))
            if old_key != target:
                key_map.setdefault(version, {})[old_key] = target
    key_map = {version: dict(sorted(mapping.items()))
               for version, mapping in sorted(key_map.items()) if mapping}

    all_conflicts = sum(len(item.conflicts) for item in unified)
    total_before = len(entries)
    empty_versions = sum(not (per_version.get(version) or []) for version in versions)
    stats = {
        "arxiv_id": arxiv_id,
        "entries_before_dedup": total_before,
        "entries_after_dedup": len(unified),
        "reduction_ratio": (
            (total_before - len(unified)) / total_before if total_before else 0.0
        ),
        "merges_by_stage": merge_counts,
        "field_conflicts": all_conflicts,
        "empty_versions": empty_versions,
        "versions_total": len(versions),
        "keys_changed": sum(
            any(old_key != target for target in targets)
            for (_, old_key), targets in original_key_targets.items()
        ),
        "missing_title_ratio": (
            sum(not item.title for item in unified) / len(unified) if unified else 0.0
        ),
        "missing_authors_ratio": (
            sum(not item.authors for item in unified) / len(unified) if unified else 0.0
        ),
        "missing_year_ratio": (
            sum(item.year is None for item in unified) / len(unified) if unified else 0.0
        ),
    }
    return UnifiedBibliography(unified, key_map, stats)


def rewrite_cites(
    text: str,
    key_map: Dict[int, Dict[str, str]],
    version: int,
) -> str:
    r"""Viết lại cite commands bằng đúng key_map của version được chỉ định.

    Gọi hàm này TRƯỚC bước so khớp full-text dedup của elements để các câu
    giống nhau nhưng khác key cite vẫn được gộp thành một element_id.
    """
    version_map = key_map.get(version, {})

    def replace(match: re.Match[str]) -> str:
        body = match.group("body")
        changed = False

        def replace_group(group_match: re.Match[str]) -> str:
            nonlocal changed
            original_keys = group_match.group(1)
            keys = [key.strip() for key in original_keys.split(",")]
            rewritten: List[str] = []
            seen = set()
            for key in keys:
                mapped = version_map.get(key, key)
                if mapped != key:
                    changed = True
                if mapped and mapped not in seen:
                    rewritten.append(mapped)
                    seen.add(mapped)
            return "{" + ", ".join(rewritten) + "}"

        updated_body = re.sub(r"\{([^{}]*)\}", replace_group, body)
        if not changed:
            return match.group(0)
        return (
            "\\" + match.group("command") + (match.group("star") or "")
            + updated_body
        )

    return _CITATION_COMMAND_PATTERN.sub(replace, text)


def _fallback_compilation_path(version_dir: Path) -> List[Path]:
    """Tìm file chính có documentclass, theo input/include và bỏ comment TeX.

    Giả định đơn giản: source dùng \\input{...} hoặc \\include{...}; macro động,
    đường dẫn phức tạp và file nằm ngoài version_dir không được mở rộng.
    """
    tex_files = sorted(version_dir.rglob("*.tex"))
    if not tex_files:
        return []
    document_pattern = re.compile(r"\\documentclass(?:\[[^\]]*\])?\s*\{")
    main_files = []
    for path in tex_files:
        content = path.read_text(encoding="utf-8", errors="ignore")
        uncommented = re.sub(r"(?<!\\)%.*", "", content)
        if document_pattern.search(uncommented):
            main_files.append(path)
    if not main_files:
        main_files = tex_files[:1]
    visited = set()
    ordered: List[Path] = []
    include_pattern = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited:
            return
        visited.add(resolved)
        ordered.append(path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        content = re.sub(r"(?<!\\)%.*", "", content)
        for match in include_pattern.finditer(content):
            target = match.group(1).strip()
            include_path = (path.parent / target)
            candidates = [include_path]
            if include_path.suffix.lower() != ".tex":
                candidates.append(include_path.with_suffix(".tex"))
            for candidate in candidates:
                if candidate.is_file():
                    visit(candidate)
                    break

    for main in sorted(main_files):
        visit(main)
    return ordered


def process_publication(
    arxiv_id: str,
    version_dirs: Dict[int, Path],
    get_compilation_path: Optional[Callable[[Path], List[Path]]] = None,
    *,
    fuzzy_threshold: float = 0.92,
) -> UnifiedBibliography:
    """Trích xuất BibTeX từng version rồi hợp nhất; version rỗng vẫn được ghi nhận."""
    per_version: Dict[int, List[BibtexEntry]] = {}
    path_getter = get_compilation_path or _fallback_compilation_path
    for version in sorted(version_dirs):
        version_dir = Path(version_dirs[version])
        compilation_path = path_getter(version_dir)
        per_version[version] = extract_bibtex(compilation_path, arxiv_id) if compilation_path else []
    return unify_publication(arxiv_id, per_version, fuzzy_threshold=fuzzy_threshold)


def _find_publications(root: Path) -> Dict[str, Dict[int, Path]]:
    """Tìm <root>/<arxiv_id>/<vN hoặc N>/...tex theo bố cục corpus đã quy ước."""
    publications: Dict[str, Dict[int, Path]] = {}
    for publication_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        versions: Dict[int, Path] = {}
        for version_dir in sorted(path for path in publication_dir.iterdir() if path.is_dir()):
            match = re.fullmatch(r"v?(\d+)", version_dir.name, flags=re.IGNORECASE)
            if match:
                version_number = int(match.group(1))
                if version_number in versions:
                    raise ValueError(
                        f"Có nhiều thư mục cho version {version_number} trong "
                        f"{publication_dir}: {versions[version_number].name}, "
                        f"{version_dir.name}"
                    )
                versions[version_number] = version_dir
        publications[publication_dir.name] = versions
    return publications


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _run_publication_job(
    job: Tuple[
        str,
        Dict[int, Path],
        Optional[Callable[[Path], List[Path]]],
        float,
        bool,
    ],
) -> Tuple[str, UnifiedBibliography, float]:
    """Worker top-level để tương thích ProcessPoolExecutor trên Windows."""
    arxiv_id, versions, path_getter, fuzzy_threshold, no_network = job
    if no_network:
        def disabled_crossref(title: str, doi: Optional[str] = None) -> None:
            _ = title, doi
            return None
        globals()["_enrich_with_crossref"] = disabled_crossref
        os.environ.pop("GEMINI_API_KEY", None)
    started = time.perf_counter()
    result = process_publication(
        arxiv_id,
        versions,
        get_compilation_path=path_getter,
        fuzzy_threshold=fuzzy_threshold,
    )
    return arxiv_id, result, time.perf_counter() - started


def _rss_tree_bytes(psutil_module: Any) -> int:
    """Đo RSS tiến trình runner và các worker đang tồn tại."""
    process = psutil_module.Process()
    processes = [process]
    try:
        processes.extend(process.children(recursive=True))
    except psutil_module.Error:
        pass
    total = 0
    for item in processes:
        try:
            total += item.memory_info().rss
        except psutil_module.Error:
            continue
    return total


def run_corpus(
    root: Path,
    out_dir: Path,
    workers: int = 1,
    no_network: bool = False,
    fuzzy_threshold: float = 0.92,
    get_compilation_path: Optional[Callable[[Path], List[Path]]] = None,
) -> Dict[str, Any]:
    """Chạy corpus và xuất Parquet (nếu có pyarrow) hoặc JSONL.

    Bố cục mặc định là <root>/<arxiv_id>/<vN hoặc N>/...tex; corpus khác có thể
    dùng get_compilation_path tùy chỉnh khi gọi process_publication/run_corpus.
    """
    if workers < 1:
        raise ValueError("workers phải lớn hơn hoặc bằng 1.")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError("fuzzy_threshold phải nằm trong khoảng [0, 1].")
    root = Path(root)
    out_dir = Path(out_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục corpus: {root}")
    if no_network:
        def disabled_crossref(title: str, doi: Optional[str] = None) -> None:
            _ = title, doi
            return None
        globals()["_enrich_with_crossref"] = disabled_crossref
        os.environ.pop("GEMINI_API_KEY", None)

    out_dir.mkdir(parents=True, exist_ok=True)
    publications = _find_publications(root)
    jobs = [(arxiv_id, versions) for arxiv_id, versions in sorted(publications.items())]
    started = time.perf_counter()
    publication_times: Dict[str, float] = {}
    try:
        import psutil
    except ImportError:
        psutil = None
    stop_sampling = threading.Event()
    peak_rss = [0]

    def sample_memory() -> None:
        while not stop_sampling.is_set():
            peak_rss[0] = max(peak_rss[0], _rss_tree_bytes(psutil))
            stop_sampling.wait(0.05)

    sampler = None
    if psutil is not None:
        sampler = threading.Thread(target=sample_memory, daemon=True)
        sampler.start()
    payloads = [
        (arxiv_id, versions, get_compilation_path, fuzzy_threshold, no_network)
        for arxiv_id, versions in jobs
    ]
    try:
        if workers == 1:
            completed = [_run_publication_job(job) for job in payloads]
        elif no_network:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                completed = list(executor.map(_run_publication_job, payloads))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                completed = list(executor.map(_run_publication_job, payloads))
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join()
    completed.sort(key=lambda item: item[0])
    for arxiv_id, _, elapsed in completed:
        publication_times[arxiv_id] = elapsed

    bib_rows = [entry.to_row() for _, result, _ in completed for entry in result.entries]
    debug_rows = [
        entry.to_debug_row()
        for _, result, _ in completed
        for entry in result.entries
    ]
    key_maps = {
        arxiv_id: {str(version): mapping for version, mapping in result.key_map.items()}
        for arxiv_id, result, _ in completed
    }
    if psutil is not None:
        peak_bytes = peak_rss[0]
    else:
        try:
            import resource
            peak_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if os.name != "darwin":
                peak_bytes *= 1024
        except ImportError:
            peak_bytes = None

    output_format = "jsonl"
    bibtex_path = out_dir / "bibtex.jsonl"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        _write_jsonl(bibtex_path, bib_rows)
    else:
        bibtex_path = out_dir / "bibtex.parquet"
        entry_type_type = pa.dictionary(pa.int8(), pa.string())
        schema = pa.schema([
            pa.field("arxiv_id", pa.string(), nullable=False),
            pa.field("citation_key", pa.string(), nullable=False),
            pa.field("entry_type", entry_type_type),
            pa.field("title", pa.string()),
            pa.field("authors", pa.list_(pa.string())),
            pa.field("year", pa.int32()),
            pa.field("doi", pa.string()),
            pa.field("journal", pa.string()),
            pa.field("url", pa.string()),
        ])
        arrays = []
        for field in schema:
            values = [row[field.name] for row in bib_rows]
            if field.name == "entry_type":
                array = pa.array(values, type=pa.string()).dictionary_encode()
                array = array.cast(field.type)
            else:
                array = pa.array(values, type=field.type)
            arrays.append(array)
        table = pa.Table.from_arrays(arrays, schema=schema)
        pq.write_table(table, bibtex_path)
        output_format = "parquet"

    _write_jsonl(out_dir / "bibtex_debug.jsonl", debug_rows)
    (out_dir / "key_map.json").write_text(
        json.dumps(key_maps, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    totals = {
        "arxiv_id": "__total__",
        "entries_before_dedup": sum(item.stats["entries_before_dedup"] for _, item, _ in completed),
        "entries_after_dedup": sum(item.stats["entries_after_dedup"] for _, item, _ in completed),
        "reduction_ratio": 0.0,
        "merges_by_stage": {
            stage: sum(item.stats["merges_by_stage"][stage] for _, item, _ in completed)
            for stage in ("doi", "arxiv_hint", "title", "citation_key", "fuzzy")
        },
        "field_conflicts": sum(item.stats["field_conflicts"] for _, item, _ in completed),
        "empty_versions": sum(item.stats["empty_versions"] for _, item, _ in completed),
        "versions_total": sum(item.stats["versions_total"] for _, item, _ in completed),
        "keys_changed": sum(item.stats["keys_changed"] for _, item, _ in completed),
    }
    before = totals["entries_before_dedup"]
    totals["reduction_ratio"] = (
        (before - totals["entries_after_dedup"]) / before if before else 0.0
    )
    for field_name in ("missing_title_ratio", "missing_authors_ratio", "missing_year_ratio"):
        totals[field_name] = (
            sum(
                item.stats[field_name] * item.stats["entries_after_dedup"]
                for _, item, _ in completed
            ) / totals["entries_after_dedup"]
            if totals["entries_after_dedup"] else 0.0
        )
    stats = {
        "publications": {
            arxiv_id: result.stats for arxiv_id, result, _ in completed
        },
        "total": totals,
        "performance": {
            "wall_time_seconds": time.perf_counter() - started,
            "average_publication_seconds": (
                sum(publication_times.values()) / len(publication_times)
                if publication_times else 0.0
            ),
            "peak_rss_bytes": peak_bytes,
            "workers": workers,
            "output_format": output_format,
            "output_sizes_bytes": {},
        },
    }
    stats_path = out_dir / "bibtex_stats.json"
    stats["performance"]["output_sizes_bytes"] = {
        path.name: path.stat().st_size
        for path in (bibtex_path, out_dir / "bibtex_debug.jsonl", out_dir / "key_map.json")
        if path.is_file()
    }
    stats["performance"]["fuzzy_threshold"] = fuzzy_threshold
    for _ in range(5):
        stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        stats_size = stats_path.stat().st_size
        if stats["performance"]["output_sizes_bytes"].get(stats_path.name) == stats_size:
            break
        stats["performance"]["output_sizes_bytes"][stats_path.name] = stats_size
    return stats


def _selftest() -> None:
    """Chạy mười nhóm kiểm thử nội bộ, chỉ dựng dữ liệu, không gọi mạng."""
    tests = 10
    passed = 0

    def entry(key: str, title: Optional[str] = None, *, year: Optional[int] = None,
              doi: Optional[str] = None, authors: Optional[List[str]] = None,
              hint: Optional[str] = None) -> BibtexEntry:
        return BibtexEntry(
            arxiv_id="2301.00001",
            citation_key=key,
            entry_type="article",
            title=title,
            authors=authors or [],
            year=year,
            doi=doi,
            arxiv_id_hint=hint,
            source_method="bib_file",
            confidence_score=1.0,
        )

    first = unify_publication("2301.00001", {
        1: [entry("vaswani2017", "Attention Is All You Need",
                  year=2017, authors=["Ashish Vaswani", "Noam Shazeer"])],
        2: [entry("vit_transformer", "Attention Is All You Need",
                  year=2017, doi="10.5555/attention", authors=["Ashish Vaswani", "Noam Shazeer"])],
        3: [entry("vit_transformer", "Attention Is All You Need",
                  doi="https://doi.org/10.5555/attention", authors=["Ashish Vaswani", "Noam Shazeer"])],
    })
    assert len(first.entries) == 1
    assert first.entries[0].citation_key == "vit_transformer"
    assert first.entries[0].title and first.entries[0].authors
    assert first.entries[0].year == 2017 and first.entries[0].doi
    assert first.key_map[1] == {"vaswani2017": "vit_transformer"}
    passed += 1

    reused = unify_publication("2301.00001", {
        1: [entry("shared", "A completely different paper about databases", year=2020),
            entry("shared", "A separate unrelated work on robotics", year=2022)],
    })
    assert len(reused.entries) == 2
    assert len({item.citation_key for item in reused.entries}) == 2
    passed += 1

    doi_conflict = unify_publication("2301.00001", {
        1: [entry("paper_a", "A sufficiently long identical paper title", doi="10.1000/a"),
            entry("paper_b", "A sufficiently long identical paper title", doi="10.1000/b")],
    })
    assert len(doi_conflict.entries) == 2
    passed += 1

    citation_text = r"\citep[see][p.3]{a, b}; \nocite{x}"
    cite_map = {1: {"a": "c", "b": "c"}}
    rewritten = rewrite_cites(citation_text, cite_map, 1)
    assert rewritten == r"\citep[see][p.3]{c}; \nocite{x}"
    assert rewrite_cites(rewritten, cite_map, 1) == rewritten
    passed += 1

    empty = unify_publication("2301.00001", {1: [], 2: [entry("only", "Some paper title")]})
    assert empty.stats["empty_versions"] == 1
    assert len(empty.entries) == 1
    passed += 1

    sample_versions = {
        1: [entry("a", "A deterministic paper title", year=2021, authors=["Smith, John"])],
        2: [entry("b", "A deterministic paper title", year=2021, authors=["John Smith"])],
        3: [],
    }
    result_a = unify_publication("2301.00001", sample_versions)
    reversed_input = {version: list(reversed(items))
                      for version, items in reversed(list(sample_versions.items()))}
    result_b = unify_publication("2301.00001", reversed_input)
    canonical_a = json.dumps(
        {"rows": [item.to_debug_row() for item in result_a.entries],
         "key_map": result_a.key_map, "stats": result_a.stats},
        ensure_ascii=False, sort_keys=True,
    )
    canonical_b = json.dumps(
        {"rows": [item.to_debug_row() for item in result_b.entries],
         "key_map": result_b.key_map, "stats": result_b.stats},
        ensure_ascii=False, sort_keys=True,
    )
    assert canonical_a == canonical_b
    passed += 1

    fuzzy = unify_publication("2301.00001", {
        1: [entry("fuzzy_a", "Robust Attention Models for Visual Recognition Tasks",
                  year=2021, authors=["Taylor Smith"])],
        2: [entry("fuzzy_b", "Robust Attention Models for Visual Recognition Task",
                  year=2021, authors=["Taylor Smith"])],
    })
    assert len(fuzzy.entries) == 1
    assert fuzzy.stats["merges_by_stage"]["fuzzy"] >= 1
    passed += 1

    same_key_fuzzy = unify_publication("2301.00001", {
        1: [entry("reused_key", "A Study of Robust Neural Models", year=2021)],
        2: [entry("reused_key", "A Study of Robust Neural Model", year=2021)],
    })
    assert len(same_key_fuzzy.entries) == 1
    passed += 1

    version_cites = r"\cite{a} \cite{a}"
    assert rewrite_cites(
        version_cites, {1: {"a": "c"}, 2: {"a": "d"}}, 1
    ) == r"\cite{c} \cite{c}"
    assert rewrite_cites(
        version_cites, {1: {"a": "c"}, 2: {"a": "d"}}, 2
    ) == r"\cite{d} \cite{d}"
    assert rewrite_cites(r"\cite  {a}", {}, 1) == r"\cite  {a}"
    passed += 1

    equivalent_doi = unify_publication("2301.00001", {
        1: [entry("doi_a", "A document with matching DOI", doi="https://doi.org/10.5555/attention")],
        2: [entry("doi_b", "A document with matching DOI", doi="10.5555/attention")],
    })
    assert len(equivalent_doi.entries) == 1
    assert equivalent_doi.entries[0].doi == "10.5555/attention"
    assert not any(conflict["field"] == "doi" for conflict in equivalent_doi.entries[0].conflicts)
    passed += 1
    print(f"SELFTEST PASSED ({passed}/{tests})")


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hợp nhất BibTeX giữa các version arXiv."
    )
    parser.add_argument("--root", type=Path, help="Thư mục corpus")
    parser.add_argument("--out", type=Path, help="Thư mục đầu ra")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.92)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args(argv)
    if args.selftest:
        _selftest()
        return 0
    if args.root is None or args.out is None:
        parser.error("--root và --out là bắt buộc nếu không dùng --selftest")
    stats = run_corpus(
        args.root,
        args.out,
        workers=args.workers,
        no_network=args.no_network,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    print(
        f"Processed {len(stats['publications'])} publications; "
        f"{stats['total']['entries_before_dedup']} -> "
        f"{stats['total']['entries_after_dedup']} entries."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
