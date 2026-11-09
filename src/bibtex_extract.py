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
