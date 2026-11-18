r"""Tolerant hand-written ``.bib`` parser (``@string``, ``#`` concatenation, malformed-entry recovery).

* brace and quote matching, ``@string`` macros (month abbreviations predefined), ``#``
  concatenation, ``@comment`` and ``@preamble`` skipped;
* a malformed entry drops only that entry: a value may not run into the next line that starts
  with ``@type{`` -- parsing resumes there;
* title and authors are converted LaTeX -> Unicode with protective braces removed; authors are
  normalized to ``"First Last"`` (``"Last, First"`` and ``"Last, Jr, First"`` inverted,
  ``others`` / ``et al.`` dropped);
* ``year`` is an int (from ``year``, else ``date``, else the ``YYMM`` of an arXiv identifier);
  ``doi`` is lowercased without resolver prefix; ``journal`` falls back to ``booktitle``.

:class:`BibEntry` is shared with :mod:`bibitem` and :mod:`bib_dedup`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .clean import clean_inline

MONTHS = {m: n for m, n in zip(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
    ("January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"))}

_ENTRY_START_RE = re.compile(r"@[ \t]*([A-Za-z]+)[ \t\n]*([{(])")
_NEXT_ENTRY_RE = re.compile(r"\n[ \t]*@[ \t]*[A-Za-z]+[ \t]*[{(]")
_FIELD_NAME_RE = re.compile(r"[ \t\n,]*([A-Za-z][\w\-:.+]*)[ \t\n]*=[ \t\n]*")
_YEAR_RE = re.compile(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)")
_NEW_ARXIV_RE = re.compile(r"(?<![\d.])(\d{2})(\d{2})\.(\d{4,5})(?:v\d+)?(?![\d])")
_OLD_ARXIV_RE = re.compile(r"\b([a-z\-]+(?:\.[A-Z]{2})?)/(\d{2})(\d{2})(\d{3})(?:v\d+)?\b")
_ARXIV_TEXT_RE = re.compile(
    r"(?:arxiv(?:\.org)?(?:/abs|/pdf)?[:/ ]\s*|abs/|eprint\s*)"
    r"((?:\d{4}\.\d{4,5})(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)", re.I)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"{}<>\]\\]+")
_DOI_PREFIX_RE = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:\s*", re.I)
_AND_RE = re.compile(r"\s+and\s+", re.I)
_ETAL_RE = re.compile(r"^(?:others|et\.?\s*al\.?|and others)$", re.I)
_SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "2nd", "3rd"}


@dataclass
class BibEntry:
    key: str = ""
    entry_type: str = "misc"
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str = ""
    journal: str = ""
    booktitle: str = ""
    url: str = ""
    eprint: str = ""
    volume: str = ""
    pages: str = ""
    publisher: str = ""
    source: str = "bib"            # bib / bibitem
    parse_method: str = "bib"      # bib / bibinfo / newblock / regex
    confidence: float = 1.0
    version: int = 0
    versions: list[int] = field(default_factory=list)
    merged_keys: list[str] = field(default_factory=list)
    raw: str = ""

    def to_row(self, arxiv_id: str) -> dict:
        return {"arxiv_id": arxiv_id, "citation_key": self.key, "entry_type": self.entry_type,
                "title": self.title or None, "authors": list(self.authors),
                "year": self.year, "doi": self.doi or None, "journal": self.journal or None,
                "url": self.url or None}

    def to_full_extra(self) -> dict:
        return {"eprint": self.eprint or None, "booktitle": self.booktitle or None,
                "volume": self.volume or None, "pages": self.pages or None,
                "publisher": self.publisher or None, "source": self.source,
                "parse_method": self.parse_method, "confidence": float(self.confidence),
                "versions": sorted(set(self.versions or [self.version])),
                "merged_keys": sorted(set(self.merged_keys or [self.key]))}


# ---------------------------------------------------------------------------------------
# Normalization helpers (also used by bibitem.py)
# ---------------------------------------------------------------------------------------
def split_top_and(s: str) -> list[str]:
    """Split an author list on ``and`` at brace depth 0."""
    out, depth, start = [], 0, 0
    i = 0
    n = len(s)
    low = s.lower()
    while i < n:
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif depth == 0 and c in " \t\n" and low.startswith("and", i + 1) and i + 4 <= n and \
                (i + 4 == n or s[i + 4] in " \t\n"):
            out.append(s[start:i])
            i += 4
            start = i
            continue
        i += 1
    out.append(s[start:])
    return [x.strip() for x in out if x.strip()]


def _split_commas_top(s: str) -> list[str]:
    out, depth, start = [], 0, 0
    for i, c in enumerate(s):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(s[start:i])
            start = i + 1
    out.append(s[start:])
    return [x.strip() for x in out]


def normalize_name(raw: str) -> str:
    """BibTeX name -> ``"First Last"`` (Unicode)."""
    raw = raw.strip().strip(",").strip()
    if not raw:
        return ""
    if raw.startswith("{") and raw.endswith("}") and raw.count("{") == 1:
        return clean_inline(raw[1:-1])  # corporate author
    parts = _split_commas_top(raw)
    if len(parts) >= 3:
        last, jr, first = parts[0], parts[1], ", ".join(parts[2:])
        name = f"{first} {last} {jr}"
    elif len(parts) == 2:
        last, first = parts
        if first.lower().strip(".") in {s.strip(".") for s in _SUFFIXES}:
            name = f"{last} {first}"
        else:
            name = f"{first} {last}"
    else:
        name = parts[0]
    out = clean_inline(name)
    out = re.sub(r"\s+", " ", out.replace("~", " ")).strip(" ,;")
    return out


def normalize_authors(raw: str) -> list[str]:
    out = []
    for a in split_top_and(raw):
        if _ETAL_RE.match(a.strip()):
            continue
        n = normalize_name(a)
        if n and not _ETAL_RE.match(n):
            out.append(n)
    return out


def normalize_doi(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = _DOI_PREFIX_RE.sub("", s)
    m = _DOI_RE.search(s)
    if not m:
        return ""
    return m.group(0).rstrip(".").lower()


def find_arxiv(text: str) -> str:
    if not text:
        return ""
    m = _ARXIV_TEXT_RE.search(text)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return ""


def year_from_arxiv(eprint: str) -> int | None:
    if not eprint:
        return None
    m = _NEW_ARXIV_RE.search(eprint)
    if m:
        return 2000 + int(m.group(1))
    m = _OLD_ARXIV_RE.search(eprint)
    if m:
        yy = int(m.group(2))
        return 1900 + yy if yy >= 91 else 2000 + yy
    return None


def parse_year(s: str) -> int | None:
    m = _YEAR_RE.search(s or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------
class _Malformed(Exception):
    pass


def _match_brace_bib(text: str, i: int) -> int:
    """``text[i] == '{'``; index after the matching ``}``. Stops at a new entry line."""
    depth = 0
    n = len(text)
    j = i
    while j < n:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return j + 1
        elif c == "\n":
            k = j + 1
            while k < n and text[k] in " \t":
                k += 1
            if k < n and text[k] == "@" and _ENTRY_START_RE.match(text, k):
                raise _Malformed
        elif c == "\\":
            j += 1
        j += 1
    raise _Malformed


def _read_value(text: str, i: int, strings: dict[str, str]) -> tuple[str, int]:
    parts = []
    n = len(text)
    while True:
        while i < n and text[i] in " \t\n":
            i += 1
        if i >= n:
            raise _Malformed
        c = text[i]
        if c == "{":
            k = _match_brace_bib(text, i)
            parts.append(text[i + 1:k - 1])
            i = k
        elif c == '"':
            j = i + 1
            depth = 0
            while j < n:
                ch = text[j]
                if ch == "\\":
                    j += 2
                    continue
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                elif ch == '"' and depth <= 0:
                    break
                elif ch == "\n" and text[j + 1:j + 2] == "@":
                    raise _Malformed
                j += 1
            if j >= n:
                raise _Malformed
            parts.append(text[i + 1:j])
            i = j + 1
        else:
            m = re.compile(r"[\w\-:.+/']+").match(text, i)
            if not m:
                raise _Malformed
            tok = m.group(0)
            low = tok.lower()
            if tok.isdigit():
                parts.append(tok)
            elif low in strings:
                parts.append(strings[low])
            elif low[:3] in MONTHS and len(low) <= 4:
                parts.append(MONTHS[low[:3]])
            else:
                parts.append(tok)
            i = m.end()
        while i < n and text[i] in " \t\n":
            i += 1
        if i < n and text[i] == "#":
            i += 1
            continue
        return "".join(parts), i


def _entry_from_fields(etype: str, key: str, f: dict[str, str]) -> BibEntry:
    e = BibEntry(key=key, entry_type=etype, source="bib", parse_method="bib", confidence=1.0)
    if f.get("title"):
        e.title = clean_inline(f["title"])
    auth = f.get("author") or f.get("editor") or ""
    if auth:
        e.authors = normalize_authors(auth)
    e.journal = clean_inline(f.get("journal") or f.get("journaltitle") or "")
    e.booktitle = clean_inline(f.get("booktitle") or "")
    if not e.journal and e.booktitle:
        e.journal = e.booktitle
    e.volume = clean_inline(f.get("volume", ""))
    e.pages = clean_inline(f.get("pages", ""))
    e.publisher = clean_inline(f.get("publisher") or f.get("organization") or f.get("institution")
                               or f.get("school") or "")
    e.doi = normalize_doi(f.get("doi", ""))
    url = f.get("url", "").strip()
    if not url:
        hp = f.get("howpublished", "")
        m = re.search(r"\\url\s*\{([^}]*)\}", hp)
        if m:
            url = m.group(1)
    e.url = url.strip()
    ep = f.get("eprint", "").strip()
    if ep and (f.get("archiveprefix", "").lower() in ("", "arxiv") or re.match(r"\d{4}\.\d{4,5}|[a-z\-]+/\d{7}", ep)):
        e.eprint = re.sub(r"^arxiv:\s*", "", ep, flags=re.I)
        e.eprint = re.sub(r"v\d+$", "", e.eprint)
    if not e.eprint:
        for k in ("journal", "note", "url", "howpublished", "volume", "arxiv", "pages", "number"):
            a = find_arxiv(f.get(k, ""))
            if a:
                e.eprint = a
                break
    if not e.doi:
        for k in ("url", "note", "howpublished"):
            if "10." in f.get(k, ""):
                e.doi = normalize_doi(f.get(k, ""))
                if e.doi:
                    break
    e.year = parse_year(f.get("year", "")) or parse_year(f.get("date", ""))
    if e.year is None and e.eprint:
        e.year = year_from_arxiv(e.eprint)
    return e


def parse_bib(text: str) -> tuple[list[BibEntry], dict]:
    """Parse a ``.bib`` file. Returns ``(entries, stats)``; never raises."""
    entries: list[BibEntry] = []
    stats = {"entries": 0, "malformed": 0, "strings": 0}
    if not text or "@" not in text:
        return entries, stats
    text = text.replace("\r\n", "\n")
    strings: dict[str, str] = {}
    i = 0
    n = len(text)
    while i < n:
        j = text.find("@", i)
        if j < 0:
            break
        m = _ENTRY_START_RE.match(text, j)
        if not m:
            i = j + 1
            continue
        etype = m.group(1).lower()
        opener = m.group(2)
        body_start = m.end()
        closer = "}" if opener == "{" else ")"
        try:
            if etype in ("comment", "preamble"):
                if opener == "{":
                    i = _match_brace_bib(text, m.end() - 1)
                else:
                    k = text.find(")", body_start)
                    i = n if k < 0 else k + 1
                continue
            if etype == "string":
                fm = _FIELD_NAME_RE.match(text, body_start)
                if not fm:
                    raise _Malformed
                val, k = _read_value(text, fm.end(), strings)
                strings[fm.group(1).lower()] = val
                stats["strings"] += 1
                while k < n and text[k] in " \t\n":
                    k += 1
                i = k + 1 if k < n and text[k] == closer else k
                continue
            # regular entry: key
            k = body_start
            km = re.compile(r"[ \t\n]*([^,\s{}()=\"#%]*)[ \t\n]*,").match(text, k)
            if not km:
                raise _Malformed
            key = km.group(1).strip()
            k = km.end()
            fields: dict[str, str] = {}
            while True:
                while k < n and text[k] in " \t\n,":
                    k += 1
                if k < n and text[k] == closer:
                    k += 1
                    break
                fm = _FIELD_NAME_RE.match(text, k)
                if not fm:
                    raise _Malformed
                val, k = _read_value(text, fm.end(), strings)
                fields[fm.group(1).lower()] = val
            i = k
            if not key:
                stats["malformed"] += 1
                continue
            e = _entry_from_fields(etype, key, fields)
            e.raw = ""
            entries.append(e)
            stats["entries"] += 1
        except _Malformed:
            stats["malformed"] += 1
            nm = _NEXT_ENTRY_RE.search(text, j + 1)
            i = nm.start() + 1 if nm else n
        except (IndexError, ValueError, RecursionError):
            stats["malformed"] += 1
            nm = _NEXT_ENTRY_RE.search(text, j + 1)
            i = nm.start() + 1 if nm else n
    return entries, stats
