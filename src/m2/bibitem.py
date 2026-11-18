r"""Conversion of ``\bibitem`` entries into BibTeX fields (``\bibinfo``, ``\newblock``, regex heuristics)
with a confidence score.

Methods, by decreasing reliability (``parse_method``):

1. ``bibinfo`` -- revtex/apsrev ``.bbl`` pasted into the ``.tex`` (``\bibinfo{author}{..}``,
   ``\bibfield``) and apacite (``\APACrefatitle``, ``\APACjournalVolNumPages``): fields are read
   directly;
2. ``newblock`` -- natbib/plain styles: block 1 = authors, block 2 = title, blocks 3+ = venue;
3. ``regex`` -- everything else: quoted (````..''``, ``".."``) or italic title, APA
   ``Authors (Year). Title. Venue``, ``Authors: Title``, ``Authors. Title. Venue`` and comma
   separated ``Authors, Title, Venue`` layouts; journal-like segments (``Phys. Rev. Lett. 90``)
   are never taken as titles.

DOI, in-text arXiv identifier (-> internal ``eprint``), URL, volume and pages are extracted by
regex; the year comes from the natbib label, then from the fields/text, then from the arXiv
``YYMM``. Every entry carries ``confidence`` in [0, 1].
"""

from __future__ import annotations

import re

from .bib_parse import (BibEntry, find_arxiv, normalize_doi, normalize_name, split_top_and,
                        year_from_arxiv)
from .clean import clean_inline
from .preprocess import Literals, match_brace, match_bracket

_BIBITEM_RE = re.compile(r"\\bibitem(?![A-Za-z@])")
_JUNK_RE = re.compile(
    r"\\(?:BibitemOpen|EOS|endbibitem|unskip|urlprefix|BIBentryALTinterwordspacing|"
    r"BIBentrySTDinterwordspacing|natexlab\{[^{}]*\}|bibAnnoteFile\s*\{[^{}]*\}|"
    r"BibitemShut\s*\{[^{}]*\}|PrintBackRefs\s*\{[^{}]*\}|APACinsertmetastar\s*\{[^{}]*\}|"
    r"CurrentBib|showISBNx\{[^{}]*\}|showISSN\{[^{}]*\}|showLCCN\{[^{}]*\}|showDOI\{[^{}]*\}|"
    r"providecommand|selectlanguage\{[^{}]*\}|href@noop\s*\{\s*\}|Eprint\s*\{[^{}]*\}\s*\{[^{}]*\}|"
    r"newblock)(?![A-Za-z@])")
_BIBINFO_RE = re.compile(r"\\bibinfo\s*\{\s*([A-Za-z ]+?)\s*\}\s*\{")
_YEAR_ANY_RE = re.compile(r"(?<![\w\-–/.:])(1[5-9]\d{2}|20\d{2})[a-z]?(?![\w\-–/])")
_YEAR_PAREN_RE = re.compile(r"\((?:[^()]{0,40}?[\s,])?(1[5-9]\d{2}|20\d{2})[a-z]?\)")
_LABEL_YEAR_RE = re.compile(r"(?:\(|\b)(1[5-9]\d{2}|20\d{2})[a-z]?(?:\)|\b)")
_URL_RE = re.compile(r"\\url\s*\{([^{}]+)\}|\\href\s*\{([^{}]+)\}|(https?://[^\s{}\\]+)")
_DOI_TEXT_RE = re.compile(r"10\.\d{4,9}/[^\s,;\"{}<>\]\\]+")
_OLD_ARXIV_BARE_RE = re.compile(
    r"\b((?:hep-th|hep-ph|hep-ex|hep-lat|gr-qc|astro-ph|cond-mat|quant-ph|math-ph|nucl-th|nucl-ex|"
    r"math|cs|physics|nlin|q-bio|q-fin|stat)(?:\.[A-Z]{2})?/\d{7})\b")
_EMPH_START_RE = re.compile(r"\\(?:emph|textit|textsl)\s*\{|\{\\(?:it|em|sl|itshape)\b\s*")
_QUOTE_RES = [
    re.compile(r"``(.{3,600}?)''"),
    re.compile(r"\\enquote\s*\{(.{3,600}?)\}"),
    re.compile(r"(?<![\w$])\"(.{3,600}?)\""),
    re.compile(r"“(.{3,600}?)”"),
    re.compile(r"\ufffd(.{3,600}?)\ufffd"),
]
_PUBLISHER_RE = re.compile(r"\b(Press|Publishing|Publishers?|Springer|Wiley|Elsevier|Prentice|Addison|"
                           r"Birkh(?:ä|a)user|CRC|Dover|North[- ]Holland|World Scientific|Pearson|"
                           r"McGraw|Academic|Verlag|Cambridge University|Oxford University|MIT Press|"
                           r"Princeton University|Routledge|Kluwer|Taylor|Chapman)\b")
_INPROC_RE = re.compile(r"(?:^|\s|\()(?:In|in)\s*:?\s+|Proceedings|Proc\.|Conference|Workshop|Symposium|"
                        r"Conf\.|Advances in Neural|NeurIPS|ICML|ICLR|CVPR|ICCV|ECCV|AAAI|IJCAI|ACL|EMNLP|"
                        r"NAACL|KDD|SIGIR|WWW")
_JOURNAL_WORDS = {
    "phys", "rev", "lett", "j", "nucl", "jhep", "jcap", "eur", "astrophys", "astron", "nature",
    "science", "proc", "ann", "math", "commun", "class", "quant", "grav", "mon", "not", "roy",
    "soc", "acta", "appl", "theor", "chem", "opt", "mod", "int", "journal", "trans", "ieee",
    "sci", "acad", "natl", "nat", "res", "rep", "rept", "adv", "comput", "stat", "mech", "lond",
    "london", "amer", "ser", "z", "zh", "eksp", "teor", "fiz", "prog", "suppl", "physica",
    "ptep", "jetp", "sov", "usp", "nuovo", "cim", "lect", "notes", "front", "biol", "med",
    "anal", "geom", "topol", "algebra", "combin", "pure", "funct", "differ", "equ", "probab",
    "theory", "relat", "fields", "inform", "syst", "control", "optim", "siam", "acm", "arxiv",
    "preprint", "e-print", "eprint", "rmp", "prl", "prd", "prb", "pra", "prc", "pre", "prx",
    "apj", "apjl", "aj", "mnras", "a&a", "aa", "pasp", "icarus", "plb", "npb", "epjc", "cqg",
}
_PARTICLES = {"von", "van", "der", "de", "da", "di", "del", "della", "le", "la", "du", "dos", "das",
              "ter", "ten", "den", "al", "el", "bin", "ibn", "y", "st.", "mc"}
_INITIAL_TOKEN_RE = re.compile(r"^(?:[A-Z][a-z]?\.(?:-?[A-Z][a-z]?\.)*|[A-Z]\.?-[A-Z]\.?|[A-Z]{1,2})$")
_NAME_TOKEN_RE = re.compile(r"^[A-ZÀ-ÖØ-ÞĀ-ſ][\w'’\-\.À-ÿĀ-ſ]*$")


# ---------------------------------------------------------------------------------------
def _brace_after(text: str, i: int) -> tuple[str | None, int]:
    """``text[i-1] == '{'`` (regex ended on the brace); return content and end."""
    k = match_brace(text, i - 1, 6000)
    if k < 0:
        return None, i
    return text[i:k - 1], k


def _emph_group(text: str, m: re.Match) -> tuple[str, int, int] | None:
    s = m.start()
    tok = m.group(0)
    if tok.startswith("{"):
        k = match_brace(text, s, 3000)
        if k < 0:
            return None
        return text[m.end():k - 1], s, k
    k = match_brace(text, m.end() - 1, 3000)
    if k < 0:
        return None
    return text[m.end():k - 1], s, k


def _strip_title(t: str) -> str:
    t = t.strip()
    t = re.sub(r"^[\s\"'“”‘’,.;:]+", "", t)
    t = re.sub(r"[\s,.;:]+$", "", t)
    t = t.strip("\"'“”‘’ \ufffd")
    t = re.sub(r"[\s,;:]+$", "", t)
    return t


def _clean(s: str) -> str:
    return clean_inline(s).strip()


def _looks_like_name(seg: str) -> bool:
    seg = seg.strip().strip(".,;:").strip()
    if not seg:
        return False
    if seg.lower() in ("et al", "et al.", "others", "...", "…"):
        return True
    toks = seg.replace("~", " ").split()
    if not 1 <= len(toks) <= 6:
        return False
    caps = 0
    for t in toks:
        tl = t.lower()
        if tl in _PARTICLES:
            continue
        if _INITIAL_TOKEN_RE.match(t) or _NAME_TOKEN_RE.match(t):
            caps += 1
            continue
        return False
    if caps == 0:
        return False
    # a name has at most a few long lowercase-free words; reject journal-like phrases
    words = [w.strip(".").lower() for w in toks]
    if sum(1 for w in words if w in _JOURNAL_WORDS) >= 2:
        return False
    return True


def _looks_like_author_list(s: str) -> bool:
    s = s.strip().strip(".,:;")
    if not s:
        return False
    s2 = re.sub(r"\s*(?:,\s*)?(?:\band\b|&)\s*", ",", s)
    segs = [x for x in (y.strip() for y in s2.split(",")) if x.strip(".…;: ")]
    if not segs:
        return False
    return all(_looks_like_name(x) for x in segs)


def _journal_like(seg: str) -> bool:
    s = seg.strip()
    if not s:
        return True
    toks = re.findall(r"[A-Za-z&]+", s)
    if not toks:
        return True
    low = [t.lower() for t in toks]
    jw = sum(1 for t in low if t in _JOURNAL_WORDS)
    abbrev = len(re.findall(r"\b[A-Z][a-z]{0,6}\.", s))
    has_digits = bool(re.search(r"\d", s))
    if re.match(r"^(?:arXiv|preprint|e-?print|Preprint)", s):
        return True
    if len(toks) <= 6 and (jw >= max(1, len(toks) // 2) or abbrev >= 2) and (has_digits or abbrev >= 2):
        return True
    if len(toks) <= 3 and has_digits:
        return True
    return False


def _split_inverted(segs: list[str]) -> list[str] | None:
    """``["Kirkpatrick", "J.", "Pascanu", "R."]`` -> ``["J. Kirkpatrick", "R. Pascanu"]``."""
    if len(segs) < 2:
        return None
    out = []
    i = 0
    while i < len(segs):
        last = segs[i].strip()
        if i + 1 < len(segs):
            first = segs[i + 1].strip()
            toks = first.split()
            if toks and all(_INITIAL_TOKEN_RE.match(t) or re.match(r"^[A-Z]\.?[A-Z]?\.?$", t) for t in toks):
                out.append(f"{first} {last}")
                i += 2
                continue
        return None
    return out


def parse_plain_authors(s: str) -> list[str]:
    s = _clean(s) if "\\" in s or "{" in s else s
    s = s.replace("~", " ")
    s = re.sub(r"\.\.\.|…", ",", s)
    s = re.sub(r"\bet\s*al\.?|\band\s+others\b", ",", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:")
    if not s:
        return []
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        out = []
        for p in parts:
            if "," in p:
                out.append(normalize_name(p))
            else:
                out.append(p)
        return [x for x in out if x]
    # BibTeX-like "Last, First and Last, First"
    if re.search(r"\band\b", s) and all("," in a for a in split_top_and(s)) and \
            all(len(a.split(",")) == 2 for a in split_top_and(s)):
        return [normalize_name(a) for a in split_top_and(s)]
    s2 = re.sub(r"\s*,?\s*(?:\band\b|&)\s+", ", ", s)
    segs = [x.strip() for x in s2.split(",") if x.strip()]
    inv = _split_inverted(segs)
    if inv:
        return inv
    out = []
    for x in segs:
        x = x.strip(" .") if not re.search(r"\b[A-Z]\.$", x) else x.strip()
        if x.lower() in ("jr", "jr.", "sr", "sr.") and out:
            out[-1] += " " + x
            continue
        if x:
            out.append(x)
    return out


def _year_from(text: str) -> int | None:
    t = _DOI_TEXT_RE.sub(" ", text)
    t = re.sub(r"https?://\S+|arXiv:\s*\S+|\d{4}\.\d{4,5}|[a-z\-]+/\d{7}", " ", t)
    t = re.sub(r"\d+\s*[-–]+\s*\d+", " ", t)  # page ranges
    ps = _YEAR_PAREN_RE.findall(t)
    if ps:
        return int(ps[-1])
    ys = _YEAR_ANY_RE.findall(t)
    if ys:
        cand = [int(y) for y in ys if int(y) <= 2027]
        if cand:
            return cand[-1]
    return None


_PERIOD_RE = re.compile(r"\.\s+(?=[A-Z0-9])")


def _period_split(s: str) -> tuple[str, str] | None:
    """Split ``"P. Clement, J. A. Nohel. Title. Venue"`` after the first period that ends a
    word of two or more letters (not an initial)."""
    for m in _PERIOD_RE.finditer(s):
        w = re.search(r"([\w'’\-]+)$", s[:m.start()])
        if w and len(w.group(1)) >= 2 and not re.fullmatch(r"[A-Z][a-z]?", w.group(1)) \
                and w.group(1).lower() not in _JOURNAL_WORDS:
            return s[:m.start()], s[m.end():]
    return None


def _venue_journal(venue_raw: str) -> tuple[str, bool]:
    """Journal / booktitle from the text after the title. Returns ``(name, is_proceedings)``."""
    if not venue_raw.strip():
        return "", False
    m = _EMPH_START_RE.search(venue_raw)
    pre = venue_raw[:m.start()] if m else venue_raw
    inproc = bool(re.search(r"(?:^|[\s,.(])[Ii]n\s*:?\s", pre[:40])) or bool(_INPROC_RE.search(venue_raw[:200]))
    if m and len(_clean(pre).strip(" ,.;:")) <= 12:
        g = _emph_group(venue_raw, m)
        if g:
            name = _strip_title(_clean(g[0]))
            if name:
                return name, inproc or bool(re.search(r"Proc|Conference|Workshop|Symposium", name))
    v = _clean(venue_raw)
    v = re.sub(r"^(?:[,.;:]\s*)*(?:[Ii]n\s*:?\s+)?", "", v)
    v = re.sub(r"\s*(?:doi|DOI)\s*:.*$", "", v)
    # journal name = text before the first volume/number/year/page indication
    cut = re.search(r"\s(?:vol\.|Vol\.|volume|no\.|pp\.|p\.)|\s?\d|\(|,", v)
    name = v[:cut.start()] if cut else v
    name = name.strip(" ,.;:")
    if len(name) < 2:
        return "", inproc
    return name, inproc or bool(re.search(r"Proc|Conference|Workshop|Symposium", name))


def _common_fields(e: BibEntry, raw: str) -> None:
    for m in _URL_RE.finditer(raw):
        u = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        if u and not e.url:
            e.url = u
    m = re.search(r"\\doibase\s*(10\.\S+?)\s*\}|\\doi\s*\{([^{}]+)\}|doi\s*:\s*(10\.\S+)|doi\.org/(10\.[^\s{}]+)",
                  raw, re.I)
    if m:
        e.doi = normalize_doi(next(g for g in m.groups() if g))
    if not e.doi:
        m = _DOI_TEXT_RE.search(raw)
        if m:
            e.doi = normalize_doi(m.group(0))
    e.eprint = find_arxiv(raw)
    if not e.eprint:
        m = _OLD_ARXIV_BARE_RE.search(raw)
        if m:
            e.eprint = m.group(1)
    vm = re.search(r"\\textbf\s*\{\s*(\d{1,5})\s*\}|\{\\bf\s+(\d{1,5})\s*\}|\bvol(?:ume)?\.?\s*~?(\d{1,5})", raw)
    if vm:
        e.volume = next(g for g in vm.groups() if g)
    pm = re.search(r"\bpp?\.\s*~?([A-Z]?\d+\s*[-–]+\s*[A-Z]?\d+|\d+)|(?<![\d.])([A-Z]?\d{1,6})\s*--?\s*([A-Z]?\d{1,6})(?![\d])", raw)
    if pm:
        if pm.group(1):
            e.pages = pm.group(1).replace("--", "–").replace(" ", "")
        else:
            e.pages = f"{pm.group(2)}–{pm.group(3)}"


def _type_and_conf(e: BibEntry, base: float, venue_text: str, inproc: bool) -> None:
    if inproc:
        e.entry_type = "inproceedings"
    elif e.journal or e.volume:
        e.entry_type = "article"
    elif _PUBLISHER_RE.search(venue_text or ""):
        e.entry_type = "book"
        m = _PUBLISHER_RE.search(venue_text)
        if m and not e.publisher:
            seg = venue_text[max(0, venue_text.rfind(",", 0, m.start()) + 1):]
            e.publisher = _strip_title(_clean(seg.split(",")[0]))
    elif e.eprint:
        e.entry_type = "misc"
    else:
        e.entry_type = "misc"
    c = base
    if not e.title:
        c -= 0.25
    if not e.authors:
        c -= 0.3
    elif any(len(a.split()) > 5 for a in e.authors):
        c -= 0.2
    if e.year is None:
        c -= 0.15
    if e.title and (len(e.title) > 300 or _journal_like(e.title)):
        c -= 0.25
    e.confidence = round(max(0.0, min(1.0, c)), 3)


# ---------------------------------------------------------------------------------------
def _bibinfo(e: BibEntry, raw: str, label_year: int | None) -> None:
    fields: dict[str, list[str]] = {}
    for m in _BIBINFO_RE.finditer(raw):
        val, _k = _brace_after(raw, m.end())
        if val is not None:
            fields.setdefault(m.group(1).strip().lower(), []).append(val)
    # apacite
    apa_auth = re.search(r"\\begin\{APACrefauthors\}(.*?)\\end\{APACrefauthors\}", raw, re.S)
    if apa_auth:
        a = apa_auth.group(1)
        a = re.sub(r"\\BPBI\b\s*", ". ", a)
        a = re.sub(r"\\BBA\b\s*(?:\{\})?", " & ", a)
        a = re.sub(r"\\BCBL\b\s*(?:\{\})?|\\BCBT\b\s*(?:\{\})?", ",", a)
        a = re.sub(r"\\BOthers\s*\{[^{}]*\}", "", a)
        fields.setdefault("authorlist", []).append(a)
    m = re.search(r"\\APACref[ab]title\s*\{", raw)
    if m:
        v, _ = _brace_after(raw, m.end())
        if v is not None:
            fields.setdefault("title", []).append(v)
    m = re.search(r"\\APACjournalVolNumPages\s*\{", raw)
    if m:
        v, k = _brace_after(raw, m.end())
        if v is not None:
            fields.setdefault("journal", []).append(v)
            m2 = re.match(r"\s*\{([^{}]*)\}", raw[k:])
            if m2:
                fields.setdefault("volume", []).append(m2.group(1))
    m = re.search(r"\\APACrefYear(?:MonthDay)?\s*\{\s*(\d{4})", raw)
    if m:
        fields.setdefault("year", []).append(m.group(1))

    authors: list[str] = []
    for a in fields.get("author", []):
        ca = _clean(a.replace("\\ ", " "))
        if ca.count(",") >= 2 or re.search(r"\band\b", ca) and ca.count(",") >= 1:
            authors += parse_plain_authors(ca)
        elif "," in ca:
            authors.append(normalize_name(ca))
        elif ca:
            authors.append(re.sub(r"\s+", " ", ca))
    for a in fields.get("authorlist", []):
        authors += parse_plain_authors(a)
    e.authors = [a for a in (x.strip(" ,.;") if not re.search(r"\b[A-Z]\.$", x) else x.strip(" ,;")
                             for x in authors) if a and not re.match(r"(?i)^(et al\.?|others)$", a)]
    if fields.get("title"):
        e.title = _strip_title(_clean(fields["title"][0]))
    j = fields.get("journal") or fields.get("booktitle") or []
    if j:
        e.journal = _strip_title(_clean(j[0]))
    if fields.get("booktitle"):
        e.booktitle = _strip_title(_clean(fields["booktitle"][0]))
    if fields.get("volume"):
        e.volume = _clean(fields["volume"][0])
    if fields.get("pages"):
        e.pages = _clean(fields["pages"][0]).replace(" ", "")
    if fields.get("publisher"):
        e.publisher = _clean(fields["publisher"][0])
    y = None
    for yv in fields.get("year", []):
        mm = re.search(r"(1[5-9]\d{2}|20\d{2})", yv)
        if mm:
            y = int(mm.group(1))
            break
    e.year = y or label_year or _year_from(raw)
    if fields.get("doi") and not e.doi:
        e.doi = normalize_doi(fields["doi"][0])
    if fields.get("eprint") and not e.eprint:
        e.eprint = find_arxiv("arXiv:" + fields["eprint"][0]) or fields["eprint"][0].strip()
    inproc = bool(fields.get("booktitle")) and not fields.get("journal")
    _type_and_conf(e, 0.95, " ".join(fields.get("publisher", [])), inproc)


def _newblock(e: BibEntry, raw: str, label_year: int | None) -> None:
    blocks = [b for b in re.split(r"\\newblock(?![A-Za-z@])", raw)]
    a_raw = blocks[0]
    t_raw = blocks[1] if len(blocks) > 1 else ""
    rest = " ".join(blocks[2:])
    a_clean = _clean(a_raw)
    ym = re.search(r"\(\s*((?:1[5-9]|20)\d{2})[a-z]?\s*\)\.?\s*$", a_clean)
    a_year = int(ym.group(1)) if ym else None
    if ym:
        a_clean = a_clean[:ym.start()]
    e.authors = parse_plain_authors(a_clean)
    t_strip = t_raw.lstrip()
    m = _EMPH_START_RE.match(t_strip)
    title = ""
    if m:
        g = _emph_group(t_strip, m)
        if g:
            title = _clean(g[0])
            after = t_strip[g[2]:]
            rest = after + " " + rest
    if not title:
        q = None
        for qre in _QUOTE_RES:
            q = qre.search(t_raw)
            if q:
                break
        if q and len(_clean(t_raw[:q.start()]).strip(" ,.")) <= 3:
            title = _clean(q.group(1))
            rest = t_raw[q.end():] + " " + rest
        else:
            ct = _clean(t_raw)
            # "Title. Journal, ..." inside one block (2-block entries)
            if len(blocks) == 2:
                parts = re.split(r"(?<=[a-z0-9)?])\.\s+(?=[A-Z])", ct, maxsplit=1)
                title = parts[0]
                if len(parts) > 1:
                    rest = parts[1]
            else:
                title = ct
    e.title = _strip_title(title)
    if e.title and _journal_like(e.title) and not rest.strip():
        rest, e.title = e.title, ""
    e.journal, inproc = _venue_journal(rest)
    if inproc:
        e.booktitle = e.journal
    e.year = label_year or a_year or _year_from(rest) or _year_from(t_raw)
    _type_and_conf(e, 0.85, _clean(rest), inproc)


def _regex(e: BibEntry, raw: str, label_year: int | None) -> None:
    title = ""
    a_part = ""
    venue_raw = ""
    # 1. quoted title
    for qre in _QUOTE_RES:
        q = qre.search(raw)
        if q and _looks_like_author_list(_clean(raw[:q.start()]).strip(" ,.:;")) or \
                (q and len(_clean(raw[:q.start()])) < 200 and q.start() > 0):
            title = _clean(q.group(1))
            a_part = raw[:q.start()]
            venue_raw = raw[q.end():]
            break
    # 2. italic title (or italic journal)
    if not title:
        m = _EMPH_START_RE.search(raw)
        if m:
            g = _emph_group(raw, m)
            if g:
                pre = _clean(raw[:m.start()]).strip(" ,.:;")
                segs = [s.strip() for s in re.split(r",(?![^()]*\))", pre) if s.strip()]
                if not pre or (segs and _looks_like_author_list(pre)):
                    title = _clean(g[0])
                    a_part = raw[:m.start()]
                    venue_raw = raw[g[2]:]
                elif ":" in pre and _looks_like_author_list(pre.split(":")[0]):
                    a_part = pre.split(":")[0]
                    title = pre.split(":", 1)[1]
                    venue_raw = raw[m.start():]
                else:
                    # last comma segment is the title, the italic group the journal
                    names = []
                    rest_segs = list(segs)
                    while rest_segs and _looks_like_name(rest_segs[0]):
                        names.append(rest_segs.pop(0))
                    if names and rest_segs:
                        a_part = ", ".join(names)
                        title = ", ".join(rest_segs)
                        venue_raw = raw[m.start():]
                    elif names:
                        a_part = ", ".join(names)
                        title = _clean(g[0])
                        venue_raw = raw[g[2]:]
    cleaned = _clean(raw)
    # 3. plain layouts
    if not title and not a_part:
        apa = re.match(r"^(?P<a>.+?)\s*\(\s*(?P<y>(?:1[5-9]|20)\d{2})[a-z]?\s*\)\s*[.,:]?\s*(?P<r>.*)$", cleaned)
        if apa and _looks_like_author_list(apa.group("a")):
            a_part = apa.group("a")
            r = apa.group("r")
            parts = re.split(r"(?<=[\w)?!])[.?!]\s+(?=[A-Z0-9(\"“])", r, maxsplit=1)
            title = parts[0]
            venue_raw = parts[1] if len(parts) > 1 else ""
            if not label_year:
                label_year = int(apa.group("y"))
        else:
            colon = re.match(r"^(?P<a>[^:]{2,300}?):\s+(?P<r>.+)$", cleaned)
            if colon and (";" in colon.group("a") or _looks_like_author_list(colon.group("a"))):
                a_part = colon.group("a")
                r = colon.group("r")
                parts = re.split(r"(?<=[\w)?!])\.\s+(?=[A-Z])", r, maxsplit=1)
                if len(parts) > 1:
                    title, venue_raw = parts[0], parts[1]
                else:
                    segs = [s.strip() for s in r.split(",")]
                    tl = [segs[0]]
                    k = 1
                    while k < len(segs) and not _journal_like(segs[k]) and not re.search(r"\d", segs[k]) \
                            and not re.match(r"(?i)(arxiv|preprint|in\b)", segs[k]):
                        tl.append(segs[k])
                        k += 1
                    title = ", ".join(tl)
                    venue_raw = ", ".join(segs[k:])
            else:
                period = _period_split(cleaned)
                if period and _looks_like_author_list(period[0]) and \
                        not _journal_like(period[1][:60]):
                    a_part = period[0]
                    r = period[1]
                    parts = re.split(r"(?<=[\w)?!])\.\s+(?=[A-Z0-9])", r, maxsplit=1)
                    title = parts[0]
                    venue_raw = parts[1] if len(parts) > 1 else ""
                else:
                    segs = [s.strip() for s in re.split(r",(?![^()]*\))", cleaned)]
                    names = []
                    k = 0
                    while k < len(segs):
                        s = segs[k]
                        s2 = re.sub(r"^(?:and|&)\s+", "", s)
                        if _looks_like_name(s2) or (" and " in s2 and _looks_like_author_list(s2)):
                            names.append(s2)
                            k += 1
                            continue
                        break
                    a_part = ", ".join(names)
                    if k < len(segs) and names:
                        cand = segs[k]
                        if _journal_like(cand) or re.match(r"(?i)^(arxiv|preprint|in\s)", cand):
                            title = ""
                            venue_raw = ", ".join(segs[k:])
                        else:
                            tl = [cand]
                            j = k + 1
                            while j < len(segs) and not _journal_like(segs[j]) and \
                                    not re.search(r"\d|preprint|arXiv|Proc|In ", segs[j]) and len(tl) < 3:
                                tl.append(segs[j])
                                j += 1
                            title = ", ".join(tl)
                            venue_raw = ", ".join(segs[j:])
                    else:
                        venue_raw = ", ".join(segs[k:])
    e.title = _strip_title(_clean(title) if "\\" in title or "{" in title else title)
    if e.title and _journal_like(e.title) and len(e.title.split()) <= 4:
        venue_raw = e.title + ", " + venue_raw
        e.title = ""
    e.authors = parse_plain_authors(a_part) if a_part else []
    e.journal, inproc = _venue_journal(venue_raw)
    if inproc:
        e.booktitle = e.journal
    e.year = label_year or _year_from(venue_raw if venue_raw else cleaned) or _year_from(cleaned)
    _type_and_conf(e, 0.7, _clean(venue_raw), inproc)


def bibitem_to_entry(key: str, label: str | None, body: str) -> BibEntry:
    raw = _JUNK_RE.sub(" ", body) if "\\" in body else body
    raw = re.sub(r"\\newblock(?![A-Za-z@])", "\\\\newblock", raw)  # keep newblock markers
    e = BibEntry(key=key, source="bibitem")
    label_year = None
    if label:
        m = _LABEL_YEAR_RE.search(label.replace("{", " ").replace("}", " "))
        if m:
            label_year = int(m.group(1))
    _common_fields(e, body)
    if "\\bibinfo" in body or "\\APACrefatitle" in body or "\\APACrefbtitle" in body:
        e.parse_method = "bibinfo"
        _bibinfo(e, body, label_year)
    elif "\\newblock" in body:
        e.parse_method = "newblock"
        _newblock(e, re.sub(r"\\(?:BibitemOpen|EOS|BibitemShut\s*\{[^{}]*\}|urlprefix)", " ", body),
                  label_year)
    else:
        e.parse_method = "regex"
        _regex(e, raw, label_year)
    if e.year is None and e.eprint:
        e.year = year_from_arxiv(e.eprint)
    if not e.journal and e.booktitle:
        e.journal = e.booktitle
    e.raw = body.strip()[:2000]
    return e


def parse_bibitems(text: str, lits: Literals | None = None, wanted: set[str] | None = None,
                   with_count: bool = False):
    """Parse every ``\\bibitem[label]{key} text`` of ``text``.

    With ``wanted`` only those keys are converted. With ``with_count`` the result is
    ``(entries, number_of_bibitems)``."""
    out: list[BibEntry] = []
    count = 0
    if "\\bibitem" not in text:
        return (out, 0) if with_count else out
    starts = list(_BIBITEM_RE.finditer(text))
    for idx, m in enumerate(starts):
        i = m.end()
        end = starts[idx + 1].start() if idx + 1 < len(starts) else len(text)
        j = i
        while j < end and text[j] in " \t\n":
            j += 1
        label = None
        if j < end and text[j] == "[":
            k = match_bracket(text, j, 3000)
            if k > 0:
                label = text[j + 1:k - 1]
                j = k
        while j < end and text[j] in " \t\n":
            j += 1
        if j >= end or text[j] != "{":
            continue
        k = match_brace(text, j, 500)
        if k < 0:
            continue
        key = text[j + 1:k - 1].strip()
        if not key:
            continue
        count += 1
        if wanted is not None and key not in wanted:
            continue
        body = text[k:end]
        if lits is not None:
            body = lits.restore_inline(body)
        try:
            out.append(bibitem_to_entry(key, label, body))
        except (IndexError, ValueError, RecursionError, StopIteration):
            out.append(BibEntry(key=key, source="bibitem", parse_method="regex", confidence=0.0,
                                raw=body[:2000]))
    return (out, count) if with_count else out
