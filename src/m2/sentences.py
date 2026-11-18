r"""Rule-based sentence splitter with protected-span masking and a byte-bounded paragraph cache.

The input is *cleaned* text (see :mod:`clean`). Algorithm:

1. **Mask** – a copy of the text is made in which every ``.``/``!``/``?`` that cannot end a
   sentence is replaced by a placeholder of the same length: inline math, ``\cite{}``/``\ref{}``,
   URLs, e-mails, DOIs, decimals and versions, abbreviations (``config.ABBREVIATIONS``),
   initials (``J. Smith``) and ellipses. Lengths are preserved, so positions found in the
   masked copy are valid in the original text ("unmasking" is free).
2. **Split** after ``[.!?]`` + optional closing quotes/brackets + whitespace, when the next
   character is a valid sentence start (uppercase, digit, ``$``, ``\``, opening bracket or
   quote). Abbreviations that may genuinely end a sentence (``etc.``, ``et al.``) split only
   before a typical sentence-initial word. A ``\cite{}`` right after a period stays with the
   preceding sentence.
3. Empty and punctuation-only pieces are dropped.

:class:`SentenceSplitter` adds an LRU cache bounded in bytes; versions of one paper share most
paragraphs, so the cache hit rate is high.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from . import config as C

_P = "\ue010"  # placeholder for a protected '.', '!' or '?'
_TRANS = str.maketrans({".": _P, "!": _P, "?": _P})

_MATH_RE = re.compile(r"\$[^$]{0,3000}\$")
_CITEREF_RE = re.compile(r"\\(?:cite|ref)\{[^{}]{0,2000}\}")
_URL_RE = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\"]+")
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+")
_NUM_RE = re.compile(r"(?<![\w.])(?:v|V|ver)?\d+(?:\.\d+)+(?![\w])|(?<=\d)\.(?=\d)")
_ELLIPSIS_RE = re.compile(r"\.{2,}|…")
_INITIAL_RE = re.compile(r"(?<![\w.\-'’])([A-Z])\.(?=(?:\s|~|\u00a0)?([A-Z][\w'’\-]*\.?|$))")
_TRAIL_PUNCT = ".,;:)]}'\""

# Words that typically start a sentence (used after terminal abbreviations and initials).
STARTERS = frozenset("""
A An The This That These Those It Its In On At By For From To We Our Us I He She They Their His
Her However Thus Hence Therefore Moreover Furthermore Finally First Second Third Then Next Also
Additionally Similarly Specifically Overall Consequently Instead Unlike Given Let Recently
Previously Despite After Before Other Most Many Some No Not Using Based Several More Further
Notably Importantly Here There Such Each All Both One Two Note If While When Since Although As
With Without Because Although Indeed Of Unfortunately Fortunately Clearly Interestingly In
Similarly Likewise Accordingly Nevertheless Nonetheless Meanwhile Conversely Alternatively
Figure Table Section Fig Tab Eq Sec Appendix Algorithm Theorem Lemma Proposition Corollary
Definition Remark Example Proof Our Their What Which Who How Why Where Yet But And Or So
""".split())

_ABBR = sorted(set(C.ABBREVIATIONS), key=len, reverse=True)
_ABBR_RE = re.compile(r"(?<![\w])(?:" + "|".join(re.escape(a) for a in _ABBR if a.endswith(".") or a.endswith("~"))
                      + r")")
_TERMINAL = C.TERMINAL_ABBREVIATIONS

_SPLIT_RE = re.compile(r"[.!?]+(?:[\"'”’)\]]|'')*(?=\s)")
_START_OK_RE = re.compile(r"\s+(?=[A-ZÀ-ÖØ-ÞΑ-Ω0-9$\\(\[\"'“‘¿¡\u00c0-\u024f])")
_CITE_AFTER_RE = re.compile(r"\s*\\cite\{[^{}]{0,2000}\}[.,;]?")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]*$")
_TRAILING_DOTS_RE = re.compile(r"(?<=[.!?])(?:\s+[.!?]+)+\s*$")


def _mask_span(chars: list[str], s: int, e: int) -> None:
    for k in range(s, e):
        if chars[k] in ".!?":
            chars[k] = _P


def _trim_end(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1] in _TRAIL_PUNCT:
        e -= 1
    return e


def mask(text: str) -> str:
    """Return a copy of ``text`` where non-terminal ``.!?`` are replaced by a placeholder."""
    if not any(c in text for c in ".!?"):
        return text
    chars = list(text)
    if "$" in text:
        for m in _MATH_RE.finditer(text):
            _mask_span(chars, m.start(), m.end())
    if "\\" in text:
        for m in _CITEREF_RE.finditer(text):
            _mask_span(chars, m.start(), m.end())
    if "://" in text or "www." in text:
        for m in _URL_RE.finditer(text):
            _mask_span(chars, m.start(), _trim_end(text, m.start(), m.end()))
    if "@" in text:
        for m in _EMAIL_RE.finditer(text):
            _mask_span(chars, m.start(), _trim_end(text, m.start(), m.end()))
    if "10." in text:
        for m in _DOI_RE.finditer(text):
            _mask_span(chars, m.start(), _trim_end(text, m.start(), m.end()))
    for m in _NUM_RE.finditer(text):
        _mask_span(chars, m.start(), m.end())
    for m in _ELLIPSIS_RE.finditer(text):
        # an ellipsis followed by a capitalised starter and preceded by a word may end a
        # sentence; keep the last dot unmasked in that case
        e = m.end()
        nxt = text[e:e + 30].lstrip()
        word = re.match(r"[A-Z][a-z]*", nxt)
        if word and word.group(0) in STARTERS and e < len(text) and text[e].isspace():
            _mask_span(chars, m.start(), e - 1)
        else:
            _mask_span(chars, m.start(), e)
    for m in _ABBR_RE.finditer(text):
        a = m.group(0)
        s, e = m.start(), m.end()
        if a.rstrip("~") in _TERMINAL or a in ("al.", "et al.", "et~al.", "etal."):
            nxt = text[e:e + 40]
            w = re.match(r"\s+([A-Z][\w'’\-]*)", nxt)
            if (w and w.group(1) in STARTERS) or not nxt.strip():
                _mask_span(chars, s, e - 1)  # keep the final dot as a boundary
                continue
        _mask_span(chars, s, e)
    for m in _INITIAL_RE.finditer(text):
        nxt = m.group(2) or ""
        word = nxt.rstrip(".")
        if word in STARTERS and not (len(word) == 1):
            continue  # "... in Appendix A. The next" -> real boundary
        if not nxt:
            continue
        _mask_span(chars, m.start(), m.end())
    return "".join(chars)


def split_sentences(text: str) -> list[str]:
    """Split cleaned text into sentences (uncached)."""
    text = text.strip()
    if not text:
        return []
    masked = mask(text)
    bounds = []
    n = len(text)
    for m in _SPLIT_RE.finditer(masked):
        end = m.end()
        ws = _START_OK_RE.match(masked, end)
        if not ws:
            continue
        # a \cite right after the period belongs to the previous sentence
        c = _CITE_AFTER_RE.match(text, end)
        if c and c.end() < n:
            after = _START_OK_RE.match(masked, c.end())
            if after:
                bounds.append((c.end(), after.end()))
            continue
        bounds.append((end, ws.end()))
    out = []
    pos = 0
    for e, nxt in bounds:
        if e <= pos:
            continue
        out.append(text[pos:e])
        pos = nxt
    out.append(text[pos:])
    res = []
    for s in out:
        s = s.strip()
        if not s:
            continue
        s = _TRAILING_DOTS_RE.sub("", s)
        if _PUNCT_ONLY_RE.match(s):
            continue
        res.append(s)
    return res


class LRUBytes:
    """LRU mapping bounded by an approximate byte size (keys and values are strings/lists)."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.size_bytes = 0
        self.data: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _cost(key, value) -> int:
        c = len(key) if isinstance(key, str) else 64
        if isinstance(value, (list, tuple)):
            for v in value:
                c += (len(v) if isinstance(v, str) else 64) + 8
        elif isinstance(value, str):
            c += len(value)
        else:
            c += 64
        return c + 48

    def get(self, key):
        v = self.data.get(key)
        if v is None:
            self.misses += 1
            return None
        self.data.move_to_end(key)
        self.hits += 1
        return v

    def put(self, key, value) -> None:
        cost = self._cost(key, value)
        if cost > self.max_bytes:
            return
        old = self.data.pop(key, None)
        if old is not None:
            self.size_bytes -= self._cost(key, old)
        self.data[key] = value
        self.size_bytes += cost
        while self.size_bytes > self.max_bytes and self.data:
            k, v = self.data.popitem(last=False)
            self.size_bytes -= self._cost(k, v)


class SentenceSplitter(LRUBytes):
    """Cached :func:`split_sentences`."""

    def __init__(self, max_bytes: int = C.SENTENCE_CACHE_BYTES):
        super().__init__(max_bytes)

    def split(self, text: str) -> list[str]:
        r = self.get(text)
        if r is not None:
            return list(r)
        r = split_sentences(text)
        self.put(text, tuple(r))
        return r
