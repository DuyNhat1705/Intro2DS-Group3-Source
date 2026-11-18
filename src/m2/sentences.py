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

_MATH_RE = re.compile(r"(?<!\\)\$[^$]{0,3000}?(?<!\\)\$")
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

_TERMINAL = C.TERMINAL_ABBREVIATIONS

_SPLIT_RE = re.compile(r"[.!?]+(?:[\"'”’)\]]|'')*(?=\s)")
_START_OK_RE = re.compile(r"\s+(?=[A-ZÀ-ÖØ-ÞΑ-Ω0-9$\\(\[\"'“‘¿¡\u00c0-\u024f])")
_CITE_AFTER_RE = re.compile(r"\s*\\cite\{[^{}]{0,2000}\}[.,;]?")
_PUNCT_ONLY_RE = re.compile(r"^[\W_]*$")
_TRAILING_DOTS_RE = re.compile(r"(?<=[.!?])(?:\s+[.!?]+)+\s*$")


def _trim_end(text: str, s: int, e: int) -> int:
    while e > s and text[e - 1] in _TRAIL_PUNCT:
        e -= 1
    return e


# Single-token abbreviations ("Fig.", "e.g.", "w.r.t.") are looked up in a set; "et al." is
# handled through its last token "al." preceded by "et".
_ABBR_SET = frozenset(a for a in C.ABBREVIATIONS if a.endswith(".") and " " not in a and "~" not in a)
_WORDDOT_RE = re.compile(r"(?<![\w.])[A-Za-z][A-Za-z.]{0,14}\.")
_STARTER_AFTER_RE = re.compile(r"\s+([A-Z][\w'’\-]*)")
_ET_RE = re.compile(r"\bet[ ~]$")
_MULTI_ABBR_RE = re.compile(r"(?:[a-z]{1,2}\.){2,4}")  # i.i.d., a.k.a., w.r.t.


def mask(text: str) -> str:
    """Return a copy of ``text`` where non-terminal ``.!?`` are replaced by a placeholder."""
    if "." not in text and "!" not in text and "?" not in text:
        return text
    spans: list[tuple[int, int]] = []
    add = spans.append
    if "$" in text:
        for m in _MATH_RE.finditer(text):
            add((m.start(), m.end()))
    if "\\" in text:
        for m in _CITEREF_RE.finditer(text):
            add((m.start(), m.end()))
    if "://" in text or "www." in text:
        for m in _URL_RE.finditer(text):
            add((m.start(), _trim_end(text, m.start(), m.end())))
    if "@" in text:
        for m in _EMAIL_RE.finditer(text):
            add((m.start(), _trim_end(text, m.start(), m.end())))
    if "10." in text:
        for m in _DOI_RE.finditer(text):
            add((m.start(), _trim_end(text, m.start(), m.end())))
    if "." in text:
        for m in _NUM_RE.finditer(text):
            add((m.start(), m.end()))
        if ".." in text or "…" in text:
            for m in _ELLIPSIS_RE.finditer(text):
                # an ellipsis followed by a capitalised starter may end a sentence: keep the
                # last dot unmasked in that case
                e = m.end()
                w = _STARTER_AFTER_RE.match(text, e)
                add((m.start(), e - 1 if (w and w.group(1) in STARTERS) else e))
        for m in _WORDDOT_RE.finditer(text):
            tok = m.group(0)
            s, e = m.start(), m.end()
            is_al = tok == "al." and _ET_RE.search(text, max(0, s - 3), s)
            if tok in _ABBR_SET or is_al or _MULTI_ABBR_RE.fullmatch(tok):
                if is_al or tok in _TERMINAL:
                    w = _STARTER_AFTER_RE.match(text, e)
                    if (w and w.group(1) in STARTERS) or not text[e:].strip():
                        add((s, e - 1))  # genuinely ends the sentence: keep the final dot
                        continue
                add((s, e))
        if "." in text:
            for m in _INITIAL_RE.finditer(text):
                nxt = m.group(2) or ""
                word = nxt.rstrip(".")
                if not nxt or (word in STARTERS and len(word) > 1):
                    continue  # "... in Appendix A. The next" -> real boundary
                add((m.start(), m.end()))
    if not spans:
        return text
    spans.sort()
    out = []
    pos = 0
    for s, e in spans:
        if e <= pos:
            continue
        if s < pos:
            s = pos
        out.append(text[pos:s])
        out.append(text[s:e].translate(_TRANS))
        pos = e
    out.append(text[pos:])
    return "".join(out)


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
