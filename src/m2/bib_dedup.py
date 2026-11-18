r"""Reference-entry deduplication within and across versions, canonical keys, field union and key map.

Within one publication, entries of all versions are grouped with union-find. Two entries denote
the same reference if any rule holds:

1. same citation key;
2. same DOI;
3. same in-text arXiv identifier;
4. equal normalized title (lowercase alphanumerics), compatible year (|delta| <= 1 or missing) and
   matching first-author surname when both have one;
5. ``rapidfuzz`` title ratio >= :data:`FUZZY_THRESHOLD` with the same year/author conditions
   (candidates blocked on the first title token).

Vetoes: rules 2-5 never merge entries whose DOIs differ, whose arXiv IDs differ, or whose years
differ by more than one.

Canonical key = the key used in the *latest* version containing the group (ties: most cited,
then alphabetical). Fields are unioned: each field takes the non-empty value of the latest
version that has it (a ``.bib`` entry outranks a ``\bibitem`` entry of the same version);
``authors`` takes the most complete list.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from .bib_parse import BibEntry

FUZZY_THRESHOLD = 93  # tuned on gold_pairs.py (sweep 90..97)
_FIELDS = ("title", "entry_type", "year", "doi", "journal", "booktitle", "url", "eprint",
           "volume", "pages", "publisher")


def norm_title(t: str) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    t = re.sub(r"\$[^$]*\$", lambda m: re.sub(r"\\[a-z]+", " ", m.group(0)), t)
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def surname(entry: BibEntry) -> str:
    if not entry.authors:
        return ""
    toks = norm_title(entry.authors[0]).split()
    toks = [t for t in toks if t not in ("jr", "sr", "ii", "iii")]
    return toks[-1] if toks else ""


def _eprint(e: BibEntry) -> str:
    return re.sub(r"v\d+$", "", (e.eprint or "").strip().lower())


def _doi(e: BibEntry) -> str:
    return (e.doi or "").strip().lower()


def _vetoed(a: BibEntry, b: BibEntry) -> bool:
    if a.doi and b.doi and _doi(a) != _doi(b):
        return True
    ea, eb = _eprint(a), _eprint(b)
    if ea and eb and ea != eb:
        return True
    if a.year and b.year and abs(a.year - b.year) > 1:
        return True
    return False


def _title_conditions(a: BibEntry, b: BibEntry) -> bool:
    sa, sb = surname(a), surname(b)
    if sa and sb and sa != sb:
        # allow initial-only vs full surname mismatch only when one is a prefix (e.g. "ho"/"ho")
        return False
    return True


def same_reference(a: BibEntry, b: BibEntry, threshold: int = FUZZY_THRESHOLD) -> bool:
    if a.key and a.key == b.key:
        return True
    if _vetoed(a, b):
        return False
    if a.doi and _doi(a) == _doi(b):
        return True
    ea = _eprint(a)
    if ea and ea == _eprint(b):
        return True
    ta, tb = norm_title(a.title), norm_title(b.title)
    if not ta or not tb or len(ta) < 8 or len(tb) < 8:
        return False
    if not _title_conditions(a, b):
        return False
    if ta == tb:
        return True
    if abs(len(ta) - len(tb)) > max(len(ta), len(tb)) * 0.2:
        return False
    return fuzz.ratio(ta, tb) >= threshold


@dataclass
class DedupResult:
    entries: list[BibEntry] = field(default_factory=list)
    key_map: dict[tuple[int, str], str] = field(default_factory=dict)
    n_before: int = 0


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def dedup_paper(entries_by_version: dict[int, list[BibEntry]],
                cite_counts: Counter | dict | None = None,
                threshold: int = FUZZY_THRESHOLD) -> DedupResult:
    cite_counts = cite_counts or {}
    flat: list[BibEntry] = []
    for v in sorted(entries_by_version):
        for e in entries_by_version[v]:
            e.version = v
            flat.append(e)
    res = DedupResult(n_before=len(flat))
    if not flat:
        return res
    uf = _UF(len(flat))
    by_key: dict[str, int] = {}
    by_doi: dict[str, int] = {}
    by_ep: dict[str, int] = {}
    titles = [norm_title(e.title) for e in flat]
    for i, e in enumerate(flat):
        if e.key in by_key:
            uf.union(by_key[e.key], i)
        else:
            by_key[e.key] = i
    # rules 2/3 with vetoes
    for i, e in enumerate(flat):
        for idx, val in ((by_doi, _doi(e)), (by_ep, _eprint(e))):
            if not val:
                continue
            j = idx.get(val)
            if j is None:
                idx[val] = i
            elif not _vetoed(flat[j], e):
                uf.union(j, i)
    # rules 4/5, blocked on the first title token
    blocks: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(titles):
        if len(t) >= 8:
            blocks[t.split()[0]].append(i)
    for idxs in blocks.values():
        if len(idxs) < 2:
            continue
        if len(idxs) > 400:
            idxs = idxs[:400]
        for x in range(len(idxs)):
            a = flat[idxs[x]]
            for y in range(x + 1, len(idxs)):
                i, j = idxs[x], idxs[y]
                if uf.find(i) == uf.find(j):
                    continue
                b = flat[j]
                if a.key == b.key:
                    continue
                if same_reference(a, b, threshold):
                    uf.union(i, j)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(len(flat)):
        groups[uf.find(i)].append(i)

    for root in sorted(groups):
        members = [flat[i] for i in groups[root]]
        latest = max(e.version for e in members)
        keys_latest = Counter(e.key for e in members if e.version == latest)
        canon = sorted(keys_latest, key=lambda k: (-cite_counts.get(k, 0), -keys_latest[k], k))[0]
        # field priority: latest version first, .bib before \bibitem, higher confidence first
        ordered = sorted(members, key=lambda e: (-e.version, e.source != "bib", -e.confidence))
        m = BibEntry(key=canon, source="bib" if any(e.source == "bib" for e in members) else "bibitem")
        for f in _FIELDS:
            for e in ordered:
                val = getattr(e, f)
                if val not in (None, "", "misc") or (f == "entry_type" and e is ordered[-1]):
                    setattr(m, f, val)
                    break
        if not m.entry_type:
            m.entry_type = "misc"
        best_auth = max(ordered, key=lambda e: len(e.authors))  # stable: latest wins ties
        m.authors = list(best_auth.authors)
        src = next((e for e in ordered if e.title), ordered[0])
        m.parse_method = src.parse_method
        m.confidence = src.confidence
        m.version = latest
        m.versions = sorted({e.version for e in members})
        m.merged_keys = sorted({e.key for e in members})
        res.entries.append(m)
        for e in members:
            res.key_map[(e.version, e.key)] = canon
    return res


_CITE_RE = re.compile(r"\\cite\{([^{}]*)\}")


def rename_cites(text: str, mapping: dict[str, str]) -> str:
    """Rename ``\\cite{}`` keys with ``mapping``; duplicate keys collapse, order is kept."""
    if not mapping or "\\cite{" not in text:
        return text

    def rep(m: re.Match) -> str:
        out = []
        for k in m.group(1).split(","):
            k = k.strip()
            if not k:
                continue
            k = mapping.get(k, k)
            if k not in out:
                out.append(k)
        return "\\cite{" + ",".join(out) + "}"
    return _CITE_RE.sub(rep, text)
