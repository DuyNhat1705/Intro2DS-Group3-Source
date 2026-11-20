r"""Optional AnyStyle refinement of low-confidence bibitem entries (off by default; never required).

Only entries that look wrong (missing title or authors, a venue taken as the title, implausibly
long author names, confidence < 0.6) are sent, in one batch, to the AnyStyle CLI
(``gem install anystyle-cli``). A returned field is accepted **only if its text appears in the
original bibitem** (after normalization), so no external knowledge can enter the entries. The
unrefined entry is kept on any error; every accepted change is logged. Results are cached by the
hash of the bibitem text.

The parser works fully without Ruby/AnyStyle: :func:`anystyle_available` returns False and
:func:`refine_entries` returns the input unchanged.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import subprocess
import tempfile
import unicodedata
from pathlib import Path

from .bib_parse import BibEntry, normalize_name
from .clean import clean_inline
from .io_utils import content_hash

_CACHE: dict[str, dict] = {}
LOG: list[str] = []


def anystyle_available() -> bool:
    return shutil.which("anystyle") is not None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return re.sub(r"[^0-9a-z]+", " ", s).strip()


def needs_refinement(e: BibEntry) -> bool:
    if e.source != "bibitem":
        return False
    if not e.title or not e.authors or e.confidence < 0.6:
        return True
    if any(len(a.split()) > 5 for a in e.authors):
        return True
    return False


def _run_anystyle(texts: list[str]) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "refs.txt"
        p.write_text("\n".join(t.replace("\n", " ") for t in texts) + "\n", encoding="utf-8")
        out = subprocess.run(["anystyle", "-f", "json", "parse", str(p)], capture_output=True,
                             text=True, timeout=300, check=False)
        if out.returncode != 0:
            raise RuntimeError(out.stderr[:300])
        data = json.loads(out.stdout or "[]")
    return data if isinstance(data, list) else []


def refine_entries(entries: list[BibEntry]) -> list[BibEntry]:
    """Return refined copies (same order). Entries that need no refinement are returned as is."""
    if not entries or not anystyle_available():
        return entries
    todo = [(i, e) for i, e in enumerate(entries) if needs_refinement(e) and e.raw]
    plain = {i: clean_inline(e.raw) for i, e in todo}
    pending = [(i, e) for i, e in todo if content_hash(plain[i]) not in _CACHE]
    if pending:
        try:
            parsed = _run_anystyle([plain[i] for i, _e in pending])
            for (i, _e), rec in zip(pending, parsed):
                _CACHE[content_hash(plain[i])] = rec
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            LOG.append(f"anystyle_error:{exc}")
            return entries
    out = list(entries)
    for i, e in todo:
        rec = _CACHE.get(content_hash(plain[i]))
        if not rec:
            continue
        src = _norm(plain[i])
        new = copy.copy(e)
        changed = []
        t = " ".join(rec.get("title") or [])
        if t and (not e.title or e.confidence < 0.6) and _norm(t) and _norm(t) in src:
            new.title = t.strip(" .,")
            changed.append("title")
        auth = rec.get("author") or []
        names = []
        for a in auth:
            nm = " ".join(x for x in (a.get("given"), a.get("particle"), a.get("family")) if x)
            if nm and all(_norm(p) in src for p in nm.split() if len(_norm(p)) > 1):
                names.append(normalize_name(nm))
        if names and (not e.authors or any(len(x.split()) > 5 for x in e.authors)):
            new.authors = names
            changed.append("authors")
        if changed:
            new.parse_method = e.parse_method + "+anystyle"
            LOG.append(f"refined:{e.key}:{','.join(changed)}")
            out[i] = new
    return out
