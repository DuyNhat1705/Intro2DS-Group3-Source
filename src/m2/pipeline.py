r"""Per-paper processing: ``process_paper(job) -> PaperResult``. Never raises.

Fixed stage order (prompt §4):

    gather -> preprocess -> scan/hierarchy -> clean/math -> sentences      (per version)
    -> BibTeX extraction (all versions) -> BibTeX dedup + key_map
    -> \cite renaming in contents -> element dedup -> rows

BibTeX dedup runs before element dedup so that renamed citations make identical sentences
match across versions. A failing version never loses the paper's other versions: it is retried
once in safe mode (macro expansion and conditional evaluation off) and recorded as ``partial``
on success, ``failed`` otherwise.
"""

from __future__ import annotations

import copy
import os
import re
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field

from . import config as C
from .bib_dedup import dedup_paper, rename_cites
from .bib_parse import BibEntry, parse_bib
from .bibitem import parse_bibitems
from .clean import clean_inline, clean_with_footnotes, has_residual_latex, strip_markers
from .dedup import build_tables
from .discover import PaperJob
from .gather import GatherResult, gather_files, resolve_bib_files
from .hierarchy import HEADING_KINDS, Node, TreeBuilder, TreeStats
from .io_utils import VersionFiles, content_hash, decode_bytes, scan_version_dirs
from .preprocess import (Literals, MacroTable, collect_definitions, collect_theorem_names,
                         expand_macros, mask_literals)
from .sentences import LRUBytes, SentenceSplitter

_TITLE_RE = re.compile(r"\\(?:title|icmltitle|aistatstitle|mytitle|papertitle)\*?\s*(?:\[[^\]]{0,500}\])?\s*\{")
_THEBIB_RE = re.compile(r"\\begin\s*\{thebibliography\}")
_THEBIB_END_RE = re.compile(r"\\end\s*\{thebibliography\}")
_CITE_IN_CONTENT_RE = re.compile(r"\\cite\{([^{}]*)\}")
_NOCITE_RE = re.compile(r"\\nocite\s*\{([^{}]*)\}")
_CUSTOM_ABS_RE = re.compile(r"\\([A-Za-z]*[Aa]bstract[A-Za-z]*)\*?\s*(?:\[[^\]]{0,200}\])?\s*\{")


# ---------------------------------------------------------------------------------------
# Worker-wide caches (shared across versions and papers inside one process)
# ---------------------------------------------------------------------------------------
_SPLITTER: SentenceSplitter | None = None
_PARA_CACHE: LRUBytes | None = None


def worker_caches() -> tuple[SentenceSplitter, LRUBytes]:
    global _SPLITTER, _PARA_CACHE
    if _SPLITTER is None:
        _SPLITTER = SentenceSplitter(C.SENTENCE_CACHE_BYTES // 2)
        _PARA_CACHE = LRUBytes(C.SENTENCE_CACHE_BYTES // 2)
    return _SPLITTER, _PARA_CACHE


@dataclass
class VersionParse:
    version: int = 0
    status: str = "ok"
    failure_reason: str = ""
    gather: GatherResult | None = None
    tree: Node | None = None
    stats: TreeStats = field(default_factory=TreeStats)
    title: str = ""
    bib_entries: list[BibEntry] = field(default_factory=list)
    n_bib_raw: int = 0
    n_bibitem: int = 0
    n_cited: int = 0
    cited: Counter = field(default_factory=Counter)
    times: dict = field(default_factory=dict)
    signature: str = ""
    reused_from_prev: bool = False
    retry_mode: str = ""
    notes: list[str] = field(default_factory=list)
    coverage: float = 1.0                                  # see coverage()
    reasons: list[str] = field(default_factory=list)       # why the version is not ok


def read_version_files(vf: VersionFiles) -> tuple[dict[str, str], dict[str, int]]:
    """Read every ``.tex``/``.bib`` of a version exactly once."""
    files: dict[str, str] = {}
    for rel, p in vf.files.items():
        size = vf.sizes.get(rel, 0)
        lim = C.MAX_TEX_FILE_BYTES if rel.lower().endswith(".tex") else C.MAX_BIB_FILE_BYTES
        if size > lim:
            files[rel] = ""
            continue
        try:
            with open(p, "rb") as fh:
                files[rel], _enc = decode_bytes(fh.read())
        except OSError:
            continue
    return files, dict(vf.sizes)


def clean_paragraph_text(raw: str) -> str:
    """Cleanup used for one paragraph (evaluation helper): literal masking + cleanup,
    footnote markers removed."""
    lits = Literals()
    text, _fns = clean_with_footnotes(mask_literals(raw, lits), lits)
    return strip_markers(text)


def _extract_title(pre: str, body: str, lits: Literals) -> str:
    for src in (pre, body[:20000]):
        m = _TITLE_RE.search(src)
        if not m:
            continue
        from .preprocess import match_brace
        k = match_brace(src, m.end() - 1)
        if k < 0:
            continue
        t = clean_inline(src[m.end():k - 1].replace("\\\\", " "), lits)
        if t:
            return t
    return ""


def _drop_gobbled(text: str, table: MacroTable) -> str:
    """Remove the arguments of author macros that discard them (``\\newcommand{\\remove}[1]{}``)."""
    names = [m.name for m in table.macros.values() if m.nargs >= 1 and not m.body.strip()]
    if not names or "\\" not in text:
        return text
    from .preprocess import match_brace
    rx = re.compile(r"\\(?:" + "|".join(map(re.escape, names)) + r")(?![A-Za-z@])\s*\{")
    out, pos = [], 0
    for m in rx.finditer(text):
        if m.start() < pos:
            continue
        k = match_brace(text, m.end() - 1)
        if k <= 0:
            continue
        out.append(text[pos:m.start()])
        pos = k
    out.append(text[pos:])
    return "".join(out)


def _thebibliography_text(body: str) -> str:
    out = []
    for m in _THEBIB_RE.finditer(body):
        e = _THEBIB_END_RE.search(body, m.end())
        out.append(body[m.end():e.start() if e else len(body)])
    return "\n".join(out)


def parse_version_files(files: dict[str, str], sizes: dict[str, int] | None = None,
                        meta_title: str = "", safe_mode: bool = False, version: int = 1,
                        splitter: SentenceSplitter | None = None,
                        para_cache: LRUBytes | None = None,
                        prev: VersionParse | None = None) -> VersionParse:
    """Parse one version from an in-memory file mapping (``.tex`` and ``.bib``)."""
    vp = VersionParse(version=version)
    t0 = time.perf_counter()
    gr = gather_files(files, sizes, safe_mode=safe_mode)
    vp.gather = gr
    t1 = time.perf_counter()
    vp.times["gather"] = t1 - t0
    if gr.status == "empty":
        vp.status = "empty"
        return vp
    bib_paths = sorted(r for r in files if r.lower().endswith(".bib"))
    resolve_bib_files(gr, bib_paths)
    vp.signature = content_hash("|".join(f"{f}:{h}" for f, h in gr.signature_parts)
                                + "|" + "|".join(f"{b}:{content_hash(files.get(b, ''))}" for b in gr.bib_files)
                                + ("|safe" if safe_mode else ""))
    if prev is not None and prev.tree is not None and prev.signature == vp.signature:
        # identical-version fast path: the compilation path did not change
        vp.tree = copy.deepcopy(prev.tree)
        vp.stats = copy.copy(prev.stats)
        vp.title = prev.title
        vp.bib_entries = [copy.copy(e) for e in prev.bib_entries]
        for e in vp.bib_entries:
            e.version = version
        vp.n_bib_raw, vp.n_bibitem, vp.n_cited = prev.n_bib_raw, prev.n_bibitem, prev.n_cited
        vp.cited = Counter(prev.cited)
        vp.status = prev.status
        vp.coverage, vp.reasons = prev.coverage, list(prev.reasons)
        vp.reused_from_prev = True
        vp.times["hierarchy"] = 0.0
        return vp

    # ---- preprocess: author macros
    pre, table = collect_definitions(gr.preamble, MacroTable())
    body, table = collect_definitions(gr.body, table)
    thm_names, _titles = collect_theorem_names(gr.preamble + "\n" + body[:5000])
    # class/author-specific abstract commands (\paperabstract{..}, \Abstract{..}): take the
    # argument as the abstract before macro expansion hides it
    custom_abs = ""
    if not re.search(r"\\begin\s*\{abstract\}", body[:300000]):
        for which in ("pre", "body"):
            src = pre if which == "pre" else body
            m = _CUSTOM_ABS_RE.search(src)
            if m and m.group(1) != "abstract":
                from .preprocess import match_brace
                k = match_brace(src, m.end() - 1)
                if k > 0:
                    custom_abs = src[m.end():k - 1]
                    src = src[:m.start()] + " " + src[k:]
                    if which == "pre":
                        pre = src
                    else:
                        body = src
                    break
    if not safe_mode:
        body = expand_macros(body, table)
        pre_x = expand_macros(pre, table) if ("\\title" in pre or "\\abstract" in pre) else pre
    else:
        pre_x = pre
    if table.truncated:
        vp.notes.append("macro_growth_limit")
    t2 = time.perf_counter()
    vp.times["preprocess"] = t2 - t1

    # ---- hierarchy + cleanup + sentences
    if splitter is None and para_cache is None:
        splitter, para_cache = SentenceSplitter(), LRUBytes(8 * 1024 * 1024)
    tb = TreeBuilder(gr.literals, thm_names, splitter=splitter, para_cache=para_cache,
                     safe_mode=safe_mode)
    pre_for_abs = pre_x if not re.search(r"\\begin\s*\{abstract\}|\\abstract\s*\{", body[:200000]) else ""
    if custom_abs:
        if not safe_mode:
            custom_abs = expand_macros(custom_abs, table)
        pre_for_abs = ""
    tree = tb.build(body, preamble=pre_for_abs, abstract_text=custom_abs)
    vp.title = _extract_title(pre_x, body, gr.literals) or (meta_title or "").strip()
    tree.content = vp.title
    vp.tree = tree
    vp.stats = tb.stats
    t3 = time.perf_counter()
    vp.times["hierarchy"] = t3 - t2

    # ---- bibliography of this version
    # cited keys first (from cleaned contents + \nocite): only cited entries are normalized
    cited: Counter = Counter()
    for n in tree.walk():
        if n.kind in ("sentence", "figure", "table") and "\\cite{" in n.content:
            for m in _CITE_IN_CONTENT_RE.finditer(n.content):
                for k in m.group(1).split(","):
                    if k:
                        cited[k] += 1
    for m in _NOCITE_RE.finditer(gr.body):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k != "*":
                cited[k] += 0 if k in cited else 1
    vp.cited = cited
    wanted = None if gr.nocite_all else set(cited)
    for b in gr.bib_files:
        ents, st = parse_bib(files.get(b, ""), wanted)
        for e in ents:
            e.source = "bib"
            e.version = version
        vp.bib_entries.extend(ents)
        vp.n_bib_raw += st["entries"]
    # a thebibliography kept inside a discarding macro (\newcommand{\remove}[1]{}) is not
    # compiled and must not contribute entries; the rest is parsed unexpanded so that
    # \doi{..}-like author macros stay recognizable
    tb_text = _thebibliography_text(_drop_gobbled(gr.body, table))
    if tb_text:
        items, n_items = parse_bibitems(tb_text, gr.literals, wanted=wanted, with_count=True)
        if os.environ.get("M2_REFINE_BIBITEM") == "1":
            from .bib_refine import refine_entries
            items = refine_entries(items)
        for e in items:
            e.version = version
        vp.n_bibitem = n_items
        vp.bib_entries.extend(items)
    vp.n_cited = len(vp.bib_entries)
    t4 = time.perf_counter()
    vp.times["bib"] = t4 - t3

    if gr.status == "root_fallback":
        vp.status = "root_fallback"
    vp.coverage = coverage(gr.body, tree)
    vp.reasons = quality_reasons(vp, gr, tb.stats, table.truncated, safe_mode)
    if vp.reasons:
        vp.status = "partial"
    if gr.cond.undecided:
        vp.notes.append(f"undecided_if:{gr.cond.undecided}")
    vp.notes.extend(gr.notes[:50])
    return vp


# ---------------------------------------------------------------------------------------
# Quality signals: a version is "ok" only when nothing suggests that content was lost
# ---------------------------------------------------------------------------------------
# prose words only: whitespace-separated alphabetic tokens (TikZ coordinates, key=value
# options and data stored in macro arguments do not count on either side)
_WORD_RE = re.compile(r"(?<!\S)[(\"'`]*([A-Za-z]{2,})[.,;:!?'\")\]]*(?!\S)")
_DRAWING_ENVS = r"tikzpicture|pgfpicture|axis|semilogyaxis|loglogaxis|forest|circuitikz|tikzcd|pspicture|picture"
_COV_DROP_RE = re.compile(
    r"\\begin\s*\{thebibliography\}.*?(?:\\end\s*\{thebibliography\}|$)"
    rf"|\\begin\s*\{{({_DRAWING_ENVS})\*?\}}.*?(?:\\end\s*\{{\1\*?\}}|$)"
    r"|\\pgfplotstableread\s*\{[^{}]*\}"
    r"|\\(?:author|affiliation|altaffiliation|address|email|thanks|institute|orcid|collaboration)"
    r"\*?\s*(?:\[[^\]]{0,300}\]\s*)?\{[^{}]{0,1000}\}"
    r"|\\[A-Za-z@]+\*?\s*\[[^\]]{0,500}\]"
    r"|\\(?:label|ref|eqref|cref|Cref|autoref|pageref|cite[a-zA-Z]*|nocite|includegraphics|input|include"
    r"|bibliography|bibliographystyle|usepackage|graphicspath|url|href|hypersetup|setlength|vspace|hspace"
    r"|addtolength|setcounter|definecolor)\*?\s*(?:\[[^\]]{0,300}\]\s*)?\{[^{}]{0,500}\}"
    r"|\\(?:begin|end)\s*\{[^}]{0,50}\}(?:\s*\{[^{}]{0,200}\})?"
    r"|\\[A-Za-z@]+\*?|[\ue000-\ue0ff]\d*[\ue000-\ue0ff]?", re.S)
_COVER_KINDS = frozenset({"sentence", "formula", "figure", "table", "abstract"}) | HEADING_KINDS


def _letters(s: str) -> int:
    return sum(map(len, _WORD_RE.findall(_COV_DROP_RE.sub(" ", s)))) if s else 0


def coverage(body: str, tree: Node | None) -> float:
    """Prose letters kept in the tree / prose letters of the compiled body (commands, options,
    labels, citation keys, file names, drawings and the bibliography removed). Near 1 for a
    complete parse; a body that is mostly lost scores low. 1.0 for a body without prose."""
    src = _letters(body)
    if src < 200 or tree is None:
        return 1.0
    out = sum(_letters(n.content) for n in tree.walk() if n.kind in _COVER_KINDS and n.content)
    return round(min(out / src, 9.99), 4)


def quality_reasons(vp: "VersionParse", gr: GatherResult, st: TreeStats, macro_limit: bool,
                    safe_mode: bool) -> list[str]:
    r = []
    if safe_mode:
        r.append("safe_mode")
    if st.recoveries:
        r.append(f"syntax_recovery:{st.recoveries}")
    if st.n_sentences == 0:
        r.append("no_sentences")
    if "no_begin_document" in gr.notes:
        r.append("no_begin_document")
    if gr.missing_body_includes:
        r.append(f"missing_include:{len(gr.missing_body_includes)}")
    if macro_limit:
        r.append("macro_growth_limit")
    if any(n.startswith(("include_limit", "big_file")) for n in gr.notes):
        r.append("include_or_size_limit")
    if st.n_sentences and vp.coverage < C.MIN_COVERAGE:
        r.append(f"low_coverage:{vp.coverage:.2f}")
    return r


# ---------------------------------------------------------------------------------------
@dataclass
class PaperResult:
    arxiv_id: str
    paper_key: str
    elements: list[dict] = field(default_factory=list)
    hierarchy: list[dict] = field(default_factory=list)
    bibtex: list[dict] = field(default_factory=list)
    bibtex_full: list[dict] = field(default_factory=list)
    key_map: list[dict] = field(default_factory=list)
    status: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    error: str = ""


def _status_row(job: PaperJob, v: int, vp: VersionParse | None, status: str, reason: str = "",
                elapsed: float = 0.0) -> dict:
    g = vp.gather if vp else None
    st = vp.stats if vp else TreeStats()
    t = vp.times if vp else {}
    return {
        "arxiv_id": job.arxiv_id, "version": v, "status": status, "failure_reason": reason,
        "root_file": (g.root_file if g else None) or "",
        "n_compile_files": len(g.compile_files) if g else 0,
        "n_redundant_files": len(g.redundant_files) if g else 0,
        "n_missing_includes": len(g.missing_includes) if g else 0,
        "n_nodes": st.n_nodes, "n_sentences": st.n_sentences, "n_formulas": st.n_formulas,
        "n_figures": st.n_figures, "n_tables": st.n_tables, "max_depth": st.max_depth,
        "n_lists": st.n_lists, "n_blocks": st.n_blocks, "n_verbatim": st.n_verbatim,
        "n_section_cmds": st.n_section_cmds, "n_section_nodes": st.n_section_nodes,
        "n_unbalanced_formulas": st.n_unbalanced_formulas,
        "n_bib_raw": vp.n_bib_raw if vp else 0, "n_bibitem": vp.n_bibitem if vp else 0,
        "n_cited": vp.n_cited if vp else 0,
        "bib_source": ("bib+bibitem" if (vp and vp.n_bib_raw and vp.n_bibitem) else
                       "bib" if (vp and vp.n_bib_raw) else "bibitem" if (vp and vp.n_bibitem) else "none"),
        "bib_fallback": bool(g.bib_fallback) if g else False,
        "reused_from_prev": bool(vp.reused_from_prev) if vp else False,
        "t_gather_ms": round(1000 * t.get("gather", 0.0), 2),
        "t_clean_ms": round(1000 * t.get("preprocess", 0.0), 2),
        "t_hierarchy_ms": round(1000 * t.get("hierarchy", 0.0), 2),
        "t_bib_ms": round(1000 * t.get("bib", 0.0), 2),
        "t_dedup_ms": 0.0,
        "elapsed_ms": round(1000 * elapsed, 2),
        "retry_mode": vp.retry_mode if vp else "",
        "coverage": float(vp.coverage) if vp else 0.0,
        "notes": ";".join((vp.notes if vp else [])[:20]),
    }


def _rename_tree(tree: Node, mapping: dict[str, str]) -> None:
    if not mapping:
        return
    for n in tree.walk():
        if "\\cite{" in n.content:
            n.content = rename_cites(n.content, mapping)


def process_paper(job: PaperJob, safe_mode: bool = False, use_fast_path: bool = True,
                  loaded: dict[int, tuple] | None = None) -> PaperResult:
    """Process all versions of one paper. Never raises: errors are recorded per version."""
    res = PaperResult(arxiv_id=job.arxiv_id, paper_key=job.paper_key)
    try:
        return _process(job, res, safe_mode, use_fast_path, loaded)
    except Exception as exc:  # last-resort guard: the paper is recorded as failed
        res.error = f"{type(exc).__name__}: {exc}"
        res.elements, res.hierarchy, res.bibtex, res.bibtex_full, res.key_map = [], [], [], [], []
        res.status = [_status_row(job, vi.version, None, "failed", "paper:" + res.error[:300])
                      for vi in job.versions]
        res.stats["traceback"] = traceback.format_exc()[-2000:]
        return res


def load_paper_files(job: PaperJob) -> tuple[dict[int, tuple], str]:
    """Read every file of every version once. Returns ``(loaded, signature)`` where the
    checkpoint signature hashes relative paths + sizes + contents + ``PARSER_VERSION``."""
    import hashlib
    h = hashlib.blake2b(digest_size=12)
    h.update(C.PARSER_VERSION.encode())
    loaded: dict[int, tuple] = {}
    for vi in job.versions:
        if not vi.on_disk:
            h.update(f"|v{vi.version}:missing".encode())
            continue
        vf = scan_version_dirs([d for d in vi.dirs if d.is_dir()])
        files, sizes = read_version_files(vf) if vf.tex else ({}, {})
        loaded[vi.version] = (vf, files, sizes)
        for rel in sorted(vf.files):
            h.update(f"|v{vi.version}/{rel}:{vf.sizes.get(rel, 0)}:".encode("utf-8", "surrogatepass"))
            h.update(content_hash(files.get(rel, "")).encode())
    return loaded, h.hexdigest()


def _process(job: PaperJob, res: PaperResult, safe_mode: bool, use_fast_path: bool,
             loaded: dict[int, tuple] | None = None) -> PaperResult:
    splitter, para_cache = worker_caches()
    parses: dict[int, VersionParse] = {}
    status_rows: dict[int, dict] = {}
    prev: VersionParse | None = None
    reuse = 0
    for vi in job.versions:
        v = vi.version
        t0 = time.perf_counter()
        if not vi.on_disk:
            status_rows[v] = _status_row(job, v, None, "missing",
                                         "listed in metadata/state but no folder")
            prev = None
            continue
        if loaded is not None and v in loaded:
            vf, files, sizes = loaded[v]
        else:
            vf = scan_version_dirs([d for d in vi.dirs if d.is_dir()])
            files, sizes = read_version_files(vf) if vf.tex else ({}, {})
        if not vf.tex:
            status_rows[v] = _status_row(job, v, None, "empty",
                                         "no .tex file" + (f" ({len(vf.bib)} .bib)" if vf.bib else ""))
            prev = None
            continue
        vp = None
        reason = ""
        for mode in (("normal", "safe") if not safe_mode else ("safe",)):
            try:
                vp = parse_version_files(files, sizes, meta_title=job.title, version=v,
                                         safe_mode=(mode == "safe"), splitter=splitter,
                                         para_cache=para_cache,
                                         prev=prev if use_fast_path else None)
                if mode == "safe":
                    vp.retry_mode = "safe"
                    vp.status = "partial"
                break
            except Exception as exc:
                reason = f"{type(exc).__name__}: {str(exc)[:200]}"
                vp = None
        elapsed = time.perf_counter() - t0
        if vp is None:
            status_rows[v] = _status_row(job, v, None, "failed", reason, elapsed)
            prev = None
            continue
        if reason and vp.retry_mode == "safe":
            vp.notes.append("safe_retry:" + reason)
        reuse += vp.reused_from_prev
        vf_ = vp
        if vf_.status == "empty":
            status_rows[v] = _status_row(job, v, vf_, "empty", "no root .tex", elapsed)
            prev = None
            continue
        if vf_.retry_mode == "safe" and "safe_mode" not in vf_.reasons:
            vf_.reasons.insert(0, "safe_mode")
        parses[v] = vf_
        status_rows[v] = _status_row(job, v, vf_, vf_.status, ";".join(vf_.reasons), elapsed)
        prev = vf_

    # ---- BibTeX dedup across versions + key map
    t_d0 = time.perf_counter()
    cite_counts: Counter = Counter()
    for vp in parses.values():
        cite_counts.update(vp.cited)
    entries_by_version = {v: vp.bib_entries for v, vp in parses.items()}
    dd = dedup_paper(entries_by_version, cite_counts=cite_counts)
    by_version_map: dict[int, dict[str, str]] = {}
    for (v, k), canon in dd.key_map.items():
        by_version_map.setdefault(v, {})[k] = canon
        res.key_map.append({"arxiv_id": job.arxiv_id, "version": v, "original_key": k,
                            "canonical_key": canon})
    # ---- \cite renaming, then element dedup
    trees = {}
    for v, vp in parses.items():
        if vp.tree is None:
            continue
        _rename_tree(vp.tree, {k: c for k, c in by_version_map.get(v, {}).items() if k != c})
        trees[v] = vp.tree
    res.elements, res.hierarchy = build_tables(job.paper_key, job.arxiv_id, trees)
    for e in dd.entries:
        row = e.to_row(job.arxiv_id)
        res.bibtex.append(row)
        full = dict(row)
        full.update(e.to_full_extra())
        res.bibtex_full.append(full)
    t_dedup = time.perf_counter() - t_d0
    n_leaf_dup = 0
    for v, tree in trees.items():
        seen = Counter((n.kind, n.content) for n in tree.walk()
                       if n.kind in ("sentence", "formula", "figure", "table"))
        n_leaf_dup += sum(c - 1 for c in seen.values() if c > 1)
    for v in sorted(status_rows):
        r = status_rows[v]
        if v in parses:
            r["t_dedup_ms"] = round(1000 * t_dedup / max(len(parses), 1), 2)
        res.status.append(r)
    nodes_total = sum(vp.stats.n_nodes for vp in parses.values())
    n_sent = sum(1 for e in res.elements if e["kind"] == "sentence")
    resid = sum(1 for e in res.elements if e["kind"] == "sentence" and has_residual_latex(e["content"]))
    res.stats = {
        "nodes_before": nodes_total, "elements_after": len(res.elements),
        "bib_before": sum(len(vp.bib_entries) for vp in parses.values()),
        "bib_raw": sum(vp.n_bib_raw + vp.n_bibitem for vp in parses.values()),
        "bib_after": len(res.bibtex), "reused_versions": reuse,
        "leaf_repeats_within_version": n_leaf_dup,
        "sentences_distinct": n_sent, "sentences_residual_latex": resid,
        "cache_hits": para_cache.hits, "cache_misses": para_cache.misses,
    }
    return res
