r"""Multi-file gathering: root detection, transitive include resolution, simple conditional
evaluation, and inlining of the compilation path into one string with a source map.

Input is an in-memory mapping ``relative path -> raw text`` of the ``.tex`` files of one version
(each file is read once by the caller). The result reproduces what LaTeX would compile:

1. ``(n)`` Drive copies are collapsed onto their canonical name (equal after comment stripping
   -> one file; different -> the name without ``(n)`` wins, the copy is redundant);
2. every file is preprocessed (literal masking, comment stripping, ``\endinput``);
3. the root is the file with an uncommented ``\documentclass`` / ``\documentstyle`` (ranked when
   there are several), with fallbacks (``root_fallback``);
4. includes are followed transitively (DFS, cycle guard, depth <= 32) after evaluating simple
   conditionals, and inlined at the command position;
5. the combined text is split into preamble and body; ``.bib`` files are resolved.
"""

from __future__ import annotations

import bisect
import posixpath
import re
from dataclasses import dataclass, field

from . import config as C
from .io_utils import content_hash, split_dup_suffix
from .preprocess import (CondNotes, Literals, clean_file_text, evaluate_all, normalize_newlines,
                         strip_comments)

_DOCCLASS_RE = re.compile(r"\\document(?:class|style)\s*(?:\\expandafter\s*)?(?:\[[^\]]{0,2000}\])?\s*\{\s*([^}\s]{1,200})\s*\}")
_BEGIN_DOC_RE = re.compile(r"\\begin\s*\{document\}")
_END_DOC_RE = re.compile(r"\\end\s*\{document\}")
_INCLUDE_RE = re.compile(
    r"\\(input|include|subfile|InputIfFileExists|import|subimport|includefrom|subincludefrom|"
    r"inputfrom|subinputfrom|loadglsentries)\*?(?![A-Za-z@])")
_INCLUDEONLY_RE = re.compile(r"\\includeonly\s*\{([^}]*)\}")
_BIB_RE = re.compile(r"\\(?:bibliography|nobibliography|addbibresource|addglobalbib|bibliographyfile)"
                     r"\s*(?:\[[^\]]{0,200}\])?\s*\{([^}]{0,2000})\}")
_NOCITE_ALL_RE = re.compile(r"\\nocite\s*\{\s*\*\s*\}")
_DEFINED_RE = re.compile(r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand)\*?\s*\{?\s*\\([A-Za-z@]+)"
                         r"|\\(?:g|e|x)?def\s*\\([A-Za-z@]+)|\\let\s*\\([A-Za-z@]+)")
_NEWIF_RE = re.compile(r"\\newif\s*\\if([A-Za-z@]+)")
_SETIF_RE = re.compile(r"\\([A-Za-z@]+?)(true|false)(?![A-Za-z@])")

MAX_COMBINED = 40 * 1024 * 1024
_BRACE_RE = re.compile(r"(?<!\\)[{}]")
_SW_OPEN, _SW_CLOSE = "", ""      # spliced switch assignments, removed after use
_SW_SPAN_RE = re.compile("[^]*")


def _search_top(rx: re.Pattern, text: str, start: int = 0) -> re.Match | None:
    """First match of ``rx`` at brace depth 0 (relative to ``start``), so that
    ``\\def\\bd{\\begin{document}}`` in a preamble is not taken for the real one. Falls back to
    the first match anywhere when none is at depth 0 (e.g. unbalanced braces)."""
    first = None
    depth = 0
    pos = start
    for m in rx.finditer(text, start):
        if first is None:
            first = m
        for b in _BRACE_RE.finditer(text, pos, m.start()):
            depth = depth + 1 if b.group() == "{" else max(0, depth - 1)
        pos = m.start()
        if depth == 0:
            return m
    return first


@dataclass
class GatherResult:
    status: str = "ok"                       # ok / root_fallback / empty
    root_file: str | None = None
    compile_files: list[str] = field(default_factory=list)
    redundant_files: list[str] = field(default_factory=list)
    excluded_files: list[str] = field(default_factory=list)
    missing_includes: list[str] = field(default_factory=list)
    missing_body_includes: list[str] = field(default_factory=list)   # .tex content lost in the body
    bib_names: list[str] = field(default_factory=list)
    bib_files: list[str] = field(default_factory=list)
    bib_fallback: bool = False
    nocite_all: bool = False
    path_macros: dict = field(default_factory=dict)       # literal macros usable in file names
    header_file: str = ""                                 # documentclass file input by the root
    text: str = ""
    preamble: str = ""
    body: str = ""
    body_offset: int = 0
    notes: list[str] = field(default_factory=list)
    literals: Literals = field(default_factory=Literals)
    cond: CondNotes = field(default_factory=CondNotes)
    signature_parts: list[tuple[str, str]] = field(default_factory=list)
    # source map: segment start offsets in ``text`` and (file, line at segment start, text of file
    # segment start offset) -- see :meth:`locate`.
    _seg_starts: list[int] = field(default_factory=list)
    _seg_info: list[tuple[str, int]] = field(default_factory=list)
    _file_text: dict[str, str] = field(default_factory=dict)

    def locate(self, offset: int) -> tuple[str, int]:
        """Map an offset of :attr:`text` to ``(file, line)`` (line counts lines of the comment
        stripped file, which is exact for files without whole-line comments)."""
        if not self._seg_starts:
            return ("?", 0)
        i = bisect.bisect_right(self._seg_starts, offset) - 1
        i = max(i, 0)
        f, file_off = self._seg_info[i]
        ftxt = self._file_text.get(f, "")
        local = file_off + (offset - self._seg_starts[i])
        return f, ftxt.count("\n", 0, max(0, local)) + 1


# ---------------------------------------------------------------------------------------
def _canonicalize(files: dict[str, str], res: GatherResult,
                  stripped: dict[str, str]) -> dict[str, str]:
    """Collapse ``(n)`` copies. Returns canonical name -> actual relative path."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for rel in files:
        d, base = posixpath.split(rel)
        canon_base, n = split_dup_suffix(base)
        # folder components may carry "(n)" too
        dparts = [split_dup_suffix(p)[0] for p in d.split("/")] if d else []
        canon = "/".join(dparts + [canon_base]) if dparts else canon_base
        n_total = n + sum(split_dup_suffix(p)[1] for p in (d.split("/") if d else []))
        groups.setdefault(canon, []).append((n_total, rel))
    mapping: dict[str, str] = {}
    for canon, members in groups.items():
        members.sort()
        keep = members[0][1]
        mapping[canon] = keep
        for _n, rel in members[1:]:
            res.redundant_files.append(rel)
            if stripped.get(rel) == stripped.get(keep):
                res.notes.append(f"dup_equal:{rel}")
            else:
                res.notes.append(f"dup_differ:{rel}")
    return mapping


_PATH_DEF_RE = re.compile(r"\\(?:g|e|x)?def\s*\\([A-Za-z@]+)\s*\{([^{}\\%\n]{0,200})\}"
                          r"|\\(?:re)?newcommand\*?\s*\{?\s*\\([A-Za-z@]+)\s*\}?\s*\{([^{}\\%\n]{0,200})\}")
_PATH_MACRO_RE = re.compile(r"\\([A-Za-z@]+)\s*")


def _path_macros(texts) -> dict[str, list[str]]:
    """Literal-valued macros (``\\def\\root{arxiv/}``) usable inside include paths."""
    out: dict[str, list[str]] = {}
    for t in texts:
        if "def" not in t and "newcommand" not in t:
            continue
        for m in _PATH_DEF_RE.finditer(t):
            k, v = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            vals = out.setdefault(k, [])
            if v.strip() not in vals:
                vals.append(v.strip())
    return out


def _resolve(name: str, cur_dir: str, root_dir: str, canon: dict[str, str],
             lower: dict[str, str], exts: tuple[str, ...] = (".tex",),
             macros: dict[str, list[str]] | None = None) -> str | None:
    if macros and "\\" in name:
        # \input{\arxivroot sec/intro}: try every literal value of the path macros
        names = [name]
        for _ in range(4):
            nxt = []
            for nm in names:
                m = _PATH_MACRO_RE.search(nm)
                if not m:
                    nxt.append(nm)
                    continue
                for v in macros.get(m.group(1), [None]):
                    if v is not None:
                        nxt.append(nm[:m.start()] + v + nm[m.end():])
            names = nxt[:16]
        for nm in names:
            if "\\" not in nm:
                r = _resolve(nm, cur_dir, root_dir, canon, lower, exts)
                if r:
                    return r
        return None
    name = name.strip().strip('"').strip("'").strip()
    if not name or "\\" in name or "#" in name:
        return None
    name = name.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    cands = []
    has_ext = posixpath.splitext(name)[1].lower() in exts
    for base in (root_dir, cur_dir, ""):
        p = posixpath.normpath(posixpath.join(base, name)) if base else posixpath.normpath(name)
        if p.startswith("../"):
            p = p.lstrip("./")
        cands.append(p)
        if not has_ext:
            for e in exts:
                cands.append(p + e)
    for c in cands:
        if c in canon:
            return c
    for c in cands:
        hit = lower.get(c.lower())
        if hit:
            return hit
    # last resort: unique basename match anywhere in the version
    base = posixpath.basename(cands[1] if not has_ext and len(cands) > 1 else cands[0]).lower()
    hits = [k for k in canon if posixpath.basename(k).lower() == base]
    if len(hits) == 1:
        return hits[0]
    return None


def _read_braced(text: str, i: int) -> tuple[str | None, int]:
    """Read ``{...}`` (no nesting expected in file names) starting at/after ``i``."""
    j = i
    n = len(text)
    while j < n and text[j] in " \t\n":
        j += 1
    if j < n and text[j] == "{":
        k = text.find("}", j + 1, j + 1 + 2000)
        if k < 0:
            return None, i
        inner = text[j + 1:k]
        # allow one level of nested braces such as {\jobname}
        while inner.count("{") > inner.count("}"):
            k2 = text.find("}", k + 1, k + 1 + 2000)
            if k2 < 0:
                break
            k = k2
            inner = text[j + 1:k]
        return inner, k + 1
    return None, i


_BARE_NAME_RE = re.compile(r"[ \t]*([^\s{}\\%$]+)")


def _parse_include(text: str, m: re.Match) -> tuple[str | None, str, int]:
    """Return ``(target name, kind, end offset)`` for the include command matched by ``m``."""
    cmd = m.group(1)
    i = m.end()
    if cmd in ("import", "subimport", "includefrom", "subincludefrom", "inputfrom", "subinputfrom"):
        d, i2 = _read_braced(text, i)
        f, i3 = _read_braced(text, i2) if d is not None else (None, i2)
        if d is None or f is None:
            return None, cmd, m.end()
        d = d.strip()
        return (posixpath.join(d, f.strip()) if d else f.strip()), cmd, i3
    if cmd == "InputIfFileExists":
        f, i2 = _read_braced(text, i)
        if f is None:
            return None, cmd, m.end()
        # skip the two following arguments {then}{else} only when they are trivial
        return f, cmd, i2
    name, i2 = _read_braced(text, i)
    if name is not None:
        return name, cmd, i2
    if cmd in ("input", "include"):
        mm = _BARE_NAME_RE.match(text, i)
        if mm and mm.group(1):
            return mm.group(1), cmd, mm.end()
    return None, cmd, m.end()


def _document_body(t: str) -> str | None:
    """Body of a file that is itself a complete document (subfiles / nested documents)."""
    if not _DOCCLASS_RE.search(t):
        return None
    b = _search_top(_BEGIN_DOC_RE, t)
    if not b:
        return None
    e = _search_top(_END_DOC_RE, t, b.end())
    return t[b.end():e.start() if e else len(t)]


def _reachable(root: str, stripped: dict[str, str], canon: dict[str, str],
               lower: dict[str, str], macros: dict[str, list[str]] | None = None) -> set[str]:
    """Cheap include closure (no conditional evaluation) used for root ranking."""
    seen = {root}
    stack = [root]
    root_dir = posixpath.dirname(root)
    while stack:
        f = stack.pop()
        t = stripped.get(canon.get(f, f), "")
        if "\\" not in t:
            continue
        for m in _INCLUDE_RE.finditer(t):
            name, _k, _e = _parse_include(t, m)
            if not name:
                continue
            r = _resolve(name, posixpath.dirname(f), root_dir, canon, lower, macros=macros)
            if r and r not in seen:
                seen.add(r)
                stack.append(r)
    return seen


def _pick_root(stripped: dict[str, str], canon: dict[str, str], lower: dict[str, str],
               res: GatherResult, macros: dict[str, list[str]] | None = None) -> str | None:
    root = _pick_class_root(stripped, canon, lower, res, macros)
    if root is None or res.status == "root_fallback":
        return root
    # The file with \documentclass is a header \input by the real main file, which has no
    # \documentclass of its own: "\input{preamble} \begin{document} ..." or
    # "\input{header} \input{body} \input{footer}". LaTeX compiles the includer.
    root_reach = _reachable(root, stripped, canon, lower, macros)
    includers = [c for c, rel in canon.items()
                 if c not in root_reach and c.lower().endswith(".tex") and not _DOCCLASS_RE.search(stripped[rel])
                 and root in _reachable(c, stripped, canon, lower, macros)]
    if not includers:
        return root
    reach = {c: _reachable(c, stripped, canon, lower, macros) for c in includers}
    if _search_top(_BEGIN_DOC_RE, stripped[canon[root]]):
        # the class file is a whole document: switch only when the includer adds other files
        # (header/body/footer split), not for a pure wrapper such as "\input{template/sample}"
        includers = [c for c in includers if reach[c] - root_reach - {c}]
        if not includers:
            return root
    top = [c for c in includers if not any(o != c and c in reach[o] for o in includers)] or includers
    top.sort(key=lambda c: (posixpath.splitext(posixpath.basename(c))[0].lower() in C.ROOT_PREFERRED_NAMES,
                            len(reach[c]), len(stripped[canon[c]]), -len(c)), reverse=True)
    res.notes.append(f"root_includes_preamble:{root}")
    res.header_file = root
    return top[0]


def _pick_class_root(stripped: dict[str, str], canon: dict[str, str], lower: dict[str, str],
                     res: GatherResult, macros: dict[str, list[str]] | None = None) -> str | None:
    cands = []
    for c, rel in canon.items():
        if not c.lower().endswith(".tex"):
            continue
        m = _DOCCLASS_RE.search(stripped[rel])
        if m:
            cands.append((c, m.group(1).lower()))
    if not cands:
        with_doc = [c for c, rel in canon.items() if _BEGIN_DOC_RE.search(stripped[rel])]
        pool = with_doc or list(canon)
        if not pool:
            return None
        res.status = "root_fallback"
        res.notes.append("root_fallback:" + ("begin_document" if with_doc else "largest"))
        return max(pool, key=lambda c: (len(stripped[canon[c]]), c))
    if len(cands) > 1:
        res.notes.append(f"root_candidates:{len(cands)}")
        non_aux = [x for x in cands if x[1] not in ("standalone", "subfiles")]
        if non_aux:
            cands = non_aux
    if len(cands) == 1:
        return cands[0][0]
    reach = {c: _reachable(c, stripped, canon, lower, macros) for c, _cls in cands}
    included = {c for c, _ in cands for o, _2 in cands if o != c and c in reach[o]}
    top = [x for x in cands if x[0] not in included] or cands

    def rank(x):
        c = x[0]
        t = stripped[canon[c]]
        stem = posixpath.splitext(posixpath.basename(c))[0].lower()
        return (bool(_search_top(_BEGIN_DOC_RE, t)), stem in C.ROOT_PREFERRED_NAMES,
                len(reach[c]), len(t), -len(c))
    top.sort(key=rank, reverse=True)
    return top[0][0]


# ---------------------------------------------------------------------------------------
def gather_files(files: dict[str, str], sizes: dict[str, int] | None = None,
                 safe_mode: bool = False) -> GatherResult:
    """Gather the compilation path of one version. Never raises on malformed input."""
    res = GatherResult()
    sizes = sizes or {}
    tex = {}
    for rel, raw in files.items():
        if not rel.lower().endswith(".tex"):
            continue
        if sizes.get(rel, len(raw)) > C.MAX_TEX_FILE_BYTES:
            res.excluded_files.append(rel)
            res.notes.append(f"big_file:{rel}")
            continue
        tex[rel] = raw
    if not tex:
        res.status = "empty"
        return res

    lits = res.literals
    stripped = {rel: clean_file_text(raw, lits) for rel, raw in tex.items()}
    canon = _canonicalize(tex, res, stripped)
    lower = {c.lower(): c for c in canon}
    res._file_text = {c: stripped[rel] for c, rel in canon.items()}

    path_macros = _path_macros(stripped.values())
    res.path_macros = path_macros
    root = _pick_root(stripped, canon, lower, res, path_macros)
    if root is None:
        res.status = "empty"
        return res
    res.root_file = canon[root] if canon[root] != root else root
    root_dir = posixpath.dirname(root)

    # context for conditionals: macros defined anywhere, switches declared anywhere
    defined: set[str] = set()
    switch_init: dict[str, bool] = {}
    names: set[str] = set()
    sw_memo: dict[str, str] = {}

    def file_switches(c: str, seen: frozenset) -> str:
        """Switch assignments (``\\Xtrue`` / ``\\Xfalse``) a file makes, including those of the
        files it includes, in source order."""
        if c in sw_memo:
            return sw_memo[c]
        t = stripped.get(canon.get(c, c), "")
        out: list[str] = []
        if "\\" in t and len(seen) < C.MAX_INCLUDE_DEPTH:
            ev = [(m.start(), 0, m) for m in _SETIF_RE.finditer(t) if m.group(1) in names]
            ev += [(m.start(), 1, m) for m in _INCLUDE_RE.finditer(t)]
            ev.sort(key=lambda x: (x[0], x[1]))
            for _p, kind, m in ev:
                if kind == 0:
                    out.append(m.group(0))
                    continue
                name, _k, _e = _parse_include(t, m)
                r = _resolve(name, posixpath.dirname(c), root_dir, canon, lower, macros=path_macros) if name else None
                if r and r != c and r not in seen:
                    out.append(file_switches(r, seen | {c}))
        sw_memo[c] = "".join(out)
        return sw_memo[c]

    def splice_switches(t: str, c: str) -> str:
        """Copy the switch assignments of included files right after their include command:
        conditionals are evaluated before includes are inlined, so ``\\input{params}`` setting
        ``\\eprinttrue`` must be visible to a later ``\\ifeprint`` of the including file."""
        if not names or "\\" not in t:
            return t
        out: list[str] = []
        pos = 0
        for m in _INCLUDE_RE.finditer(t):
            if m.start() < pos:
                continue
            name, _k, end = _parse_include(t, m)
            r = _resolve(name, posixpath.dirname(c), root_dir, canon, lower, macros=path_macros) if name else None
            if r is None or r == c:
                continue
            a = file_switches(r, frozenset({c}))
            if a:
                out.append(t[pos:end])
                out.append(_SW_OPEN + a + _SW_CLOSE)
                pos = end
        if not out:
            return t
        out.append(t[pos:])
        return "".join(out)

    if not safe_mode:
        for t in stripped.values():
            if "\\" in t:
                for m in _DEFINED_RE.finditer(t):
                    defined.add(m.group(1) or m.group(2) or m.group(3))
        rt = stripped[canon[root]]
        for t in stripped.values():
            names.update(_NEWIF_RE.findall(t))
        if names:
            pre_end = _search_top(_BEGIN_DOC_RE, rt)
            pre = splice_switches(rt[:pre_end.start()] if pre_end else rt, root)
            for m in _SETIF_RE.finditer(pre):
                if m.group(1) in names:
                    switch_init[m.group(1)] = m.group(2) == "true"

    includeonly: set[str] | None = None
    mi = _INCLUDEONLY_RE.search(stripped[canon[root]])
    if mi:
        includeonly = {posixpath.splitext(x.strip())[0] for x in mi.group(1).split(",") if x.strip()}
        res.notes.append("includeonly")

    pieces: list[str] = []
    seg_starts: list[int] = []
    seg_info: list[tuple[str, int]] = []
    pos_total = [0]
    compile_seen: list[str] = []
    missing: list[str] = []
    missing_at: list[tuple[int, str]] = []

    def emit(s: str, f: str, file_off: int):
        if not s:
            return
        seg_starts.append(pos_total[0])
        seg_info.append((f, file_off))
        pieces.append(s)
        pos_total[0] += len(s)

    def prepare(c: str, is_root: bool) -> str:
        t = stripped[canon[c]]
        if not is_root and c != res.header_file:
            body = _document_body(t)
            if body is not None:
                res.notes.append(f"nested_document:{c}")
                t = body
        if not safe_mode and "\\if" in t:
            t = splice_switches(t, c)
            init = switch_init if not is_root else None
            if init:
                # make preamble switch values visible to included files
                t2 = "".join(f"\\{k}{'true' if v else 'false'}" for k, v in init.items())
                head = "".join(f"\\newif\\if{k}" for k in init)
                t = evaluate_all(head + t2 + "\n" + t, defined, res.cond)
                t = t[t.find("\n") + 1:] if "\n" in t else t
            else:
                t = evaluate_all(t, defined, res.cond)
            if _SW_OPEN in t:
                t = _SW_SPAN_RE.sub("", t)
        return t

    def inline(c: str, ancestry: tuple[str, ...], depth: int):
        if canon[c] not in compile_seen:
            compile_seen.append(canon[c])
        t = prepare(c, depth == 0)
        cur_dir = posixpath.dirname(c)
        pos = 0
        for m in _INCLUDE_RE.finditer(t):
            if m.start() < pos:
                continue
            name, cmd, end = _parse_include(t, m)
            if name is None:
                continue
            if cmd == "include" and includeonly is not None:
                stem = posixpath.splitext(name.strip())[0]
                if stem not in includeonly:
                    emit(t[pos:m.start()], c, pos)
                    pos = end
                    continue
            base_dir = cur_dir
            if cmd in ("subimport", "subincludefrom", "subinputfrom", "import", "includefrom",
                       "inputfrom"):
                base_dir = cur_dir
            target = _resolve(name, base_dir, root_dir, canon, lower, macros=path_macros)
            emit(t[pos:m.start()], c, pos)
            pos = end
            if target is None:
                nm = name.strip()
                if nm not in missing:
                    missing.append(nm)
                    missing_at.append((pos_total[0], nm))
                continue
            if target in ancestry or target == c:
                res.notes.append(f"include_cycle:{target}")
                continue
            if depth + 1 > C.MAX_INCLUDE_DEPTH or pos_total[0] > MAX_COMBINED:
                res.notes.append(f"include_limit:{target}")
                continue
            if cmd == "include":
                emit("\n\n", c, pos)
            else:
                emit(" ", c, pos)
            inline(target, ancestry + (c,), depth + 1)
            emit("\n\n" if cmd == "include" else " ", c, pos)
        emit(t[pos:], c, pos)

    inline(root, (), 0)
    text = "".join(pieces)
    res.text = text
    res._seg_starts = seg_starts
    res._seg_info = seg_info
    res.compile_files = compile_seen
    res.missing_includes = missing
    res.signature_parts = [(f, content_hash(stripped[f])) for f in compile_seen]
    in_path = set(compile_seen)
    for c, rel in canon.items():
        if rel not in in_path and rel not in res.redundant_files:
            res.redundant_files.append(rel)
    res.redundant_files.sort()

    b = _search_top(_BEGIN_DOC_RE, text)
    if b:
        res.preamble = text[:b.start()]
        e = _search_top(_END_DOC_RE, text, b.end())
        res.body = text[b.end():e.start() if e else len(text)]
        res.body_offset = b.end()
        res.missing_body_includes = [nm for p, nm in missing_at if p >= b.start()
                                     and posixpath.splitext(nm)[1].lower() in ("", ".tex")]
        if not e:
            res.notes.append("no_end_document")
    else:
        res.preamble = ""
        res.body = text
        res.body_offset = 0
        res.notes.append("no_begin_document")

    # bibliography
    names: list[str] = []
    for m in _BIB_RE.finditer(text):
        for x in m.group(1).split(","):
            x = x.strip()
            if x and x not in names:
                names.append(x)
    res.bib_names = names
    res.nocite_all = bool(_NOCITE_ALL_RE.search(text))
    return res


def resolve_bib_files(res: GatherResult, bib_paths: list[str]) -> None:
    """Resolve :attr:`GatherResult.bib_names` against the ``.bib`` files of the version.

    If no name resolves, every ``.bib`` of the version is used (``bib_fallback``)."""
    canon = {p: p for p in bib_paths}
    lower = {p.lower(): p for p in bib_paths}
    root_dir = posixpath.dirname(res.root_file or "")
    out: list[str] = []
    for name in res.bib_names:
        r = _resolve(name, root_dir, root_dir, canon, lower, exts=(".bib",), macros=res.path_macros)
        if r and r not in out:
            out.append(r)
    if not out and bib_paths:
        # Prefer files without "(n)" copies
        best: dict[str, str] = {}
        for p in sorted(bib_paths):
            c, n = split_dup_suffix(posixpath.basename(p))
            best.setdefault(posixpath.join(posixpath.dirname(p), c), p)
        out = sorted(best.values())
        res.bib_fallback = True
        res.notes.append("bib_fallback_all")
    res.bib_files = out


def strip_for_signature(raw: str) -> str:
    """Comment-stripped text used for the identical-version fast path."""
    return strip_comments(normalize_newlines(raw))
