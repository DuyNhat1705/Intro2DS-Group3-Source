r"""Multi-file LaTeX gathering and cleanup for Milestone 2.

Stage 1, :func:`gather_tex`, rebuilds what LaTeX compiles for one paper
version. It picks the root file (the one with ``\documentclass``), follows
``\input`` / ``\include`` / ``\subfile`` / ``\import`` in compilation order,
skips commented-out commands and inactive ``\iffalse`` / ``\newif`` branches,
and lists the ``.bib`` files named by ``\bibliography``. Files that are not
reached are redundant and ignored.

Stage 2, :func:`cleanup_latex`, keeps the content and drops the presentation.
It expands the authors' own macros, cuts the preamble, rewrites every formula
into one of two forms (``$...$`` inline, ``\begin{equation}...\end{equation}``
for blocks), removes non-semantic formatting and canonicalises spelling
variants, so that the same sentence in two versions becomes the same string.
Sections, lists, figures, tables, theorem-like environments, ``\cite`` and the
bibliography are kept for the hierarchy and BibTeX stages.

Corpus helpers for the report: :func:`validate_cleaned`, :func:`gather_report`,
:func:`scan_corpus` and :func:`dedup_ablation`. Source map (T15):
``GatherTex.origins``, :func:`locate` and :func:`chars_by_file` tell which
file and line every piece of ``source`` comes from. Before/after viewer
(T17): :func:`clean_blocks`, :func:`before_after_html` and
:func:`show_before_after`. From the repository root::

    python src/gather_clean_tex.py data --out gather_report.csv --workers 2
    python src/gather_clean_tex.py data --out gather_report.csv --workers 2 --resume
"""

from __future__ import annotations

import argparse
import bisect
import csv
import difflib
import hashlib
import html
import itertools
import json
import os
import re
import sys
import time
import unicodedata
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable

from utils import GatherTex

__all__ = [
    "ABLATION_LEVELS",
    "AmbiguousRootError",
    "CLEANUP_STEPS",
    "GatherError",
    "GatherWarning",
    "NoRootError",
    "REPORT_FIELDS",
    "before_after_html",
    "chars_by_file",
    "clean_blocks",
    "cleanup_latex",
    "dedup_ablation",
    "gather_report",
    "gather_tex",
    "locate",
    "read_checkpoint",
    "scan_corpus",
    "show_before_after",
    "validate_cleaned",
]


# ============================================================== errors


class GatherError(Exception):
    """The compilation path of a version cannot be determined."""


class NoRootError(GatherError):
    r"""No ``.tex`` file of the version declares ``\documentclass``."""


class AmbiguousRootError(GatherError):
    """Several root candidates remain after every tie-breaking rule."""


class GatherWarning(UserWarning):
    """A recoverable gathering problem (missing include, include cycle, ...)."""


# ============================================================== T1: patterns

# Root declaration; group 1 is the class name (LaTeX 2.09 used \documentstyle).
_DOCUMENT_CLASS = re.compile(r"\\document(?:class|style)\s*(?:\[[^\]]*\])?\s*\{([^{}]*)\}")

# Static includes. \import-style commands take a directory and a file; \input
# also accepts the brace-less TeX form "\input file". The look-aheads keep
# \includegraphics, \inputencoding, ... out.
_INCLUDE = re.compile(
    r"\\(?:"
    r"(?P<cmd2>import|subimport|inputfrom|includefrom|subinputfrom|subincludefrom)\*?"
    r"\s*\{(?P<dir>[^{}]*)\}\s*\{(?P<file2>[^{}]*)\}"
    r"|(?P<cmd1>input|include|subfile)(?![A-Za-z@])"
    r"\s*(?:\{(?P<file1>[^{}]*)\}|(?P<bare>[^\s{}\\%]+))"
    r")"
)

# Bibliography databases: \bibliography{a,b} (BibTeX), \addbibresource (biblatex).
_BIBLIOGRAPHY = re.compile(
    r"\\(?:bibliography|addbibresource|addglobalbib|nobibliography)(?![A-Za-z@])\*?"
    r"\s*(?:\[[^\]]*\]\s*)?\{([^{}]*)\}"
)

_BEGIN_DOCUMENT = re.compile(r"\\begin\s*\{document\}")
_END_DOCUMENT = re.compile(r"\\end\s*\{document\}")

# Verbatim-like material is never touched: comments, macros and whitespace
# rules do not apply inside it.
_VERBATIM_NAMES = r"verbatim\*?|Verbatim\*?|BVerbatim|lstlisting|minted|alltt"
_VERBATIM_ENV = re.compile(r"\\begin\{(" + _VERBATIM_NAMES + r")\}.*?\\end\{\1\}", re.S)
_VERB = re.compile(r"\\verb\*?([^\sA-Za-z*]).*?\1")

# Arguments that must survive text rewriting untouched (keys, labels, paths).
_ARG_PROTECT = re.compile(
    r"\\(?:url|href|path|cite[A-Za-z]*|nocite|[A-Za-z]*ref|label|includegraphics|bibitem)"
    r"(?![A-Za-z@])\*?\s*(?:\[[^\]]*\]\s*)*\{[^{}]*\}"
)

_MAX_INCLUDE_DEPTH = 32
_MAX_MACRO_PASSES = 10

# Private-use characters: placeholders for protected spans and a marker for
# removed commands (a line holding only markers is deleted entirely).
_PH = re.compile("\ue000(\\d+)\ue001")
_MARK = "\ue002"
_KEEP_SPACE = "\ue003"


_PLACEHOLDER_IDS = itertools.count()


class _Shelf:
    """Swap spans for placeholders and put them back later.

    Ids are unique across shelves, so nested shelves (one per cleanup step)
    never restore each other's placeholders.
    """

    def __init__(self) -> None:
        self.items: dict[int, str] = {}

    def put(self, s: str) -> str:
        i = next(_PLACEHOLDER_IDS)
        self.items[i] = s
        return f"\ue000{i}\ue001"

    def hide(self, pattern: re.Pattern, text: str) -> str:
        return pattern.sub(lambda m: self.put(m.group(0)), text)

    def restore(self, text: str) -> str:
        for _ in range(8):  # placeholders may nest
            new = _PH.sub(lambda m: self.items.get(int(m.group(1)), m.group(0)), text)
            if new == text:
                break
            text = new
        return text


# ============================================================== brace helpers

_BRACE_TOKEN = re.compile(r"\\.|[{}]", re.S)
_CS = re.compile(r"\\(?:[A-Za-z@]+|.)", re.S)
_ARG_GAP = re.compile(r"[ \t]*\n?[ \t]*")  # spaces TeX skips between arguments
# Spaces TeX skips after a control word (one end of line, but never a blank line).
_AFTER_CS = re.compile(r"[ \t]*(?:\n(?![ \t]*\n)[ \t]*)?")
_SPACES = re.compile(r"[ \t]*")


def _match_brace(text: str, i: int) -> int:
    r"""Index of the ``}`` closing the ``{`` at ``text[i]``, or -1 (``\{`` ignored)."""
    depth = 0
    for m in _BRACE_TOKEN.finditer(text, i):
        tok = m.group(0)
        if tok == "{":
            depth += 1
        elif tok == "}":
            depth -= 1
            if depth == 0:
                return m.start()
    return -1


def _read_arg(text: str, pos: int) -> tuple[str, int] | None:
    """Read one mandatory TeX argument: a braced group or a single token."""
    pos = _ARG_GAP.match(text, pos).end()
    if pos >= len(text):
        return None
    ch = text[pos]
    if ch == "{":
        j = _match_brace(text, pos)
        return None if j < 0 else (text[pos + 1 : j], j + 1)
    if ch == "\\":
        m = _CS.match(text, pos)
        return (m.group(0), m.end()) if m else None
    if ch in "}%":
        return None
    return ch, pos + 1


def _read_optional(text: str, pos: int, open_: str = "[", close: str = "]") -> tuple[str | None, int]:
    """Read ``[...]`` (braces may nest inside); returns (None, pos) if absent."""
    p = _ARG_GAP.match(text, pos).end()
    if p >= len(text) or text[p] != open_:
        return None, pos
    depth = 0
    i = p + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == close and depth <= 0:
            return text[p + 1 : i], i + 1
        i += 1
    return None, pos


def _read_args(text: str, pos: int, spec: str) -> tuple[list[str | None] | None, int]:
    """Read arguments by spec: ``{`` mandatory, ``[`` / ``(`` optional, ``s`` skip spaces."""
    args: list[str | None] = []
    for ch in spec:
        if ch == "s":
            pos = _SPACES.match(text, pos).end()
        elif ch == "{":
            r = _read_arg(text, pos)
            if r is None:
                return None, pos
            args.append(r[0])
            pos = r[1]
        else:
            close = "]" if ch == "[" else ")"
            a, pos = _read_optional(text, pos, ch, close)
            args.append(a)
    return args, pos


def _braces_balanced(text: str) -> bool:
    depth = 0
    for m in _BRACE_TOKEN.finditer(text):
        tok = m.group(0)
        if tok == "{":
            depth += 1
        elif tok == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def _document_body(text: str) -> str:
    r"""The part between ``\begin{document}`` and ``\end{document}`` (all if absent)."""
    m = _BEGIN_DOCUMENT.search(text)
    if not m:
        return text
    body = text[m.end() :]
    e = _END_DOCUMENT.search(body)
    return body[: e.start()] if e else body


# ============================================================== T2: comments


# One left-to-right scan: whichever starts first wins, as in TeX (a "%" before
# "\verb" comments it out; a "%" inside verbatim or a URL is kept).
# "%" starts a comment unless escaped. An even run of backslashes before it
# ("\\%") is a line break followed by a real comment, so the run is kept
# (group "bs"). Like TeX, the comment also eats its end of line and the next
# line's indentation: "foo%\nbar" is "foobar", and a comment-only line leaves
# no blank line behind (a blank line would be a spurious paragraph break).
# When the next line is blank it stays: that empty line is a real \par.
_STRIP_SCAN = re.compile(
    r"(?P<keep>\\begin\{(?P<venv>" + _VERBATIM_NAMES + r")\}.*?\\end\{(?P=venv)\}"
    r"|\\verb\*?(?P<vd>[^\sA-Za-z*])[^\n]*?(?P=vd)"
    r"|\\(?:url|href|path)\s*\{[^{}\n]*\})"
    r"|\\begin\{comment\}.*?\\end\{comment\}\n?"
    r"|(?<!\\)(?P<bs>(?:\\\\)*)%[^\n]*(?:\n(?![ \t]*\n)[ \t]*)?",
    re.S,
)


def _strip_comments_mapped(source: str) -> tuple[str, list[tuple[int, int]]]:
    """Strip comments and return ``(text, anchors)`` for the source map (T15).

    ``anchors`` holds ``(offset in text, newlines removed so far)`` after
    every removal, so a position in the stripped text can be traced back to
    its line in the file.
    """
    out: list[str] = []
    anchors: list[tuple[int, int]] = []
    pos = length = removed = 0
    for m in _STRIP_SCAN.finditer(source):
        if m.group("keep") is not None:
            continue
        kept = m.group("bs") or ""
        chunk = source[pos : m.start()] + kept
        out.append(chunk)
        length += len(chunk)
        removed += source.count("\n", m.start(), m.end())
        anchors.append((length, removed))
        pos = m.end()
    out.append(source[pos:])
    return "".join(out), anchors


def _strip_comments(source: str) -> str:
    r"""Remove ``%`` comments and ``comment`` environments, which never reach the PDF.

    Verbatim material and URLs are protected, so ``\verb|%|`` and
    ``\url{a%20b}`` survive, and ``50\%`` is a percent sign, not a comment.
    """
    return _strip_comments_mapped(source)[0]


class _LineTable:
    """Map an offset of a comment-stripped file back to its 1-based line in the file."""

    def __init__(self, text: str, anchors: list[tuple[int, int]]) -> None:
        self.newlines = [m.start() for m in re.finditer("\n", text)]
        self.anchor_pos = [a[0] for a in anchors]
        self.anchor_removed = [a[1] for a in anchors]

    def line(self, offset: int) -> int:
        k = bisect.bisect_right(self.anchor_pos, offset) - 1
        return 1 + bisect.bisect_left(self.newlines, offset) + (self.anchor_removed[k] if k >= 0 else 0)


# ============================================================== T3: assets


def _index_files(version_dir: Path) -> dict[str, list[Path]]:
    """Map lower-cased basenames to the files of the version, for fallback lookups."""
    index: dict[str, list[Path]] = {}
    for p in version_dir.rglob("*"):
        if p.is_file():
            index.setdefault(p.name.lower(), []).append(p)
    return index


def _find_asset(
    name: str,
    extension: str,
    root_dir: Path,
    source_dir: Path,
    version_dir: Path,
    *,
    index: dict[str, list[Path]] | None = None,
) -> Path | None:
    r"""Resolve the file named in ``\input{name}`` / ``\bibliography{name}``.

    LaTeX resolves names against the directory it runs in, i.e. the root
    file's directory, so ``root_dir`` comes first; the including file's
    directory and the version directory follow as lenient fallbacks for
    sloppy paths. Like LaTeX, ``name + extension`` is tried before the bare
    name. Last resort: a unique file with the same basename anywhere in the
    version. Nothing outside ``version_dir`` is ever returned.
    """
    name = name.strip().strip('"').strip()
    if not name:
        return None
    candidates = [name] if name.lower().endswith(extension) else [name + extension, name]
    version_dir = version_dir.resolve()
    for base in dict.fromkeys((root_dir, source_dir, version_dir)):
        for cand in candidates:
            try:
                p = (base / cand).resolve()
            except (OSError, ValueError):
                continue
            if p.is_file() and p.is_relative_to(version_dir):
                return p
    if index is None:
        index = _index_files(version_dir)
    for cand in candidates:
        hits = index.get(Path(cand.replace("\\", "/")).name.lower(), [])
        if len(hits) == 1:
            return hits[0].resolve()
    return None


# ============================================================== T4: gathering

# Tokens the expander reacts to. "\\" pairs are matched first so that "\\fi"
# (a line break followed by the letters "fi") is never read as \fi.
_TOKEN = re.compile(
    r"\\(?:"
    r"(?P<esc>\\)"
    r"|newif\s*\\if(?P<newif>[A-Za-z@]+)"
    r"|(?P<if>if[A-Za-z@]*)"
    r"|(?P<else>else)(?![A-Za-z@])"
    r"|(?P<fi>fi)(?![A-Za-z@])"
    r"|(?P<endinput>endinput)(?![A-Za-z@])"
    r"|includeonly\s*\{(?P<only>[^{}]*)\}"
    r"|(?P<cmd2>import|subimport|inputfrom|includefrom|subinputfrom|subincludefrom)\*?"
    r"\s*\{(?P<dir>[^{}]*)\}\s*\{(?P<file2>[^{}]*)\}"
    r"|(?P<cmd1>input|include|subfile)(?![A-Za-z@])"
    r"\s*(?:\{(?P<file1>[^{}]*)\}|(?P<bare>[^\s{}\\%]+))"
    r"|(?P<flag>[A-Za-z@]+?)(?P<value>true|false)(?![A-Za-z@])"
    r")"
)

# Conditionals terminated by \fi whose truth value is unknown to us: they are
# kept verbatim but still counted, so that their \fi does not close ours.
_PRIMITIVE_IFS = frozenset(
    "if ifcat ifnum ifdim ifodd ifvmode ifhmode ifmmode ifinner ifvoid ifhbox "
    "ifvbox ifx ifeof iftrue iffalse ifcase ifdefined ifcsname iffontchar "
    "ifincsname ifpdf ifxetex ifluatex ifpdftex".split()
)
# Commands that merely start with "if" and take braced arguments instead of \fi.
_NOT_CONDITIONALS = frozenset(
    "iff ifthenelse ifnextchar ifstar ifempty ifblank ifdef ifndef ifundef "
    "ifcsdef ifcsundef ifbool iftoggle ifstrequal ifstrempty ifdefempty "
    "ifcsempty ifnumcomp ifdimcomp ifboolexpr ifinlist ifentrytype "
    "iffieldundef ifcitation ifnameundef ifpackageloaded ifclassloaded "
    "iflanguage ifcurrentfield".split()
)

# arXiv compiles with (pdf)LaTeX from TeX Live: these engine tests are decidable.
_ENGINE_IFS = {"ifxetex": False, "ifXeTeX": False, "ifluatex": False, "ifLuaTeX": False}
_KNOWN_DEFINED = frozenset(
    "pdfoutput pdftexversion pdfstrcmp pdfpagewidth pdfpageheight pdfminorversion pdfcompresslevel "
    "pdfobjcompresslevel pdfinfo pdfsuppresswarningpagegroup pdfgentounicode pdfglyphtounicode".split()
)
_KNOWN_UNDEFINED = frozenset("XeTeXversion XeTeXrevision directlua luatexversion".split())
_EMPTY_NAMES = frozenset({"empty", "@empty"})
_UNDEFINED_NAMES = frozenset({"undefined", "@undefined"})

_IF_DETOKENIZE = re.compile(r"\s*\\relax\s*\\detokenize\s*(\\expandafter\s*)?(?=\{)")
_RELAX_AFTER = re.compile(r"\s*\\relax(?![A-Za-z@])")
_IFX_PAIR = re.compile(r"\s*\\([A-Za-z@]+)\s*\\([A-Za-z@]+)(?![A-Za-z@])")
_IF_ONE_CS = re.compile(r"\s*\\([A-Za-z@]+)(?![A-Za-z@])")
_IFCSNAME = re.compile(r"\s*([A-Za-z@]+)\s*\\endcsname(?![A-Za-z@])")


def _find_definition(name: str, texts: Iterable[str]) -> str | None:
    """Body of the author's definition of ``\\name`` in ``texts``, or None when there is none."""
    n = re.escape(name)
    rx = re.compile(
        r"\\(?:(?:re)?newcommand|providecommand|DeclareRobustCommand)\*?\s*\{?\s*\\" + n
        + r"(?![A-Za-z@])\s*\}?\s*(?:\[\d\]\s*)?(?:\[[^\]]*\]\s*)?(?=\{)"
        r"|\\[gex]?def\s*\\" + n + r"(?![A-Za-z@])\s*(?=\{)"
        r"|\\let\s*\\" + n + r"(?![A-Za-z@])\s*=?\s*\\(?P<let>[A-Za-z@]+)"
    )
    for t in texts:
        m = rx.search(t)
        if m is None:
            continue
        if m.group("let") is not None:
            return "" if m.group("let") in _EMPTY_NAMES else "\\" + m.group("let")
        r = _read_arg(t, m.end())
        if r is not None:
            return r[0]
    return None


def _defined(name: str, texts: Iterable[str]) -> bool | None:
    if name in _KNOWN_DEFINED or _find_definition(name, texts) is not None:
        return True
    if name in _KNOWN_UNDEFINED:
        return False
    return None  # a package might define it: unknown


def _evaluate_if(name: str, text: str, pos: int, texts: list[str]) -> tuple[bool, int] | None:
    r"""Decide the common "is this macro empty / defined?" tests; returns (value, end) or None.

    ``\if\relax\detokenize{..}\relax`` (emptiness test), ``\ifx\M\empty``,
    ``\ifx\M\undefined``, ``\ifdefined\M``, ``\ifcsname M\endcsname`` and the
    engine tests ``\ifxetex`` / ``\ifluatex``. A test on a macro argument
    (``#1``) is never decided: it runs at each use, not here.
    """
    if name in _ENGINE_IFS:
        return _ENGINE_IFS[name], pos
    if name == "if":
        m = _IF_DETOKENIZE.match(text, pos)
        if m is None:
            return None
        r = _read_arg(text, m.end())
        if r is None:
            return None
        arg, end = r
        tail = _RELAX_AFTER.match(text, end)
        if tail is None or "#" in arg:
            return None
        arg = arg.strip()
        if m.group(1) and re.fullmatch(r"\\[A-Za-z@]+", arg):  # \detokenize\expandafter{\M}
            body = _find_definition(arg[1:], texts)
            if body is None:
                return None
            return body.strip() == "", tail.end()
        return arg == "", tail.end()
    if name == "ifx":
        m = _IFX_PAIR.match(text, pos)
        if m is None:
            return None
        a, b = m.group(1), m.group(2)
        if a in _EMPTY_NAMES | _UNDEFINED_NAMES:
            a, b = b, a
        if b in _EMPTY_NAMES:
            body = _find_definition(a, texts)
            if body is None:
                return (False, m.end()) if a in _KNOWN_UNDEFINED | _KNOWN_DEFINED else None
            return body == "", m.end()
        if b in _UNDEFINED_NAMES:
            d = _defined(a, texts)
            return None if d is None else (not d, m.end())
        return None
    if name == "ifdefined":
        m = _IF_ONE_CS.match(text, pos)
        d = _defined(m.group(1), texts) if m else None
        return None if d is None else (d, m.end())
    if name == "ifcsname":
        m = _IFCSNAME.match(text, pos)
        d = _defined(m.group(1), texts) if m else None
        return None if d is None else (d, m.end())
    return None


_ROOT_NAMES = frozenset(
    "main ms paper article manuscript arxiv root preprint submission document "
    "thesis report draft".split()
)
_NON_ROOT_CLASSES = frozenset({"standalone", "subfiles"})


def _read_tex(path: Path, encoding: str, notes: list[str]) -> str:
    data = path.read_bytes()
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        text = data.decode("latin-1")
        notes.append(f"encoding fallback (latin-1): {path.name}")
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _Expander:
    r"""Walk the sources like TeX does: one token stream across included files.

    With a ``version_dir`` it expands includes; without one it only evaluates
    conditionals (used by :func:`cleanup_latex` on plain strings).
    """

    def __init__(
        self,
        version_dir: Path | None = None,
        texts: dict[Path, str] | None = None,
        index: dict[str, list[Path]] | None = None,
        encoding: str = "utf-8-sig",
        notes: list[str] | None = None,
        anchors: dict[Path, list[tuple[int, int]]] | None = None,
    ) -> None:
        self.version_dir = version_dir
        self.texts = texts if texts is not None else {}
        self.index = index
        self.encoding = encoding
        self.notes = notes if notes is not None else []
        self.anchors = anchors if anchors is not None else {}
        self.tables: dict[Path, _LineTable] = {}
        self.reset(True)

    def reset(self, conditionals: bool) -> None:
        self.conditionals = conditionals
        # T15: (offset in the output, length, file, offset in the stripped file)
        self.spans: list[tuple[int, int, Path, int]] = []
        self.tex_files: list[Path] = []
        self.missing: list[str] = []
        self.newifs: dict[str, bool] = {}
        self.stack: list[list] = []  # [kind ("known"/"unknown"), taking: bool]
        self.includeonly: set[str] | None = None
        self.used_conditionals = False
        self.undecided = 0  # conditionals kept verbatim because their test is unknown
        self.root_dir: Path | None = None
        # Everything emitted so far: a test sees only the definitions TeX has read before it.
        self.emitted: list[str] = []

    def snapshot(self) -> tuple:
        return (self.tex_files, self.missing, list(self.notes), self.spans, self.undecided)

    def restore(self, state: tuple) -> None:
        self.tex_files, self.missing, self.notes[:], self.spans, self.undecided = state

    # ------------------------------------------------------------ helpers
    def emitting(self) -> bool:
        return all(f[1] for f in self.stack if f[0] == "known")

    def open_known(self) -> bool:
        return any(f[0] == "known" for f in self.stack)

    def _text(self, path: Path) -> str:
        if path not in self.texts:
            self.texts[path], self.anchors[path] = _strip_comments_mapped(_read_tex(path, self.encoding, self.notes))
        return self.texts[path]

    def line_of(self, path: Path, offset: int) -> int:
        """1-based line in the original file of an offset in its stripped text."""
        table = self.tables.get(path)
        if table is None:
            table = self.tables[path] = _LineTable(self.texts.get(path, ""), self.anchors.get(path, []))
        return table.line(offset)

    def where(self, path: Path | None, offset: int) -> str:
        if path is None or self.version_dir is None:
            return ""
        return f" ({path.relative_to(self.version_dir).as_posix()}:{self.line_of(path, offset)})"

    def _classify_if(self, name: str, text: str, pos: int):
        """``None``: not a conditional; ``("known", value[, end])``; ``("unknown", True)``."""
        if name == "iftrue":
            return ("known", True)
        if name == "iffalse":
            return ("known", False)
        if name[2:] in self.newifs:
            return ("known", self.newifs[name[2:]])
        if name in _NOT_CONDITIONALS:
            return None
        # "\ifx\foo\undefined \newcommand\foo..\fi": the definition after the test does not count.
        decided = _evaluate_if(name, text, pos, ["".join(self.emitted)])
        if decided is not None:
            return ("known", decided[0], decided[1])
        if name not in _PRIMITIVE_IFS:
            nxt = _ARG_GAP.match(text, pos).end()
            if nxt < len(text) and text[nxt] == "{":
                return None  # an \ifsomething{..}{..} command, not a conditional
        return ("unknown", True)

    # ------------------------------------------------------------ walking
    def walk(self, text: str, *, path: Path | None = None, base_dir: Path | None = None,
             chain: tuple[Path, ...] = (), depth: int = 0, offset: int = 0, file_off: int = 0) -> str:
        """Emit the active text. ``offset`` is where this text starts in the final
        output and ``file_off`` where it starts in its stripped file (source map)."""
        out: list[str] = []
        out_len = 0
        pos = 0
        files = self.version_dir is not None and path is not None

        def put(s: str, src: int | None = None) -> None:
            nonlocal out_len
            if not s:
                return
            if src is not None:  # text of this file (an include's content was recorded by its own walk)
                self.emitted.append(s)
                if files:
                    self.spans.append((offset + out_len, len(s), path, file_off + src))
            out.append(s)
            out_len += len(s)

        while True:
            m = _TOKEN.search(text, pos)
            if m is None:
                if self.emitting():
                    put(text[pos:], pos)
                break
            if self.emitting():
                put(text[pos : m.start()], pos)
            pos = m.end()
            tok = m.group(0)
            g = m.group

            def emit(s: str = tok, at: int = m.start()) -> None:
                if self.emitting():
                    put(s, at)

            if g("esc") is not None:
                emit()
            elif g("if") is not None:
                kind = self._classify_if(g("if"), text, pos) if self.conditionals else None
                if kind is None:
                    emit()
                elif kind[0] == "unknown":
                    emit()
                    self.stack.append(["unknown", True])
                    if self.emitting():
                        self.undecided += 1
                else:
                    self.stack.append(["known", kind[1]])
                    self.used_conditionals = True
                    pos = _AFTER_CS.match(text, kind[2] if len(kind) > 2 else pos).end()
            elif g("else") is not None:
                if not self.conditionals or not self.stack:
                    emit()
                elif self.stack[-1][0] == "known":
                    self.stack[-1][1] = not self.stack[-1][1]
                    pos = _AFTER_CS.match(text, pos).end()
                else:
                    emit()
            elif g("fi") is not None:
                if not self.conditionals or not self.stack:
                    emit()
                elif self.stack.pop()[0] == "unknown":
                    emit()
                else:
                    pos = _AFTER_CS.match(text, pos).end()
            elif g("newif") is not None:
                if not self.conditionals:
                    emit()
                else:
                    if self.emitting():
                        self.newifs[g("newif")] = False
                    pos = _AFTER_CS.match(text, pos).end()
            elif g("flag") is not None:
                if self.conditionals and g("flag") in self.newifs:
                    if self.emitting():
                        self.newifs[g("flag")] = g("value") == "true"
                    pos = _AFTER_CS.match(text, pos).end()
                else:
                    emit()
            elif g("endinput") is not None:
                if files and self.emitting():
                    break
                emit()
            elif g("only") is not None:
                if files:
                    if self.emitting():
                        self.includeonly = {_strip_tex_ext(s) for s in g("only").split(",") if s.strip()}
                else:
                    emit()
            else:  # an include command
                if not files:
                    emit()
                elif self.emitting():
                    put(self._include(m, path, base_dir, chain, depth, offset + out_len, file_off))
        return "".join(out)

    def _include(self, m: re.Match, path: Path, base_dir: Path, chain: tuple[Path, ...], depth: int,
                 offset: int = 0, file_off: int = 0) -> str:
        cmd = m.group("cmd1") or m.group("cmd2")
        if m.group("cmd1"):
            name = (m.group("file1") if m.group("file1") is not None else m.group("bare")).strip()
            new_base = base_dir
        else:
            d = m.group("dir").strip()
            name = m.group("file2").strip()
            # \import{dir} is relative to the run directory, \subimport to the current file.
            new_base = (path.parent / d) if cmd.startswith("sub") else (self.root_dir / d)
        if not name:
            return ""
        if "#" in name:
            return m.group(0)  # "\input{#1.sum}" inside a macro definition: a parameter, not a file
        if "\\" in name:
            self.notes.append(f"include named by a macro not followed: {name}{self.where(path, file_off + m.start())}")
            return m.group(0)
        if cmd in ("include", "includefrom", "subincludefrom") and self.includeonly is not None:
            if _strip_tex_ext(name) not in self.includeonly:
                self.notes.append(f"skipped by \\includeonly: {name}")
                return ""
        target = _find_asset(name, ".tex", new_base, path.parent, self.version_dir, index=self.index)
        at = self.where(path, file_off + m.start())
        if target is None:
            self.missing.append(name)
            self.notes.append(f"include not found: {name}{at}")
            warnings.warn(f"{self.version_dir}: include not found: {name}{at}", GatherWarning, stacklevel=2)
            return ""
        if target in chain:
            self.notes.append(f"include cycle skipped: {name}{at}")
            warnings.warn(f"{self.version_dir}: include cycle: {name}{at}", GatherWarning, stacklevel=2)
            return ""
        if depth >= _MAX_INCLUDE_DEPTH:
            self.notes.append(f"include depth limit reached: {name}{at}")
            return ""
        if target not in self.tex_files:
            self.tex_files.append(target)
        text = self._text(target)
        body_off = 0
        if cmd == "subfile" or _DOCUMENT_CLASS.search(text):
            body = _document_body(text)  # a standalone/subfile document: drop its preamble
            body_off = max(text.find(body), 0) if body else 0
            text = body
        prefix = 2 if "include" in cmd else 0
        content = self.walk(text, path=target, base_dir=new_base, chain=chain + (target,), depth=depth + 1,
                            offset=offset + prefix, file_off=body_off)
        if "include" in cmd:
            return "\n\n" + content + "\n\n"  # \include starts and ends with \clearpage
        # TeX turns the file's final end-of-line into a space; drop one newline.
        return content[:-1] if content.endswith("\n") else content

    def run(self, root: Path) -> str:
        source = self._run_once(root, conditionals=True)
        if not self.used_conditionals:
            return source
        if self.open_known() or not _braces_balanced(source):
            unclosed = self.open_known()
            kept = self.snapshot()
            alt = self._run_once(root, conditionals=False)
            if unclosed or _braces_balanced(alt):
                self.notes.append("conditional evaluation disabled: it unbalanced the source")
                return alt
            self.restore(kept)
        return source

    def _run_once(self, root: Path, *, conditionals: bool) -> str:
        self.reset(conditionals)
        self.root_dir = root.parent
        self.tex_files = [root]
        return self.walk(self._text(root), path=root, base_dir=root.parent, chain=(root,), depth=0)


def _strip_tex_ext(name: str) -> str:
    name = name.strip()
    return name[:-4] if name.lower().endswith(".tex") else name


def _resolve_conditionals(text: str) -> str:
    r"""Evaluate ``\iffalse``/``\newif`` branches of a plain string (no includes)."""
    ex = _Expander()
    out = ex.walk(text)
    if ex.used_conditionals and (ex.open_known() or (_braces_balanced(text) and not _braces_balanced(out))):
        return text
    return out


def _include_targets(text: str, path: Path, version_dir: Path, index) -> set[Path]:
    targets = set()
    for m in _INCLUDE.finditer(text):
        if m.group("cmd1"):
            name = m.group("file1") if m.group("file1") is not None else m.group("bare")
        else:
            name = f"{m.group('dir').strip().rstrip('/')}/{m.group('file2').strip()}"
        t = _find_asset(name, ".tex", path.parent, path.parent, version_dir, index=index)
        if t is not None:
            targets.add(t)
    return targets


def _pick_root(texts: dict[Path, str], version_dir: Path, index) -> tuple[Path, str]:
    r"""Choose the root file and say why.

    Candidates declare ``\documentclass`` outside comments and are not
    ``standalone`` / ``subfiles`` fragments. Ties are broken, in order, by
    having ``\begin{document}``, not being included by another file, a
    conventional name (main, ms, paper, ... or the arXiv id), and size.
    """
    cands = []
    for p, t in texts.items():
        m = _DOCUMENT_CLASS.search(t)
        if m and m.group(1).strip() not in _NON_ROOT_CLASSES:
            cands.append(p)
    if not cands:
        raise NoRootError(f"{version_dir}: no .tex file declares \\documentclass")
    if len(cands) == 1:
        return cands[0], "single \\documentclass"

    pool, reasons = list(cands), []

    def narrow(keep: list[Path], why: str) -> None:
        nonlocal pool
        if keep and len(keep) < len(pool):
            pool = keep
            reasons.append(why)

    narrow([p for p in pool if _BEGIN_DOCUMENT.search(texts[p])], "has \\begin{document}")
    if len(pool) > 1:
        included: set[Path] = set()
        for p, t in texts.items():
            included |= _include_targets(t, p, version_dir, index) - {p}
        narrow([p for p in pool if p not in included], "not included by another file")
    if len(pool) > 1:
        paper_id = version_dir.parent.name.lower()
        narrow([p for p in pool if p.stem.lower() in _ROOT_NAMES or p.stem.lower() == paper_id],
               "conventional file name")
    if len(pool) > 1:
        biggest = max(len(texts[p]) for p in pool)
        narrow([p for p in pool if len(texts[p]) == biggest], "largest file")
    if len(pool) > 1:
        names = ", ".join(sorted(p.name for p in pool))
        raise AmbiguousRootError(f"{version_dir}: cannot choose a root among {names}")
    return pool[0], f"{len(cands)} candidates; " + ", ".join(reasons)


def gather_tex(version_dir: Path, *, encoding: str = "utf-8-sig") -> GatherTex | None:
    r"""Expand static inputs in compilation order and list referenced .bib files.

    Returns ``None`` for an empty version: the directory exists but holds no
    ``.tex`` file (Milestone 1 keeps such versions as empty records).

    Raises :class:`FileNotFoundError` when the directory does not exist,
    :class:`NoRootError` when no file declares ``\documentclass`` and
    :class:`AmbiguousRootError` when the root cannot be chosen.

    An include whose file is absent (a TikZ figure, a ``.bbl`` dropped by the
    crawler, ...) does not fail the version: it is skipped, recorded in
    ``GatherTex.missing`` and reported with a :class:`GatherWarning`.

    The returned ``source`` has comments and inactive ``\iffalse`` /
    ``\newif`` branches removed and every reachable file spliced in place;
    nothing else is changed (``\bibitem`` entries are intact).
    """
    version_dir = Path(version_dir)
    if not version_dir.is_dir():
        raise FileNotFoundError(f"version directory not found: {version_dir}")
    version_dir = version_dir.resolve()
    tex_paths = sorted(p for p in version_dir.rglob("*") if p.is_file() and p.suffix.lower() == ".tex")
    if not tex_paths:
        return None

    notes: list[str] = []
    index = _index_files(version_dir)
    texts: dict[Path, str] = {}
    anchors: dict[Path, list[tuple[int, int]]] = {}
    for p in tex_paths:
        texts[p], anchors[p] = _strip_comments_mapped(_read_tex(p, encoding, notes))
    root, reason = _pick_root(texts, version_dir, index)

    expander = _Expander(version_dir, texts, index, encoding, notes, anchors)
    source = expander.run(root)
    missing = list(expander.missing)
    origins = _line_origins(source, expander, root)
    if expander.undecided:
        notes.append(f"{expander.undecided} conditional(s) with an undecidable test kept verbatim")

    bib_files: list[Path] = []
    for m in _BIBLIOGRAPHY.finditer(source):
        for name in m.group(1).split(","):
            name = name.strip()
            if not name:
                continue
            p = _find_asset(name, ".bib", root.parent, root.parent, version_dir, index=index)
            if p is None:
                missing.append(name if name.lower().endswith(".bib") else name + ".bib")
                f, ln = origins[min(source.count("\n", 0, m.start()), len(origins) - 1)]
                notes.append(f"bibliography not found: {name} ({f}:{ln})")
                warnings.warn(f"{version_dir}: bibliography not found: {name} ({f}:{ln})", GatherWarning, stacklevel=2)
            elif p not in bib_files:
                bib_files.append(p)

    return GatherTex(
        root=root,
        source=source,
        tex_files=list(expander.tex_files),
        bib_files=bib_files,
        missing=missing,
        root_reason=reason,
        notes=list(dict.fromkeys(notes)),
        origins=origins,
    )


def _line_origins(source: str, expander: _Expander, root: Path) -> list[tuple[str, int]]:
    """T15: for every line of ``source``, the file and line it was copied from."""
    spans = expander.spans
    starts = [s[0] for s in spans]
    rel: dict[Path, str] = {}
    origins: list[tuple[str, int]] = []
    line_start = 0
    while True:
        nl = source.find("\n", line_start)
        end = len(source) if nl < 0 else nl
        probe = line_start
        while probe < end and source[probe] in " \t":
            probe += 1  # indentation may belong to a line whose comment ate the newline
        if probe == end:
            probe = line_start
        k = bisect.bisect_right(starts, probe) - 1
        if k < 0:
            path, off = root, 0
        else:
            begin, length, path, file_off = spans[k]
            off = file_off + min(probe - begin, max(length - 1, 0))
        if path not in rel:
            rel[path] = path.relative_to(expander.version_dir).as_posix()
        origins.append((rel[path], expander.line_of(path, off)))
        if nl < 0:
            return origins
        line_start = nl + 1


def locate(tex: GatherTex, offset: int) -> tuple[str, int] | None:
    """T15: ``(file, line)`` that produced ``tex.source[offset]``, e.g. ``("sec/intro.tex", 42)``.

    Precise to the line: a line of ``source`` that mixes two files (an
    ``\\input`` in mid-line) is attributed to the file it starts in.
    """
    if not tex.origins:
        return None
    i = tex.source.count("\n", 0, max(0, min(offset, len(tex.source))))
    return tex.origins[min(i, len(tex.origins) - 1)]


def chars_by_file(tex: GatherTex) -> dict[str, int]:
    """T15: how many characters of ``tex.source`` each file contributed (root first)."""
    counts: dict[str, int] = {}
    for line, (name, _) in zip(tex.source.split("\n"), tex.origins):
        counts[name] = counts.get(name, 0) + len(line) + 1
    return counts


# ============================================================== T10: macros

_DEF_START = re.compile(
    r"\\\\|\\(newcommand|renewcommand|providecommand|DeclareRobustCommand|DeclareMathOperator"
    r"|def|gdef|edef|xdef|let|newenvironment|renewenvironment"
    r"|NewDocumentCommand|RenewDocumentCommand|ProvideDocumentCommand|DeclareDocumentCommand"
    r"|NewCommandCopy|RenewCommandCopy|DeclareCommandCopy)(?![A-Za-z@])(\*?)"
)
_XPARSE_DEFS = frozenset({"NewDocumentCommand", "RenewDocumentCommand", "ProvideDocumentCommand",
                          "DeclareDocumentCommand"})
_COMMAND_COPY = frozenset({"NewCommandCopy", "RenewCommandCopy", "DeclareCommandCopy"})
# xparse argument specs we can expand: an optional "O{default}" first, then "m"s.
_XPARSE_SPEC = re.compile(r"(?:O\{([^{}]*)\})?(m*)")
# Never redefined by expansion: the hierarchy and BibTeX stages rely on them.
_PROTECTED_MACROS = frozenset(
    "begin end section subsection subsubsection paragraph subparagraph chapter part "
    "item label ref eqref autoref cref Cref caption bibitem appendix title input include "
    "documentclass usepackage bibliography footnote url href includegraphics maketitle "
    "newcommand renewcommand def let abstract".split()
)
_PROTECTED_ENVS = frozenset(
    "document itemize enumerate description figure figure* table table* tabular "
    "equation equation* align align* thebibliography abstract proof theorem lemma "
    "corollary proposition definition remark example".split()
)
# Bodies relying on TeX internals cannot be expanded by substitution.
_UNSAFE_BODY = re.compile(
    r"\\[A-Za-z]*@|\\(?:if[A-Za-z]*|else|fi|expandafter|csname|futurelet|afterassignment"
    r"|noexpand|edef|xdef|def|let)(?![A-Za-z])"
)
_PARAMS = re.compile(r"(?:#[1-9])*")
_LET_GAP = re.compile(r"\s*=?\s?")


def _unsafe_body(body: str) -> bool:
    r"""True if a body needs TeX itself; ``\ifmmode..\else..\fi`` is fine (the math step decides it)."""
    if "\\ifmmode" in body:
        body = _choose_ifmmode(body, True) + _choose_ifmmode(body, False)
    return _UNSAFE_BODY.search(body) is not None


def _parse_cs_name(text: str, pos: int) -> tuple[str, int] | None:
    r"""Read ``\name`` or ``{\name}`` (the defined command) after a definition command."""
    p = _ARG_GAP.match(text, pos).end()
    if p < len(text) and text[p] == "{":
        j = _match_brace(text, p)
        if j < 0:
            return None
        name = text[p + 1 : j].strip()
        p = j + 1
    else:
        m = _CS.match(text, p)
        if not m:
            return None
        name, p = m.group(0), m.end()
    if not name.startswith("\\") or len(name) < 2:
        return None
    return name[1:], p


def _parse_definition(text: str, pos: int, kind: str, star: str):
    """Parse one definition; returns (end, item) or None when it is not parsable."""
    if kind in ("newenvironment", "renewenvironment"):
        r = _read_arg(text, pos)
        if r is None:
            return None
        env, p = r[0].strip(), r[1]
        nargs, default = 0, None
        opt, p2 = _read_optional(text, p)
        if opt is not None:
            if not opt.strip().isdigit():
                return None
            nargs, p = int(opt.strip()), p2
            opt2, p3 = _read_optional(text, p)
            if opt2 is not None:
                default, p = opt2, p3
        b = _read_arg(text, p)
        if b is None:
            return None
        e = _read_arg(text, b[1])
        if e is None:
            return None
        return e[1], ("env", env, nargs, default, b[0], e[0])

    if kind == "let":
        r = _parse_cs_name(text, pos)
        if r is None:
            return None
        name, p = r
        p = _LET_GAP.match(text, p).end()
        m = _CS.match(text, p)
        if m:
            return m.end(), ("let", name, m.group(0))
        return (p + 1, None) if p < len(text) else None

    if kind in _COMMAND_COPY:  # \NewCommandCopy{\a}{\b} is \let\a\b
        r = _parse_cs_name(text, pos)
        t = _parse_cs_name(text, r[1]) if r else None
        if t is None:
            return None
        return t[1], ("let", r[0], "\\" + t[0])

    if kind in _XPARSE_DEFS:  # \NewDocumentCommand{\name}{O{x} m m}{body}
        r = _parse_cs_name(text, pos)
        spec = _read_arg(text, r[1]) if r else None
        body = _read_arg(text, spec[1]) if spec else None
        if body is None:
            return None
        m = _XPARSE_SPEC.fullmatch(re.sub(r"\s+", "", spec[0]))
        if m is None:
            return body[1], None  # star or delimited arguments: drop the definition, keep usages
        nargs = len(m.group(2)) + (m.group(1) is not None)
        k = "providecommand" if kind == "ProvideDocumentCommand" else "newcommand"
        return body[1], ("macro", k, r[0], nargs, m.group(1), body[0])

    r = _parse_cs_name(text, pos)
    if r is None:
        return None
    name, p = r

    if kind in ("def", "gdef", "edef", "xdef"):
        k = text.find("{", p)
        if k < 0:
            return None
        params = re.sub(r"\s+", "", text[p:k])
        j = _match_brace(text, k)
        if j < 0:
            return None
        if not _PARAMS.fullmatch(params) or params != "".join(f"#{i}" for i in range(1, len(params) // 2 + 1)):
            return j + 1, None  # delimited parameters: drop the definition, keep usages
        return j + 1, ("macro", kind, name, len(params) // 2, None, text[k + 1 : j])

    if kind == "DeclareMathOperator":
        b = _read_arg(text, p)
        if b is None:
            return None
        return b[1], ("macro", kind, name, 0, None, f"\\operatorname{star}{{{b[0]}}}")

    # \newcommand-like: {\name}[n][default]{body}
    nargs, default = 0, None
    opt, p2 = _read_optional(text, p)
    if opt is not None:
        if not opt.strip().isdigit():
            return None
        nargs, p = int(opt.strip()), p2
        opt2, p3 = _read_optional(text, p)
        if opt2 is not None:
            default, p = opt2, p3
    b = _read_arg(text, p)
    if b is None:
        return None
    return b[1], ("macro", kind, name, nargs, default, b[0])


def _collect_definitions(text: str):
    """Remove macro/environment definitions from the text and return them."""
    macros: dict[str, tuple[int, str | None, str]] = {}
    envs: dict[str, tuple[int, str | None, str, str]] = {}
    out: list[str] = []
    pos = 0
    while True:
        m = _DEF_START.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        if m.group(1) is None:  # "\\"
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        parsed = _parse_definition(text, m.end(), m.group(1), m.group(2))
        if parsed is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        end, item = parsed
        out.append(text[pos : m.start()])
        out.append(_MARK)
        pos = end
        if item is None:
            continue
        if item[0] == "env":
            _, env, nargs, default, begin, finish = item
            if env not in _PROTECTED_ENVS and not _unsafe_body(begin + finish):
                envs[env] = (nargs, default, begin, finish)
        elif item[0] == "let":
            _, name, target = item
            if name in _PROTECTED_MACROS or name.startswith("cite"):
                continue
            if target[1:] in macros:
                macros[name] = macros[target[1:]]
            elif target[1:] != name:
                macros[name] = (0, None, target)
        else:
            _, kind, name, nargs, default, body = item
            if name in _PROTECTED_MACROS or name.startswith("cite"):
                continue
            if kind == "providecommand" and name in macros:
                continue
            if _unsafe_body(body) or re.search(r"\\" + re.escape(name) + r"(?![A-Za-z@])", body):
                macros.pop(name, None)  # internal or self-referential: leave usages alone
                continue
            macros[name] = (nargs, default, body)
    return "".join(out), macros, envs


def _substitute(body: str, args: list[str]) -> str:
    def rep(m: re.Match) -> str:
        if m.group(0) == "##":
            return "#"
        i = int(m.group(1)) - 1
        return args[i] if i < len(args) else ""

    return re.sub(r"##|#([1-9])", rep, body)


def _call_args(text: str, pos: int, nargs: int, default: str | None) -> tuple[list[str], int] | None:
    args: list[str] = []
    need = nargs
    if default is not None and nargs > 0:
        opt, p2 = _read_optional(text, pos)
        args.append(default if opt is None else opt)
        pos = p2 if opt is not None else pos
        need -= 1
    for _ in range(need):
        r = _read_arg(text, pos)
        if r is None:
            return None
        args.append(r[0])
        pos = r[1]
    return args, pos


def _expand_once(text: str, rx: re.Pattern, macros, envs) -> tuple[str, bool]:
    out: list[str] = []
    pos = 0
    changed = False
    while True:
        m = rx.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        name = m.group("m") or m.group("s")
        env = m.group("env")
        if name is None and env is None:  # "\\"
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        if name is not None:
            nargs, default, body = macros[name]
            got = _call_args(text, m.end(), nargs, default)
            if got is None:
                out.append(text[pos : m.end()])
                pos = m.end()
                continue
            args, end = got
            rep = _substitute(body, args)
            if end < len(text) and text[end].isalpha() and _TRAILING_CS.search(rep):
                rep += " "  # keep "\foo{a}b" from fusing into one control word
        else:
            nargs, default, begin, finish = envs[env]
            if m.group("be") == "begin":
                got = _call_args(text, m.end(), nargs, default)
                if got is None:
                    out.append(text[pos : m.end()])
                    pos = m.end()
                    continue
                args, end = got
                rep = _substitute(begin, args)
            else:
                end, rep = m.end(), finish
        out.append(text[pos : m.start()])
        out.append(rep)
        pos = end
        changed = True
    return "".join(out), changed


def _macros_step(text: str) -> str:
    r"""Evaluate conditionals, drop definitions and expand the authors' macros.

    Handles ``\newcommand``/``\renewcommand``/``\providecommand`` (with
    ``[n]`` arguments and an optional ``[default]``), ``\def`` with simple
    ``#1#2`` parameters, ``\let``, ``\DeclareMathOperator`` and
    ``\newenvironment``. Without this, ``\def\be{\begin{equation}}`` would
    hide formulas from math normalization. Macros using TeX internals are
    left unexpanded rather than guessed.
    """
    text = _resolve_conditionals(text)
    text, macros, envs = _collect_definitions(text)
    if not macros and not envs:
        return text
    letters = sorted((n for n in macros if re.fullmatch(r"[A-Za-z@]+", n)), key=len, reverse=True)
    symbols = [n for n in macros if len(n) == 1 and not n.isalpha()]
    # Every named group must exist; "(?!)" never matches.
    parts = [
        r"\\\\",
        r"\\(?P<m>" + "|".join(map(re.escape, letters)) + r")(?![A-Za-z@])" if letters else r"(?P<m>(?!))",
        r"\\(?P<s>" + "|".join(map(re.escape, symbols)) + r")" if symbols else r"(?P<s>(?!))",
        (r"\\(?P<be>begin|end)\{(?P<env>" + "|".join(map(re.escape, sorted(envs, key=len, reverse=True)))
         + r")\}") if envs else r"(?P<be>(?!))(?P<env>(?!))",
    ]
    rx = re.compile("|".join(parts))
    limit = 10 * len(text) + 100_000
    for _ in range(_MAX_MACRO_PASSES):
        text, changed = _expand_once(text, rx, macros, envs)
        if not changed or len(text) > limit:
            break
    # An environment's begin code may itself redefine something (\renewcommand{\qedsymbol}..).
    return _collect_definitions(text)[0]


# ============================================================== T5: body

_TITLE = re.compile(r"\\title(?![A-Za-z@])")
_ABSTRACT_ENV = re.compile(r"\\begin\{abstract\}.*?\\end\{abstract\}", re.S)
_ABSTRACT_CMD = re.compile(r"\\abstract(?![A-Za-z@])\s*(?=\{)")


def _preamble_abstract(preamble: str) -> str | None:
    """``acmart`` and several journal classes declare the abstract before ``\\begin{document}``."""
    m = _ABSTRACT_ENV.search(preamble)
    if m:
        return m.group(0)
    m = _ABSTRACT_CMD.search(preamble)
    if m:
        r = _read_arg(preamble, m.end())
        if r is not None and r[0].strip():
            return "\\begin{abstract}\n" + r[0].strip() + "\n\\end{abstract}"
    return None


def _body_step(text: str) -> str:
    r"""Keep ``\begin{document}...\end{document}``; carry ``\title`` and the abstract over from the preamble."""
    m = _BEGIN_DOCUMENT.search(text)
    if not m:
        return text
    preamble, body = text[: m.start()], _document_body(text)
    if "\\begin{abstract}" not in body:
        abstract = _preamble_abstract(preamble)
        if abstract:
            body = abstract + "\n\n" + body
    if not _TITLE.search(body):
        t = _TITLE.search(preamble)
        if t:
            _, p = _read_optional(preamble, t.end())
            r = _read_arg(preamble, p)
            if r is not None and r[0].strip():
                title = re.sub(r"\\\\\*?(?:\[[^\]]*\])?", " ", r[0])  # line breaks inside the title
                body = "\\title{" + " ".join(title.split()) + "}\n\n" + body
    return body


# ============================================================== T6/T7: math

_ALIGN_LIKE = frozenset({"align", "align*", "flalign", "flalign*", "eqnarray", "eqnarray*"})
_ALIGNAT = frozenset({"alignat", "alignat*", "xalignat", "xalignat*"})
_GATHER_LIKE = frozenset({"gather", "gather*", "multline", "multline*"})
_DISPLAY_ENVS = (
    frozenset({"equation", "equation*", "displaymath", "dmath", "dmath*"}) | _ALIGN_LIKE | _ALIGNAT | _GATHER_LIKE
)
_INLINE_ENVS = frozenset({"math"})
_MATH_OPEN = re.compile(
    r"\\\\|\\\$|\$\$|\$|\\\(|\\\[|\\begin\{("
    + "|".join(re.escape(e) for e in sorted(_DISPLAY_ENVS | _INLINE_ENVS, key=len, reverse=True))
    + r")\}"
)
_CLOSERS = {
    "$": re.compile(r"\\\\|\\\$|\$"),
    "$$": re.compile(r"\\\\|\\\$|\$\$"),
    "\\(": re.compile(r"\\\\|\\\)"),
    "\\[": re.compile(r"\\\\|\\\]"),
}
_CLOSE_TOKEN = {"$": "$", "$$": "$$", "\\(": "\\)", "\\[": "\\]"}
_BLANK_LINE = re.compile(r"\n[ \t]*\n")
_SUBEQUATIONS = re.compile(r"\\(?:begin|end)\{subequations\}")
# Numbering only, dropped from formulas (name -> (argument spec, None) as in _transform_commands).
_MATH_NOISE = {"label": ("{", None), "tag": ("{", None), "nonumber": ("s", None), "notag": ("s", None)}


def _find_close(text: str, pos: int, opener: str) -> tuple[int, int] | None:
    rx, want = _CLOSERS[opener], _CLOSE_TOKEN[opener]
    while True:
        m = rx.search(text, pos)
        if m is None:
            return None
        if m.group(0) == want:
            return m.start(), m.end()
        pos = m.end()


def _segments(text: str) -> list[tuple[str, str, str, str]]:
    r"""Split text into ``(kind, open, body, close)`` pieces.

    ``kind`` is ``text``, ``inline`` or ``display``; text pieces carry their
    content in ``open``. Escaped dollars, ``\\`` line breaks and unclosed
    delimiters are text. Inline math never crosses a blank line (TeX forbids
    it), so a stray ``$`` cannot swallow the rest of the paper.
    """
    out: list[tuple[str, str, str, str]] = []
    pos = start = 0
    while True:
        m = _MATH_OPEN.search(text, pos)
        if m is None:
            break
        tok = m.group(0)
        if tok in ("\\\\", "\\$"):
            pos = m.end()
            continue
        env = m.group(1)
        if env is not None:
            close = "\\end{" + env + "}"
            j = text.find(close, m.end())
            if j < 0:
                pos = m.end()
                continue
            body, end = text[m.end() : j], j + len(close)
            kind = "inline" if env in _INLINE_ENVS else "display"
        else:
            found = _find_close(text, m.end(), tok)
            if found is None:
                pos = m.end()
                continue
            body, end = text[m.end() : found[0]], found[1]
            close = _CLOSE_TOKEN[tok]
            kind = "display" if tok in ("$$", "\\[") else "inline"
            if kind == "inline" and _BLANK_LINE.search(body):
                pos = m.end()
                continue
        if m.start() > start:
            out.append(("text", text[start : m.start()], "", ""))
        out.append((kind, tok, body, close))
        pos = start = end
    if start < len(text):
        out.append(("text", text[start:], "", ""))
    return out


def _env_name(opener: str) -> str | None:
    m = re.match(r"\\begin\{([^}]*)\}", opener)
    return m.group(1) if m else None


def _clean_math_body(body: str) -> str:
    body = _transform_commands(body, _MATH_NOISE, mark="")
    body = _transform_commands(body, {"ensuremath": ("{", 0)}, mark="")
    return body.strip()


_ENSUREMATH = re.compile(r"\\ensuremath(?![A-Za-z@])")


def _ensuremath_to_dollars(text: str) -> str:
    out, pos = [], 0
    while True:
        m = _ENSUREMATH.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        r = _read_arg(text, m.end())
        if r is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        out.append(text[pos : m.start()] + "$" + r[0].strip() + "$")
        pos = r[1]


def _display(opener: str, body: str) -> str:
    env = _env_name(opener)
    body = _clean_math_body(body)
    if env in _ALIGN_LIKE:
        body = "\\begin{aligned}" + body + "\\end{aligned}"
    elif env in _ALIGNAT:
        r = _read_arg(body, 0)
        if r is not None:
            body = "\\begin{alignedat}{" + r[0].strip() + "}" + body[r[1] :].strip() + "\\end{alignedat}"
        else:
            body = "\\begin{aligned}" + body + "\\end{aligned}"
    elif env in _GATHER_LIKE:
        body = "\\begin{gathered}" + body + "\\end{gathered}"
    return "\\begin{equation}" + body + "\\end{equation}"


_IFMMODE = re.compile(r"\\\\|\\ifmmode(?![A-Za-z@])")
_IF_TOKENS = re.compile(r"\\\\|\\(if[A-Za-z@]*|else|fi)(?![A-Za-z@])")


def _split_conditional(text: str, pos: int) -> tuple[str, str, int] | None:
    r"""From just after an ``\if..`` token: ``(true branch, false branch, end after \fi)``."""
    depth, else_at = 0, None
    for m in _IF_TOKENS.finditer(text, pos):
        name = m.group(1)
        if name is None:
            continue
        if name.startswith("if"):
            if name in _PRIMITIVE_IFS or (name not in _NOT_CONDITIONALS and text[m.end() : m.end() + 1] != "{"):
                depth += 1
        elif name == "else":
            if depth == 0 and else_at is None:
                else_at = (m.start(), m.end())
        elif depth == 0:
            stop = else_at[0] if else_at else m.start()
            false = text[else_at[1] : m.start()] if else_at else ""
            return text[pos:stop].lstrip(" \t"), false.lstrip(" \t"), m.end()
        else:
            depth -= 1
    return None


def _choose_ifmmode(s: str, math: bool) -> str:
    out, pos = [], 0
    while True:
        m = _IFMMODE.search(s, pos)
        if m is None:
            out.append(s[pos:])
            return "".join(out)
        r = None if m.group(0) == "\\\\" else _split_conditional(s, m.end())
        if r is None:
            out.append(s[pos : m.end()])
            pos = m.end()
            continue
        out.append(s[pos : m.start()] + (r[0] if math else r[1]))
        pos = r[2]


def _in_math(text: str, pos: int) -> bool:
    off = 0
    for kind, opener, body, close in _segments(text):
        off += len(opener) + len(body) + len(close)
        if pos < off:
            return kind != "text"
    return False


def _resolve_ifmmode(text: str) -> str:
    r"""Replace each ``\ifmmode A\else B\fi`` by A inside math and by B in text.

    The mode is read at the position of ``\ifmmode`` itself: a ``$`` inside
    B (``\else$\nu$MSM\fi``) must not hide the conditional from us.
    """
    pos = 0
    for _ in range(1000):
        m = _IFMMODE.search(text, pos)
        while m is not None and m.group(0) == "\\\\":
            m = _IFMMODE.search(text, m.end())
        if m is None:
            break
        r = _split_conditional(text, m.end())
        if r is None:
            pos = m.end()
            continue
        branch = r[0] if _in_math(text, m.start()) else r[1]
        text = text[: m.start()] + branch + text[r[2] :]
        pos = m.start()
    return text


def _math_step(text: str) -> str:
    r"""Block math becomes ``\begin{equation}...\end{equation}``, inline math ``$...$``.

    Multi-line environments keep their ``&`` / ``\\`` layout inside
    ``aligned`` / ``alignedat`` / ``gathered``, so the result is still valid
    LaTeX. Labels, ``\tag`` and ``\nonumber`` (numbering only) are removed.
    ``\ifmmode A\else B\fi`` (left by macros and .bbl accents) is decided
    here, where the mode of every position is known.
    """
    text = _resolve_ifmmode(text)
    text = _SUBEQUATIONS.sub(_MARK, text)
    parts = []
    for kind, opener, body, _close in _segments(text):
        if kind == "text":
            parts.append(_ensuremath_to_dollars(opener))
        elif kind == "inline":
            parts.append("$" + _clean_math_body(body) + "$")
        else:
            parts.append(_display(opener, body))
    return "".join(parts)


# ============================================================== T8: formatting

# name -> (argument spec, action): None drops the command, an int keeps that
# argument's content, a callable builds the replacement from the arguments.
_FORMAT_SWITCHES = (
    "centering raggedright raggedleft RaggedRight noindent indent small footnotesize "
    "scriptsize tiny normalsize large Large LARGE huge Huge hline toprule midrule "
    "bottomrule clearpage newpage cleardoublepage pagebreak nopagebreak maketitle "
    "medskip bigskip smallskip FloatBarrier vfill hfill hfil vfil null protect relax "
    "onecolumn twocolumn tableofcontents listoffigures listoftables allowbreak sloppy "
    "fussy bfseries itshape rmfamily sffamily ttfamily mdseries upshape scshape slshape "
    "normalfont em bf it rm sf tt sc sl selectfont xspace balance raggedbottom "
    "flushbottom frenchspacing nonfrenchspacing phantomsection strut unskip hrule "
    "endhead endfirsthead endfoot endlastfoot IEEEpeerreviewmaketitle and AND begingroup endgroup "
    "displaystyle textstyle scriptstyle scriptscriptstyle"
).split()
_FORMAT_DROP_ARGS = {
    "vspace": "{", "hspace": "{", "label": "{", "setlength": "{{", "addtolength": "{{",
    "setcounter": "{{", "addtocounter": "{{", "rule": "[{{", "pagestyle": "{",
    "thispagestyle": "{", "bibliographystyle": "{", "captionsetup": "[{", "graphicspath": "{",
    "hypersetup": "{", "enlargethispage": "{", "linespread": "{", "fontsize": "{{", "color": "[{",
    "definecolor": "{{{", "needspace": "{", "addcontentsline": "{{{", "pdfbookmark": "[{{",
    "specialrule": "{{{", "cline": "{", "addlinespace": "[", "arrayrulecolor": "[{",
    "rowcolor": "[{[[", "columncolor": "[{[[", "cellcolor": "[{", "setcopyright": "{", "copyrightyear": "{",
    # front matter that is not body text
    "author": "[{", "affiliation": "[{", "affil": "[{", "address": "[{", "email": "[{",
    "thanks": "{", "institute": "{", "orcid": "{", "ORCID": "{", "date": "{", "keywords": "{",
    "ccsdesc": "[{", "urladdr": "[{", "curraddr": "[{", "subjclass": "[{", "received": "[{",
    "IEEEauthorblockN": "{", "IEEEauthorblockA": "{", "icmlauthor": "{{", "icmlaffiliation": "{{",
    "icmlcorrespondingauthor": "{{", "icmlkeywords": "{", "icmltitlerunning": "{",
    "authorrunning": "{", "titlerunning": "{", "runningtitle": "{", "runningauthor": "{",
    "shorttitle": "{", "shortauthors": "{", "markboth": "{{", "acmConference": "[{{{",
    "acmBooktitle": "{", "acmDOI": "{", "acmISBN": "{", "acmYear": "{", "acmJournal": "{",
    "acmVolume": "{", "acmNumber": "{", "acmArticle": "{", "acmMonth": "{", "acmPrice": "{",
    "acmSubmissionID": "{", "ead": "[{", "cortext": "[{", "fntext": "[{", "tnotetext": "[{",
    "corresp": "[{", "Description": "[{",  # acmart's screen-reader text for figures
}
_FORMAT_UNWRAP = {
    "textbf": ("{", 0), "textit": ("{", 0), "emph": ("{", 0), "textsc": ("{", 0),
    "textrm": ("{", 0), "textsf": ("{", 0), "texttt": ("{", 0), "textmd": ("{", 0),
    "textup": ("{", 0), "textsl": ("{", 0), "textnormal": ("{", 0), "underline": ("{", 0),
    "uline": ("{", 0), "uuline": ("{", 0), "textls": ("[{", 1), "hl": ("{", 0),
    "textcolor": ("[{{", 2), "colorbox": ("[{{", 2), "fcolorbox": ("[{{{", 3),
    "resizebox": ("{{{", 2), "scalebox": ("{[{", 2), "rotatebox": ("[{{", 2),
    "adjustbox": ("{{", 1), "parbox": ("[[[{{", 4), "makebox": ("[[{", 2),
    "framebox": ("[[{", 2), "raisebox": ("{[[{", 3), "fbox": ("{", 0), "centerline": ("{", 0),
    "texorpdfstring": ("{{", 0), "MakeUppercase": ("{", 0), "MakeLowercase": ("{", 0),
    "MakeTextUppercase": ("{", 0), "MakeTextLowercase": ("{", 0),
}
_LAYOUT_ENVS = {
    "center": "", "flushleft": "", "flushright": "", "minipage": "[[[{", "adjustbox": "{",
    "small": "", "footnotesize": "", "scriptsize": "", "raggedright": "", "spacing": "{",
    "singlespace": "", "onehalfspace": "", "doublespace": "", "sloppypar": "",
}
_DROP_ENVS = re.compile(r"\\begin\{(CCSXML)\}.*?\\end\{\1\}", re.S)
_FLOAT_OPTION = re.compile(
    r"(\\begin\{(?:figure|table|algorithm|sidewaysfigure|sidewaystable|SCfigure|wrapfigure|wraptable"
    r"|tabular|tabularx|tabular\*|longtable|subfigure|subtable)\*?\})[ \t]*\[[^\]\n]*\]"
)
_CMIDRULE = re.compile(r"\\cmidrule(?:\s*\([^)]*\))?\s*(?:\[[^\]]*\])?\s*\{[^{}]*\}")
_SKIP_GLUE = re.compile(
    r"\\[vh]skip\s*[-+]?[\d.]*\s*(?:\\?[A-Za-z]+)(?:\s*(?:plus|minus)\s*[-+]?[\d.]*\s*\\?[A-Za-z]+)*"
)
_TEXT_SYMBOLS = re.compile(r"\\\\|\\([@/-])")  # \@ \/ \- (spacing and hyphenation hints)
_LIST_OPEN = re.compile(
    r"\\begin\{(?:itemize|enumerate|description|compactitem|compactenum|compactdesc"
    r"|inparaenum|asparaenum|asparaitem)\*?\}"
)
_PARAM_ASSIGN = re.compile(
    r"\\(?:hangindent|hangafter|parindent|parskip|baselineskip|lineskip|tabcolsep|arraycolsep"
    r"|columnsep|leftskip|rightskip|itemsep|parsep|topsep|labelsep|leftmargin|emergencystretch"
    r"|tolerance|pretolerance|hbadness|vbadness|abovedisplayskip|belowdisplayskip|abovecaptionskip"
    r"|belowcaptionskip|textfloatsep|floatsep|intextsep)(?![A-Za-z@])"
    r"\s*(?:=\s*)?[-+]?(?:\d+\.?\d*|\.\d+)\s*(?:true\s*)?(?:pt|em|ex|cm|mm|in|bp|sp|pc|mu|dd|cc|\\[A-Za-z@]+)?"
)
_DIM = r"\s*[-+]?(?:\d+\.?\d*|\.\d+)\s*(?:pt|em|ex|cm|mm|in|bp|sp|pc|mu|dd|cc|\\[A-Za-z@]+)\s*"
# "\\[2pt]" and "\\*" are the same row/line break as "\\"; "\tabularnewline" too.
_ROW_BREAK = re.compile(r"(?<!\\)((?:\\\\)*)\\\\\*?(?:\s*\[" + _DIM + r"\])?|\\tabularnewline(?![A-Za-z@])")
_PAR = re.compile(r"\\\\|\\par(?![A-Za-z@])")
_LINEBREAK_WORDS = re.compile(r"\\\\|\\(?:newline|linebreak)(?![A-Za-z@])(?:\[[^\]]*\])?")


def _command_regex(names: Iterable[str]) -> re.Pattern:
    alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
    return re.compile(r"\\\\|\\(" + alt + r")(?![A-Za-z@])\*?")


_REGEX_CACHE: dict[tuple, re.Pattern] = {}


def _transform_commands(text: str, table: dict[str, tuple[str, object]], *, mark: str = _MARK,
                        passes: int = 6) -> str:
    """Apply a command table repeatedly (nested commands need several passes)."""
    key = tuple(sorted(table))
    rx = _REGEX_CACHE.get(key)
    if rx is None:
        rx = _REGEX_CACHE[key] = _command_regex(table)
    for _ in range(passes):
        out: list[str] = []
        pos = 0
        changed = False
        while True:
            m = rx.search(text, pos)
            if m is None:
                out.append(text[pos:])
                break
            if m.group(1) is None:  # "\\"
                out.append(text[pos : m.end()])
                pos = m.end()
                continue
            spec, action = table[m.group(1)]
            args, end = _read_args(text, m.end(), spec)
            if args is None:
                out.append(text[pos : m.end()])
                pos = m.end()
                continue
            if action is None:
                rep = mark
            elif callable(action):
                rep = action(args)
            else:
                rep = args[action] or ""
            original = text[m.start() : end]
            out.append(text[pos : m.start()])
            out.append(rep)
            changed |= rep != original
            pos = end
        text = "".join(out)
        if not changed:
            break
    return text


# siunitx: "\SI{3}{\milli\meter}" and "3~mm" are the same content.
_SI_PREFIXES = {
    "yocto": "y", "zepto": "z", "atto": "a", "femto": "f", "pico": "p", "nano": "n", "micro": "\u00b5",
    "milli": "m", "centi": "c", "deci": "d", "deca": "da", "deka": "da", "hecto": "h", "kilo": "k",
    "mega": "M", "giga": "G", "tera": "T", "peta": "P", "exa": "E", "zetta": "Z", "yotta": "Y",
    "kibi": "Ki", "mebi": "Mi", "gibi": "Gi", "tebi": "Ti",
}
_SI_UNITS = {
    "meter": "m", "metre": "m", "gram": "g", "second": "s", "ampere": "A", "kelvin": "K", "mole": "mol",
    "candela": "cd", "hertz": "Hz", "newton": "N", "pascal": "Pa", "joule": "J", "watt": "W",
    "coulomb": "C", "volt": "V", "farad": "F", "ohm": "\u03a9", "siemens": "S", "weber": "Wb",
    "tesla": "T", "henry": "H", "degreeCelsius": "\u00b0C", "celsius": "\u00b0C", "lumen": "lm",
    "lux": "lx", "becquerel": "Bq", "gray": "Gy", "sievert": "Sv", "katal": "kat", "radian": "rad",
    "steradian": "sr", "percent": "\\%", "degree": "\u00b0", "arcminute": "\u2032", "arcsecond": "\u2033",
    "minute": "min", "hour": "h", "day": "d", "liter": "L", "litre": "L", "hectare": "ha", "tonne": "t",
    "electronvolt": "eV", "dalton": "Da", "atomicmassunit": "u", "astronomicalunit": "au",
    "angstrom": "\u00c5", "bar": "bar", "decibel": "dB", "neper": "Np", "bel": "B", "byte": "B",
    "bit": "bit", "mmHg": "mmHg", "knot": "kn", "barn": "b",
}
_SI_POWERS = {"squared": "\u00b2", "cubed": "\u00b3", "square": "\u00b2", "cubic": "\u00b3"}


def _si_unit(spec: str | None) -> str:
    r"""``\kilo\meter\per\second\squared`` -> ``km/s²``; literal units (``m/s``) are kept."""
    spec = (spec or "").strip()
    if "\\" not in spec:
        return spec
    out, prefix, pending_power, sep = [], "", "", " "
    for m in re.finditer(r"\\([A-Za-z]+)(?:\s*\{([^{}]*)\})?|([^\\\s]+)", spec):
        name, arg, literal = m.group(1), m.group(2), m.group(3)
        if literal is not None:
            out.append(sep + literal)
            sep = " "
        elif name in _SI_PREFIXES:
            prefix += _SI_PREFIXES[name]
        elif name == "per":
            sep = "/"
        elif name in ("square", "cubic"):
            pending_power = _SI_POWERS[name]
        elif name in ("squared", "cubed") and out:
            out[-1] += _SI_POWERS[name]
        elif name in ("tothe", "raiseto") and out and arg:
            out[-1] += "^" + arg
        else:
            out.append(sep + prefix + _SI_UNITS.get(name, name) + pending_power)
            prefix, pending_power, sep = "", "", " "
    return "".join(out).strip()


def _si_list(nums: str | None) -> str:
    return ", ".join(x.strip() for x in (nums or "").split(";"))


def _escape_detokenized(s: str | None) -> str:
    r"""``\detokenize{bridge_good}`` prints ``bridge_good``, i.e. ``bridge\_good`` in normal text."""
    return re.sub(r"(?<!\\)([_&#%$])", r"\\\1", _s(s))


def _s(a: str | None) -> str:
    return (a or "").strip()


def _unit_value(opt: str | None, unit: str | None) -> str:
    r"""siunitx ``\unit[options]{u}`` vs. the ``units`` package's ``\unit[value]{u}``."""
    value = "" if opt is None or "=" in opt else opt.strip()
    return f"{value} {_si_unit(unit)}".strip()


# Every command the formatting step knows: name -> (argument spec, action).
_FORMAT_TABLE: dict[str, tuple[str, object]] = {
    **{n: ("s", None) for n in _FORMAT_SWITCHES},
    **{n: (spec, None) for n, spec in _FORMAT_DROP_ARGS.items()},
    **_FORMAT_UNWRAP,
    "includegraphics": ("[{", lambda a: "\\includegraphics{" + _s(a[1]) + "}"),
    "SI": ("[{[{", lambda a: f"{_s(a[1])} {_si_unit(a[3])}".strip()),
    "qty": ("[{{", lambda a: f"{_s(a[1])} {_si_unit(a[2])}".strip()),
    "si": ("[{", lambda a: _si_unit(a[1])),
    "unit": ("[{", lambda a: _unit_value(a[0], a[1])),
    "num": ("[{", lambda a: _s(a[1])),
    "ang": ("[{", lambda a: _s(a[1]).replace(";", "\u00b0 ") + "\u00b0"),
    "SIrange": ("[{{{", lambda a: f"{_s(a[1])}\u2013{_s(a[2])} {_si_unit(a[3])}"),
    "qtyrange": ("[{{{", lambda a: f"{_s(a[1])}\u2013{_s(a[2])} {_si_unit(a[3])}"),
    "numrange": ("[{{", lambda a: f"{_s(a[1])}\u2013{_s(a[2])}"),
    "SIlist": ("[{{", lambda a: f"{_si_list(a[1])} {_si_unit(a[2])}"),
    "qtylist": ("[{{", lambda a: f"{_si_list(a[1])} {_si_unit(a[2])}"),
    "numlist": ("[{", lambda a: _si_list(a[1])),
    "detokenize": ("{", lambda a: _escape_detokenized(a[0])),
}
_LAYOUT_ENV_RX = re.compile(r"\\(begin|end)\{(" + "|".join(map(re.escape, _LAYOUT_ENVS)) + r")\}")


def _unwrap_layout_envs(text: str) -> str:
    out, pos = [], 0
    while True:
        m = _LAYOUT_ENV_RX.search(text, pos)
        if m is None:
            out.append(text[pos:])
            return "".join(out)
        end = m.end()
        if m.group(1) == "begin" and _LAYOUT_ENVS[m.group(2)]:
            args, end2 = _read_args(text, end, _LAYOUT_ENVS[m.group(2)])
            if args is not None:
                end = end2
        out.append(text[pos : m.start()] + _MARK)
        pos = end


def _drop_list_options(text: str) -> str:
    """``\\begin{enumerate}[label=(\\roman*)]`` -> ``\\begin{enumerate}`` (label style is layout)."""
    out, pos = [], 0
    for m in _LIST_OPEN.finditer(text):
        if m.start() < pos:
            continue
        opt, end = _read_optional(text, m.end())
        if opt is not None:
            out.append(text[pos : m.end()])
            pos = end
    out.append(text[pos:])
    return "".join(out)


def _formatting_step(text: str) -> str:
    r"""Remove presentation commands; keep the words they decorate.

    Three groups: switches (``\centering``, ``\midrule``, ``\small``, ...)
    are dropped; commands with layout arguments (``\vspace{..}``,
    ``\label{..}``, author/affiliation front matter, ...) are dropped with
    their arguments; decorations (``\textbf{x}``, ``\emph{x}``,
    ``\textcolor{c}{x}``, ``\resizebox{w}{h}{x}``, ...) keep their content.
    Float placement options (``[htbp]``) and layout environments
    (``center``, ``minipage``) disappear; ``\includegraphics`` keeps its
    path but not its size options. Structure and citations are untouched.
    """
    text = _DROP_ENVS.sub(_MARK, text)
    text = _FLOAT_OPTION.sub(r"\1", text)
    text = _drop_list_options(text)
    text = _PARAM_ASSIGN.sub(_MARK, text)
    text = _ROW_BREAK.sub(lambda m: (m.group(1) or "") + "\\\\", text)
    text = _CMIDRULE.sub(_MARK, text)
    text = _SKIP_GLUE.sub(_MARK, text)
    text = _PAR.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "\n\n", text)
    text = _LINEBREAK_WORDS.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else " ", text)
    text = _TEXT_SYMBOLS.sub(lambda m: m.group(0) if m.group(1) is None else "", text)
    text = _unwrap_layout_envs(text)
    return _transform_commands(text, _FORMAT_TABLE)


# ============================================================== T11: canonical form

_ACCENTS = {"'": "\u0301", "`": "\u0300", "^": "\u0302", '"': "\u0308", "~": "\u0303",
            "=": "\u0304", ".": "\u0307", "u": "\u0306", "v": "\u030c", "H": "\u030b", "c": "\u0327"}
_ACCENT = re.compile(
    r"\\\\|\\([`'^\"~=.])\s*(?:\{\s*(\\i|[A-Za-z])\s*\}|(\\i(?![A-Za-z])|[A-Za-z]))"
    r"|\\([uvHc])\s*\{\s*(\\i|[A-Za-z])\s*\}"
)
_TEXT_MACROS = {
    "ldots": "\u2026", "dots": "\u2026", "textellipsis": "\u2026", "textendash": "\u2013",
    "textemdash": "\u2014", "textquoteright": "'", "textquoteleft": "'",
    "textquotedblleft": '"', "textquotedblright": '"', "ss": "\u00df", "o": "\u00f8",
    "O": "\u00d8", "ae": "\u00e6", "AE": "\u00c6", "aa": "\u00e5", "AA": "\u00c5",
    "l": "\u0142", "L": "\u0141", "textdagger": "\u2020", "dag": "\u2020", "textdaggerdbl": "\u2021",
    "ddag": "\u2021", "textsection": "\u00a7", "S": "\u00a7", "textparagraph": "\u00b6", "P": "\u00b6",
    "textbullet": "\u2022", "textdegree": "\u00b0", "textcopyright": "\u00a9", "copyright": "\u00a9",
    "textregistered": "\u00ae", "texttrademark": "\u2122", "pounds": "\u00a3", "textsterling": "\u00a3",
    "texteuro": "\u20ac", "euro": "\u20ac", "textperiodcentered": "\u00b7",
}
_TEXT_MACRO_RX = re.compile(
    r"\\\\|\\(" + "|".join(sorted(_TEXT_MACROS, key=len, reverse=True)) + r")(?![A-Za-z@])(?:\{\})?"
)
# "\" followed by a space or an end of line is a control space.
_TEXT_SPACES = re.compile(r"\\\\|\\(?:\s|[,;:]|thinspace(?![A-Za-z@])|enspace(?![A-Za-z@])|quad(?![A-Za-z@])|qquad(?![A-Za-z@]))")
_MATH_SPACING = re.compile(
    r"\\\\|\\[,;:!>\s]|\\(?:quad|qquad|enspace|thinspace|medspace|thickspace|negthinspace"
    r"|negmedspace|negthickspace|displaystyle|textstyle|scriptstyle|scriptscriptstyle)(?![A-Za-z@])"
)
_MATH_SIZING = re.compile(r"\\\\|\\(?:[Bb]igg?[lrm]?|left|right|middle)(?![A-Za-z@])(\s*\.)?")
_SINGLE_GROUP = re.compile(r"(?<!\\)([_^])\{\s*([A-Za-z0-9])\s*\}")
_MATH_TEXT_CMD = re.compile(
    r"\\(?:text|textrm|textit|textbf|textsf|texttt|textnormal|mbox|hbox|intertext|shortintertext)\s*(?=\{)"
)


def _accent(m: re.Match) -> str:
    if m.group(0) == "\\\\":
        return m.group(0)
    acc = m.group(1) or m.group(4)
    letter = m.group(2) or m.group(3) or m.group(5)
    letter = "i" if letter == "\\i" else letter
    return unicodedata.normalize("NFC", letter + _ACCENTS[acc])


_TRAILING_CS = re.compile(r"\\[A-Za-z@]+$")
_LIGATURE_CHARS = frozenset("-'`<>,!?")  # TeX ligatures that "{}" deliberately breaks


def _unwrap_bare_groups(text: str) -> str:
    r"""Remove grouping braces that are not command arguments: ``{word}`` -> ``word``."""
    stack: list[tuple[int, bool]] = []
    drop: set[int] = set()
    for m in _BRACE_TOKEN.finditer(text):
        tok, i = m.group(0), m.start()
        if tok == "{":
            j = i - 1
            while j >= 0 and text[j] in " \t\n":
                j -= 1
            prev = text[j] if j >= 0 else ""
            if prev == "}" and j in drop:
                is_arg = False  # "{\it a} {\bf b}": the previous group is itself bare
            else:
                is_arg = prev in "]}^_\ue001\\" or bool(
                    re.search(r"\\[A-Za-z@]+\*?$", text[max(0, j - 60) : j + 1])
                )
            stack.append((i, not is_arg))
        elif tok == "}" and stack:
            start, bare = stack.pop()
            if bare and i + 1 < len(text) and (text[i + 1].isalpha() or text[i + 1] == "@") \
                    and _TRAILING_CS.search(text, max(start + 1, i - 64), i):
                bare = False  # "{\alpha}b" must not become the control word "\alphab"
            if bare and i == start + 1 and 0 < start and i + 1 < len(text) \
                    and text[start - 1] in _LIGATURE_CHARS and text[i + 1] in _LIGATURE_CHARS:
                bare = False  # "-{}-" prints two hyphens: "--" would become an en dash
            if bare:
                drop.update((start, i))
    if not drop:
        return text
    return "".join(ch for k, ch in enumerate(text) if k not in drop)


def _canon_text(s: str) -> str:
    s = _ACCENT.sub(_accent, s)
    s = _TEXT_MACRO_RX.sub(lambda m: m.group(0) if m.group(1) is None else _TEXT_MACROS[m.group(1)], s)
    # Unicode quotes first: "\u2019\u2019" must end up like "''" in the same pass.
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("``", '"').replace("''", '"')
    s = re.sub(r"(?<!\\)`", "'", s)
    s = s.replace("---", "\u2014").replace("--", "\u2013")
    s = _TEXT_SPACES.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else " ", s)
    return _unwrap_bare_groups(s)


def _hide_math_text(s: str, shelf: _Shelf) -> str:
    out, pos = [], 0
    while True:
        m = _MATH_TEXT_CMD.search(s, pos)
        if m is None:
            out.append(s[pos:])
            return "".join(out)
        j = _match_brace(s, m.end())
        if j < 0:
            out.append(s[pos : m.end()])
            pos = m.end()
            continue
        cmd = re.sub(r"\s+", "", s[m.start() : m.end()])
        inner = " ".join(s[m.end() + 1 : j].split())
        out.append(s[pos : m.start()] + shelf.put(cmd + "{" + inner + "}"))
        pos = j + 1


def _canon_math(s: str) -> str:
    shelf = _Shelf()
    # A removed command is a token boundary; "~" is just a space in math.
    s = s.replace(_MARK, " ")
    s = re.sub(r"(?<!\\)~", " ", s)
    s = _hide_math_text(s, shelf)
    s = _MATH_SPACING.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "", s)
    s = _MATH_SIZING.sub(lambda m: m.group(0) if m.group(0) == "\\\\" else "", s)
    s = _SINGLE_GROUP.sub(r"\1\2", s)
    s = re.sub(r"(\\[A-Za-z@]+)\s+(?=[A-Za-z@])", "\\1" + _KEEP_SPACE, s)
    s = re.sub(r"\s+", "", s).replace(_KEEP_SPACE, " ")
    return shelf.restore(s)


_COLUMN_SPEC_ENVS = {
    "tabular": "[{", "array": "[{", "longtable": "[{", "supertabular": "{", "tabular*": "{[{",
    "tabularx": "{[{", "tabulary": "{[{", "xltabular": "{[{", "tblr": "[{", "longtblr": "[{",
}
_COLUMN_SPEC_OPEN = re.compile(
    r"\\begin\{(" + "|".join(sorted(map(re.escape, _COLUMN_SPEC_ENVS), key=len, reverse=True))
    + r")\}|\\multicolumn(?![A-Za-z@])"
)


def _hide_column_specs(text: str, shelf: _Shelf) -> str:
    r"""Shelve column specifications: in ``{p{3cm}@{}l}`` the braces are syntax, not grouping."""
    out, pos = [], 0
    for m in _COLUMN_SPEC_OPEN.finditer(text):
        if m.start() < pos:
            continue
        args, end = _read_args(text, m.end(), _COLUMN_SPEC_ENVS[m.group(1)] if m.group(1) else "{{")
        if args is None:
            continue
        out.append(text[pos : m.start()])
        out.append(shelf.put(text[m.start() : end]))
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _canonical_step(text: str) -> str:
    r"""Give equal content one spelling, so cross-version deduplication can see it.

    Text: accents to Unicode (``\'e`` -> é), LaTeX quotes and dashes to
    their characters, ``\ldots`` to …, control spaces to spaces, bare
    grouping braces removed. Math: spacing (``\,``, ``\quad``), sizing
    (``\left``, ``\big``) and style switches removed, ``x^{2}`` -> ``x^2``,
    insignificant whitespace removed (``x + y`` and ``x+y`` become equal).
    Citation keys, labels, URLs and paths are protected.
    """
    text = unicodedata.normalize("NFC", text)
    shelf = _Shelf()
    text = shelf.hide(_ARG_PROTECT, text)
    text = _hide_column_specs(text, shelf)
    parts = []
    for kind, opener, body, close in _segments(text):
        if kind == "text":
            parts.append(_canon_text(opener))
            continue
        body = _canon_math(body)
        if kind == "inline" and not body:
            parts.append(" ")  # "$\,$" was only spacing; "$$" would reopen as display math
        else:
            parts.append(opener + body + close)
    return shelf.restore("".join(parts))


# ============================================================== T9: whitespace


def _whitespace_step(text: str) -> str:
    """Collapse spaces, keep exactly one blank line between paragraphs.

    A line that only held removed commands disappears instead of becoming a
    blank line, which would split a paragraph in two.
    """
    shelf = _Shelf()
    text = shelf.hide(_ARG_PROTECT, text)
    text = re.sub(r"(?<!\\)~", " ", text)
    text = text.replace("\t", " ").replace("\u00a0", " ")
    lines = []
    for line in text.split("\n"):
        if _MARK in line:
            # "\foo<removed>bar" must not fuse into the control word "\foobar".
            line = re.sub(r"(\\[A-Za-z@]+)" + _MARK + r"+(?=[A-Za-z@])", r"\1 ", line)
            line = line.replace(_MARK, "")
            if not line.strip():
                continue
        lines.append(line.strip())
    text = "\n".join(lines)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return shelf.restore(text)


# ============================================================== cleanup


CLEANUP_STEPS: tuple[str, ...] = ("comments", "macros", "body", "math", "formatting", "canonical", "whitespace")
_STEP_FUNCS: dict[str, Callable[[str], str]] = {
    "comments": _strip_comments,
    "macros": _macros_step,
    "body": _body_step,
    "math": _math_step,
    "formatting": _formatting_step,
    "canonical": _canonical_step,
    "whitespace": _whitespace_step,
}
# Cumulative cleanup levels for the ablation figure of the report (T14).
ABLATION_LEVELS: dict[str, tuple[str, ...]] = {
    "raw": (),
    "whitespace": ("comments", "whitespace"),
    "math": ("comments", "macros", "body", "math", "whitespace"),
    "full": CLEANUP_STEPS,
}


def cleanup_latex(source: str, *, steps: Iterable[str] | None = None) -> str:
    r"""Normalize formatting and math while preserving structural commands and cites.

    Runs :data:`CLEANUP_STEPS` in order (``steps`` selects a subset, for the
    ablation study). The result keeps sectioning commands, ``\appendix``,
    list environments and ``\item``, floats and ``\caption``, theorem-like
    environments, every ``\cite`` variant, ``thebibliography``/``\bibitem``
    and blank lines between paragraphs. Math appears only as ``$...$`` and
    ``\begin{equation}...\end{equation}``. The function is idempotent:
    ``cleanup_latex(cleanup_latex(x)) == cleanup_latex(x)``.
    """
    chosen = CLEANUP_STEPS if steps is None else tuple(steps)
    unknown = set(chosen) - set(CLEANUP_STEPS)
    if unknown:
        raise ValueError(f"unknown cleanup steps: {sorted(unknown)}")
    text = source.replace("\r\n", "\n").replace("\r", "\n")
    shelf = _Shelf()
    text = shelf.hide(_VERBATIM_ENV, text)
    text = shelf.hide(_VERB, text)
    for name in CLEANUP_STEPS:
        if name in chosen:
            text = _STEP_FUNCS[name](text)
    text = text.replace(_MARK, "")
    return shelf.restore(text)


# ============================================================== T12: validation

_ENV_TOKEN = re.compile(r"\\\\|\\(begin|end)\s*\{([^{}]*)\}")
_DOLLAR = re.compile(r"\\\\|\\\$|\$")
# Unclosed delimiters, or environments the math step should have rewritten.
_LEFTOVER_MATH = re.compile(
    r"\\\\|\\\[|\\\(|\\begin\{(?:align|alignat|flalign|gather|multline|eqnarray|displaymath"
    r"|math|equation\*)\*?\}"
)
_LEFTOVER_FORMAT = re.compile(
    r"\\\\|\\(?:centering|midrule|toprule|bottomrule|hline|vspace|hspace|noindent|textbf|emph)(?![A-Za-z@])"
)
_LEFTOVER_DEF = re.compile(r"\\\\|\\(?:newcommand|renewcommand|def|newenvironment)(?![A-Za-z@])")


def validate_cleaned(text: str) -> list[str]:
    """Self-check a cleaned text; returns issue codes (empty list = well-formed).

    Codes: ``unbalanced_braces``, ``odd_dollar``, ``unbalanced_env``,
    ``leftover_math`` (a delimiter that should have been normalized),
    ``leftover_format``, ``leftover_definition`` and ``leftover_conditional``
    (a TeX test that could not be decided, kept verbatim).
    """
    issues = []
    if not _braces_balanced(text):
        issues.append("unbalanced_braces")
    if sum(1 for m in _DOLLAR.finditer(text) if m.group(0) == "$") % 2:
        issues.append("odd_dollar")
    stack: list[str] = []
    ok = True
    for m in _ENV_TOKEN.finditer(text):
        if m.group(1) is None:
            continue
        name = m.group(2).strip()
        if m.group(1) == "begin":
            stack.append(name)
        elif stack and stack[-1] == name:
            stack.pop()
        else:
            ok = False
            break
    if not ok or stack:
        issues.append("unbalanced_env")
    # Segment first: "$a$$b$" is two inline formulas, not a "$$" display.
    segments = _segments(text)
    leftover = any(
        kind != "text" and opener not in ("$", "\\begin{equation}") for kind, opener, _b, _c in segments
    ) or any(
        m.group(0) != "\\\\"
        for kind, chunk, _b, _c in segments if kind == "text"
        for m in _LEFTOVER_MATH.finditer(chunk)
    )
    if leftover:
        issues.append("leftover_math")
    if any(m.group(0) != "\\\\" for m in _LEFTOVER_FORMAT.finditer(text)):
        issues.append("leftover_format")
    if any(m.group(0) != "\\\\" for m in _LEFTOVER_DEF.finditer(text)):
        issues.append("leftover_definition")
    if any(m.group(1) in _PRIMITIVE_IFS for m in _IF_TOKENS.finditer(text)):
        issues.append("leftover_conditional")
    return issues


# ============================================================== T13: statistics


def _version_number(name: str) -> int | None:
    m = re.fullmatch(r"v(\d+)", name)
    return int(m.group(1)) if m else None


# Columns of a gather_report row, in CSV order.
REPORT_FIELDS = (
    "paper", "version", "status", "root", "root_reason", "n_tex_total", "n_tex_used", "n_tex_unused",
    "n_bib_total", "n_bib_used", "n_missing", "missing", "missing_ext", "notes", "chars_by_file",
    "source_chars", "cleaned_chars", "issues", "idempotent", "gather_sec", "cleanup_sec",
)
_INT_FIELDS = frozenset({"version", "n_tex_total", "n_tex_used", "n_tex_unused", "n_bib_total",
                         "n_bib_used", "n_missing", "source_chars", "cleaned_chars"})
_FLOAT_FIELDS = frozenset({"gather_sec", "cleanup_sec"})


def gather_report(version_dir: Path, *, cleanup: bool = True, check_idempotence: bool = False,
                  dump_dir: Path | None = None) -> dict:
    """One statistics row for a version: status, file counts, missing files, timings.

    With ``dump_dir``, also write ``<dump_dir>/<paper>/<version>/source.tex``
    (gathered, comments stripped) and ``cleaned.tex`` (after cleanup) so the
    result can be inspected by hand.

    ``status`` is ``ok``, ``partial`` (some include or .bib missing),
    ``empty``, ``missing_dir``, ``no_root``, ``ambiguous_root`` or ``error``.
    """
    version_dir = Path(version_dir)
    files = [p for p in version_dir.rglob("*") if p.is_file()] if version_dir.is_dir() else []
    n_tex = sum(p.suffix.lower() == ".tex" for p in files)
    n_bib = sum(p.suffix.lower() == ".bib" for p in files)
    row = dict.fromkeys(REPORT_FIELDS, "")
    row.update(
        paper=version_dir.parent.name, version=_version_number(version_dir.name),
        n_tex_total=n_tex, n_tex_used=0, n_tex_unused=n_tex, n_bib_total=n_bib, n_bib_used=0,
        n_missing=0, source_chars=0, cleaned_chars=0, gather_sec=0.0, cleanup_sec=0.0,
    )
    t0 = time.perf_counter()
    try:
        tex = gather_tex(version_dir)
    except FileNotFoundError:
        row["status"] = "missing_dir"
        return row
    except NoRootError:
        row["status"] = "no_root"
        return row
    except AmbiguousRootError as e:
        row["status"], row["notes"] = "ambiguous_root", str(e)
        return row
    except Exception as e:  # keep the corpus scan going, but report it
        row["status"], row["notes"] = "error", f"{type(e).__name__}: {e}"
        return row
    finally:
        row["gather_sec"] = round(time.perf_counter() - t0, 4)
    if tex is None:
        row["status"] = "empty"
        return row
    row.update(
        status="partial" if tex.missing else "ok",
        root=tex.root.relative_to(version_dir.resolve()).as_posix(),
        root_reason=tex.root_reason,
        n_tex_used=len(tex.tex_files),
        n_tex_unused=n_tex - len(tex.tex_files),
        n_bib_used=len(tex.bib_files),
        n_missing=len(tex.missing),
        missing=";".join(tex.missing),
        missing_ext=";".join(Path(m).suffix.lower() or "(none)" for m in tex.missing),
        notes=";".join(tex.notes),
        chars_by_file=";".join(f"{k}:{v}" for k, v in chars_by_file(tex).items()),
        source_chars=len(tex.source),
    )
    out_dir = Path(dump_dir) / version_dir.parent.name / version_dir.name if dump_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "source.tex").write_text(tex.source, encoding="utf-8")
    if cleanup:
        t1 = time.perf_counter()
        try:
            cleaned = cleanup_latex(tex.source)
        except Exception as e:
            row["status"], row["notes"] = "error", f"cleanup {type(e).__name__}: {e}"
            return row
        row["cleanup_sec"] = round(time.perf_counter() - t1, 4)
        row["cleaned_chars"] = len(cleaned)
        if out_dir:
            (out_dir / "cleaned.tex").write_text(cleaned, encoding="utf-8")
        row["issues"] = ";".join(validate_cleaned(cleaned))
        if check_idempotence:
            row["idempotent"] = cleanup_latex(cleaned) == cleaned
    return row


def _version_dirs(paper_dir: Path) -> list[Path]:
    return sorted(
        (d for d in paper_dir.iterdir() if d.is_dir() and _version_number(d.name) is not None),
        key=lambda d: _version_number(d.name),
    )


def _paper_rows(paper_dir: Path, cleanup: bool, check_idempotence: bool,
                dump_dir: Path | None = None, skip: frozenset[int] = frozenset()) -> list[dict]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GatherWarning)
        return [gather_report(v, cleanup=cleanup, check_idempotence=check_idempotence, dump_dir=dump_dir)
                for v in _version_dirs(paper_dir) if _version_number(v.name) not in skip]


def _coerce_row(raw: dict) -> dict | None:
    """A checkpoint CSV row back to gather_report types; ``None`` if it is damaged."""
    if None in raw or any(v is None for v in raw.values()):
        return None  # too few or too many fields: a line cut short by a crash
    row = dict(raw)
    try:
        for k in _INT_FIELDS:
            row[k] = int(row[k])
        for k in _FLOAT_FIELDS:
            row[k] = float(row[k])
    except ValueError:
        return None
    row["idempotent"] = {"True": True, "False": False}.get(row["idempotent"], "")
    return row


def read_checkpoint(path: Path) -> list[dict]:
    """Rows already written to a :func:`scan_corpus` checkpoint CSV (empty if absent).

    Damaged lines (e.g. the last one, if the process was killed mid-write) are
    dropped so those versions are processed again; a later row for the same
    version replaces an earlier one.
    """
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        if tuple(reader.fieldnames) != REPORT_FIELDS:
            raise ValueError(f"{path} has different columns than this version of the report; "
                             "write to a new file instead of resuming")
        rows: dict[tuple[str, int], dict] = {}
        for raw in reader:
            row = _coerce_row(raw)
            if row is not None:
                rows[row["paper"], row["version"]] = row
    return list(rows.values())


class _Checkpoint:
    """Append-only CSV that is flushed to disk after every paper."""

    def __init__(self, path: Path, rows: list[dict]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Rewrite what was kept (drops a truncated last line), then append to it.
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, self.path)
        self._file = open(self.path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=REPORT_FIELDS)

    def write(self, rows: list[dict]) -> None:
        self._writer.writerows(rows)
        self._file.flush()
        os.fsync(self._file.fileno())

    def close(self) -> None:
        self._file.close()


def scan_corpus(data_dir: Path, *, cleanup: bool = True, check_idempotence: bool = False,
                workers: int = 1, limit: int | None = None, dump_dir: Path | None = None,
                checkpoint: Path | None = None, resume: bool = False) -> list[dict]:
    """Run :func:`gather_report` on every ``<paper>/v<N>`` directory under ``data_dir``.

    ``workers > 1`` processes papers in parallel (call it under
    ``if __name__ == "__main__":`` on Windows).

    With ``checkpoint``, each paper's rows are appended to that CSV as soon as
    the paper is done, so an interrupted scan keeps its work. ``resume=True``
    keeps the rows already in the file and processes only the versions it
    does not list; without it the file is started over. A paper whose worker
    process dies (out of memory, crash) is reported with a warning and left
    out of the file, so the next resumed run tries it again.

    Returns every row of the scan, those read back from the checkpoint
    included, sorted by paper and version.
    """
    if resume and checkpoint is None:
        raise ValueError("resume=True needs a checkpoint file")
    papers = sorted(
        p for p in Path(data_dir).iterdir()
        if p.is_dir() and any(_version_number(d.name) is not None for d in p.iterdir() if d.is_dir())
    )
    if limit is not None:
        papers = papers[:limit]
    previous = read_checkpoint(checkpoint) if resume else []
    done: dict[str, set[int]] = {}
    for row in previous:
        done.setdefault(row["paper"], set()).add(row["version"])
    todo = []
    for p in papers:
        skip = frozenset(done.get(p.name, ()))
        if not skip or any(_version_number(v.name) not in skip for v in _version_dirs(p)):
            todo.append((p, skip))

    sink = _Checkpoint(checkpoint, previous) if checkpoint is not None else None
    rows = list(previous)
    failed: list[str] = []

    def keep(paper_rows: list[dict]) -> None:
        rows.extend(paper_rows)
        if sink is not None:
            sink.write(paper_rows)

    try:
        if workers <= 1:
            for p, skip in todo:
                keep(_paper_rows(p, cleanup, check_idempotence, dump_dir, skip))
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_paper_rows, p, cleanup, check_idempotence, dump_dir, skip): p
                           for p, skip in todo}
                for fut in as_completed(futures):
                    try:
                        keep(fut.result())
                    except Exception as e:  # BrokenProcessPool, or an error outside gather_report
                        failed.append(f"{futures[fut].name} ({type(e).__name__})")
    finally:
        if sink is not None:
            sink.close()
    if failed:
        warnings.warn(f"{len(failed)} paper(s) not processed, rerun with resume to retry: "
                      + ", ".join(sorted(failed)), RuntimeWarning, stacklevel=2)
    rows.sort(key=lambda r: (r["paper"], r["version"]))
    return rows


# ============================================================== T14: ablation


def dedup_ablation(sources_by_version: dict[int, str],
                   levels: dict[str, tuple[str, ...]] | None = None) -> dict[str, dict[str, int]]:
    """Count paragraph units before/after exact deduplication at each cleanup level.

    Paragraphs (blank-line separated, whitespace-collapsed) are a parser-
    independent proxy for elements: the more consistent the cleanup, the more
    paragraphs of one version exactly equal a paragraph of another one.
    Pass the ``GatherTex.source`` of every version of one publication.
    """
    result = {}
    for level, steps in (levels or ABLATION_LEVELS).items():
        units: list[list[str]] = []
        for src in sources_by_version.values():
            cleaned = cleanup_latex(src, steps=steps)
            units.append([u for u in (" ".join(p.split()) for p in _BLANK_LINE.split(cleaned)) if u])
        total = sum(len(u) for u in units)
        distinct = len({p for u in units for p in u})
        result[level] = {"units_total": total, "units_distinct": distinct, "units_shared": total - distinct}
    return result


# ============================================================== T17: before/after viewer

_BLOCK_MARK = "\ue004"  # private-use character: survives every cleanup step untouched
_BLOCK_TOKEN = re.compile(
    r"\\\\|\\[{}%$&#_]|\\(begin|end)\s*\{[^{}]*\}|([{}])|(\n[ \t]*\n\s*)"
)
_BLOCK_SPLIT = re.compile(_BLOCK_MARK + r"(\d+)" + _BLOCK_MARK)
_DIFF_TOKEN = re.compile(r"\s+|\\[A-Za-z@]+\*?|[A-Za-z0-9]+|.", re.S)


def _top_level_blocks(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Paragraph-like blocks of ``text[start:end]``, never cutting a group or an environment."""
    spans, block_start, depth, envs = [], start, 0, 0
    for m in _BLOCK_TOKEN.finditer(text, start, end):
        if m.group(1):
            envs = envs + 1 if m.group(1) == "begin" else max(envs - 1, 0)
        elif m.group(2):
            depth = depth + 1 if m.group(2) == "{" else max(depth - 1, 0)
        elif m.group(3) and depth == 0 and envs == 0:
            spans.append((block_start, m.start()))
            block_start = m.end()
    spans.append((block_start, end))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def clean_blocks(tex: GatherTex) -> list[dict]:
    """T17: pair every body block of ``tex.source`` with its cleaned text.

    The whole document is cleaned once, with an invisible marker before each
    block, so macros, conditionals and preamble definitions apply exactly as
    in :func:`cleanup_latex`; the cleaned output is then cut at the markers.
    Each item: ``{"origin": "file:line", "before": str, "after": str}``. The
    first item may be the title/abstract carried over from the preamble.
    """
    src = tex.source
    b = _BEGIN_DOCUMENT.search(src)
    if b is None:
        return [{"origin": "", "before": src, "after": cleanup_latex(src)}]
    e = _END_DOCUMENT.search(src, b.end())
    body_end = e.start() if e else len(src)
    spans = _top_level_blocks(src, b.end(), body_end)
    marked = [src[: b.end()]]
    for i, (s, t) in enumerate(spans):
        marked.append(f"\n\n{_BLOCK_MARK}{i}{_BLOCK_MARK}\n\n{src[s:t]}")
    marked.append("\n\n" + src[body_end:])
    cleaned = cleanup_latex("".join(marked))
    parts = _BLOCK_SPLIT.split(cleaned)
    after = {int(parts[k]): parts[k + 1].strip("\n") for k in range(1, len(parts) - 1, 2)}
    if len(after) != len(spans):  # a marker was swallowed: show the version as one block
        return [{"origin": "", "before": src, "after": _BLOCK_SPLIT.sub("", cleaned)}]
    blocks = []
    head = parts[0].strip("\n")
    if head:
        blocks.append({"origin": "preamble", "before": "", "after": head})
    for i, (s, t) in enumerate(spans):
        f, ln = locate(tex, s + len(src[s:t]) - len(src[s:t].lstrip())) or ("", 0)
        blocks.append({"origin": f"{f}:{ln}" if f else "", "before": src[s:t].strip("\n"), "after": after[i]})
    return blocks


def _diff_markup(before: str, after: str) -> tuple[str, str, bool]:
    """Token-level diff: removed tokens marked in ``before``, added ones in ``after``."""
    a, b = _DIFF_TOKEN.findall(before), _DIFF_TOKEN.findall(after)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    left, right = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        x, y = html.escape("".join(a[i1:i2])), html.escape("".join(b[j1:j2]))
        if op == "equal":
            left.append(x)
            right.append(y)
            continue
        if x:
            left.append(f"<del>{x}</del>")
        if y:
            right.append(f"<ins>{y}</ins>")
    changed = " ".join(before.split()) != " ".join(after.split())
    return "".join(left), "".join(right), changed


_VIEWER_TEMPLATE = """<div class="bav" id="__ID__">
<style>
#__ID__{--bg:#ffffff;--fg:#1f2328;--muted:#656d76;--line:#d0d7de;--panel:#f6f8fa;--del:#ffebe9;--delfg:#82071e;--ins:#dafbe1;--insfg:#116329;
 font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:12px;max-width:100%;box-sizing:border-box}
@media (prefers-color-scheme:dark){html:not([data-theme=light]) #__ID__{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--panel:#161b22;--del:#4b1d22;--delfg:#ffa198;--ins:#16351f;--insfg:#7ee787}}
html[data-theme=dark] #__ID__{--bg:#0d1117;--fg:#e6edf3;--muted:#8d96a0;--line:#30363d;--panel:#161b22;--del:#4b1d22;--delfg:#ffa198;--ins:#16351f;--insfg:#7ee787}
#__ID__ .bar{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-bottom:8px}
#__ID__ select{font:inherit;padding:4px 6px;max-width:100%;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:6px}
#__ID__ .stats{color:var(--muted);font-size:13px}
#__ID__ .head,#__ID__ .row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
#__ID__ .head{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin:6px 0}
#__ID__ .rows{max-height:640px;overflow:auto;border-top:1px solid var(--line)}
#__ID__ .row{padding:6px 0;border-bottom:1px solid var(--line)}
#__ID__ .cell{background:var(--panel);border-radius:6px;padding:6px 8px;min-width:0}
#__ID__ pre{margin:0;white-space:pre-wrap;word-break:break-word;font:12.5px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace}
#__ID__ .loc{color:var(--muted);font-size:11.5px;margin-bottom:2px;font-family:ui-monospace,Consolas,monospace}
#__ID__ del{background:var(--del);color:var(--delfg);text-decoration:line-through}
#__ID__ ins{background:var(--ins);color:var(--insfg);text-decoration:none}
@media (max-width:700px){#__ID__ .head,#__ID__ .row{grid-template-columns:1fr}}
</style>
<div class="bar">
 <label>Version <select class="pick"></select></label>
 <label><input type="checkbox" class="only" checked> only changed blocks</label>
 <span class="stats"></span>
</div>
<div class="head"><div>Before (gathered source)</div><div>After cleanup_latex</div></div>
<div class="rows"></div>
<script>
(function(){
 var data=__DATA__;var root=document.getElementById("__ID__");
 var pick=root.querySelector(".pick"),only=root.querySelector(".only"),rows=root.querySelector(".rows"),stats=root.querySelector(".stats");
 data.forEach(function(v,i){var o=document.createElement("option");o.value=i;o.textContent=v.label;pick.appendChild(o);});
 function render(){
  var v=data[+pick.value]||data[0];if(!v){rows.textContent="No version.";return;}
  var shown=v.blocks.filter(function(b){return !only.checked||b.changed;});
  stats.textContent=v.n_blocks+" blocks, "+v.changed+" changed \\u00b7 "+v.source_chars+" \\u2192 "+v.cleaned_chars+" chars"+(v.issues?" \\u00b7 issues: "+v.issues:"");
  rows.innerHTML=shown.map(function(b){
   return '<div class="row"><div class="cell"><div class="loc">'+b.origin+'</div><pre>'+b.before+'</pre></div><div class="cell"><div class="loc">&nbsp;</div><pre>'+b.after+'</pre></div></div>';
  }).join("")||'<div class="row"><div class="cell">No changed block.</div></div>';
 }
 pick.addEventListener("change",render);only.addEventListener("change",render);render();
})();
</script>
</div>"""


def before_after_html(version_dirs: Iterable[Path], *, max_blocks: int = 120, max_chars: int = 4000) -> str:
    """T17: a self-contained HTML viewer of gathered vs cleaned text, block by block.

    Pick a version in the drop-down; removed tokens are struck out on the left,
    added ones highlighted on the right, and every block shows the file and
    line it comes from (T15). The HTML has no external dependency, so it works
    in a Jupyter notebook, in a built Jupyter Book page and as a standalone file.
    """
    data = []
    for v in version_dirs:
        v = Path(v)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GatherWarning)
            try:
                tex = gather_tex(v)
            except Exception as e:  # show the failure instead of hiding the version
                data.append({"label": f"{v.parent.name}/{v.name} ({type(e).__name__})", "blocks": [], "n_blocks": 0,
                             "changed": 0, "source_chars": 0, "cleaned_chars": 0, "issues": str(e)})
                continue
        if tex is None:
            continue
        pairs = clean_blocks(tex)
        blocks = []
        for blk in pairs[:max_blocks]:
            left, right, diff = _diff_markup(blk["before"][:max_chars], blk["after"][:max_chars])
            blocks.append({"origin": html.escape(blk["origin"]), "before": left, "after": right, "changed": diff})
        changed = sum(" ".join(b["before"].split()) != " ".join(b["after"].split()) for b in pairs)
        cleaned = cleanup_latex(tex.source)
        data.append({
            "label": f"{v.parent.name}/{v.name}", "blocks": blocks, "n_blocks": len(pairs), "changed": changed,
            "source_chars": len(tex.source), "cleaned_chars": len(cleaned),
            "issues": ", ".join(validate_cleaned(cleaned)),
        })
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    # Deterministic id (same page on every book build), unique per viewer in one notebook.
    uid = f"bav{hashlib.sha1(payload.encode()).hexdigest()[:10]}{next(_PLACEHOLDER_IDS)}"
    return _VIEWER_TEMPLATE.replace("__ID__", uid).replace("__DATA__", payload)


def show_before_after(version_dirs: Iterable[Path], **kwargs):
    """T17 in a notebook / Jupyter Book cell: ``show_before_after(Path("data/2609-27836").glob("v*"))``."""
    from IPython.display import HTML  # imported lazily: the module itself needs only the stdlib

    return HTML(before_after_html(version_dirs, **kwargs))


# ============================================================== CLI


def _summary(rows: list[dict], elapsed: float, resumed: int = 0) -> str:
    status = Counter(r["status"] for r in rows)
    ext = Counter(e for r in rows for e in r["missing_ext"].split(";") if e)
    issues = Counter(i for r in rows for i in r["issues"].split(";") if i)
    reasons = Counter(r["root_reason"].split(";")[-1].strip() for r in rows if r["root_reason"])
    new = len(rows) - resumed
    lines = [
        f"versions: {len(rows)}" + (f" ({resumed} from checkpoint, {new} this run)" if resumed else "")
        + f"   wall time: {elapsed:.1f}s"
        + (f"   avg/version this run: {elapsed / new:.3f}s" if new else ""),
        f"gather: {sum(r['gather_sec'] for r in rows):.1f}s   cleanup: {sum(r['cleanup_sec'] for r in rows):.1f}s",
        "status: " + ", ".join(f"{k}={v}" for k, v in status.most_common()),
        "root choice: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()),
        f"tex files used/total: {sum(r['n_tex_used'] for r in rows)}/{sum(r['n_tex_total'] for r in rows)}   "
        f"bib used/total: {sum(r['n_bib_used'] for r in rows)}/{sum(r['n_bib_total'] for r in rows)}",
        "missing by extension: " + (", ".join(f"{k}={v}" for k, v in ext.most_common()) or "none"),
        "cleanup issues: " + (", ".join(f"{k}={v}" for k, v in issues.most_common()) or "none"),
    ]
    idem = [r["idempotent"] for r in rows if r["idempotent"] != ""]
    if idem:
        lines.append(f"idempotent: {sum(idem)}/{len(idem)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Gather and clean every paper version under a data directory.")
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--out", type=Path,
                    help="CSV with one row per version, written after every paper (checkpoint)")
    ap.add_argument("--resume", action="store_true",
                    help="keep the rows already in --out and process only the versions it lacks")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, help="only the first N papers")
    ap.add_argument("--no-cleanup", action="store_true", help="gather only")
    ap.add_argument("--check-idempotence", action="store_true")
    ap.add_argument("--dump-dir", type=Path,
                    help="write <paper>/<version>/source.tex and cleaned.tex here (must be outside data_dir)")
    ap.add_argument("--html", type=Path, help="write a before/after viewer (T17) to this .html file")
    ap.add_argument("--html-limit", type=int, default=20, help="versions in the viewer (default 20)")
    args = ap.parse_args(argv)
    if args.dump_dir and args.dump_dir.resolve().is_relative_to(args.data_dir.resolve()):
        ap.error("--dump-dir must not be inside data_dir (it would add .tex files to the papers)")
    if args.resume and not args.out:
        ap.error("--resume needs --out (the checkpoint to continue)")
    try:
        resumed = len(read_checkpoint(args.out)) if args.resume else 0
    except ValueError as e:
        ap.error(str(e))
    t0 = time.perf_counter()
    rows = scan_corpus(args.data_dir, cleanup=not args.no_cleanup, check_idempotence=args.check_idempotence,
                       workers=args.workers, limit=args.limit, dump_dir=args.dump_dir,
                       checkpoint=args.out, resume=args.resume)
    elapsed = time.perf_counter() - t0
    print(_summary(rows, elapsed, resumed))
    if args.out:
        print(f"report: {args.out.resolve()}")
    if args.dump_dir:
        print(f"dumped .tex files: {args.dump_dir.resolve()}")
    if args.html:
        shown = [args.data_dir / r["paper"] / f"v{r['version']}" for r in rows if r["status"] in ("ok", "partial")]
        page = before_after_html(shown[: args.html_limit])
        args.html.write_text(
            "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<title>Cleanup before and after</title>"
            "<style>body{margin:16px;background:#fff}@media (prefers-color-scheme:dark){body{background:#0d1117}}</style>"
            f"</head><body>{page}</body></html>\n",
            encoding="utf-8",
        )
        print(f"before/after viewer: {args.html.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
