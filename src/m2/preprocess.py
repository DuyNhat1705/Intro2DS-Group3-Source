r"""Preprocessing: comment stripping, ``\iffalse..\fi`` / ``comment`` env removal, literal masking
(``\verb``, verbatim-like environments) and expansion of author-defined macros.

Literal spans are replaced by placeholders made of Unicode private-use characters, so that no
later stage (comment stripping, cleanup, sentence splitting) can damage them:

* ``\uE000<n>\uE001`` -- inline literal (``\verb``, ``\url`` argument); restored as plain text;
* ``\uE002<n>\uE003`` -- block literal (``verbatim``, ``lstlisting`` ...); dropped from prose and
  counted in the statistics.

All functions are linear-time and never raise on malformed input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config as C

IL_OPEN, IL_CLOSE, BL_OPEN, BL_CLOSE = "\ue000", "\ue001", "\ue002", "\ue003"
_PH_INLINE_RE = re.compile("\ue000(\\d+)\ue001")
_PH_BLOCK_RE = re.compile("\ue002(\\d+)\ue003")


@dataclass
class Literals:
    """Store of masked literal spans of one document."""

    items: list[str] = field(default_factory=list)
    n_block: int = 0

    def inline(self, text: str) -> str:
        self.items.append(text)
        return f"{IL_OPEN}{len(self.items) - 1}{IL_CLOSE}"

    def block(self, text: str) -> str:
        self.items.append(text)
        self.n_block += 1
        return f"{BL_OPEN}{len(self.items) - 1}{BL_CLOSE}"

    def restore_inline(self, text: str) -> str:
        if IL_OPEN not in text:
            return text
        return _PH_INLINE_RE.sub(lambda m: self._get(m.group(1)), text)

    def _get(self, idx: str) -> str:
        try:
            return self.items[int(idx)]
        except (ValueError, IndexError):
            return ""


def drop_block_placeholders(text: str) -> str:
    return _PH_BLOCK_RE.sub(" ", text) if BL_OPEN in text else text


# ---------------------------------------------------------------------------------------
# Literal masking
# ---------------------------------------------------------------------------------------
_VERB_ENV_NAMES = "|".join(re.escape(e) for e in sorted(C.VERBATIM_ENVS, key=len, reverse=True))
_VERB_ENV_RE = re.compile(r"\\begin\s*\{(" + _VERB_ENV_NAMES + r")\}")
# \verb|..|, \verb*+..+, \lstinline|..|, \mintinline{lang}|..| ; delimiter = any non-letter.
_VERB_INLINE_RE = re.compile(
    r"\\(?:verb\*?|lstinline(?:\[[^\]\n]{0,200}\])?|mintinline\{[^}\n]{0,40}\}|spverb)"
    r"([^A-Za-z\s{])(.{0,500}?)\1"
)
_LST_BRACE_RE = re.compile(r"\\(?:lstinline|mintinline\{[^}\n]{0,40}\}|texttt\{\\detokenize)\{([^{}\n]{0,500})\}")
_URL_RE = re.compile(r"\\(url|nolinkurl|path)\s*\{([^{}\n]{0,2000})\}")
_HREF_RE = re.compile(r"\\href\s*\{([^{}\n]{0,2000})\}")


def mask_literals(text: str, lits: Literals) -> str:
    r"""Mask verbatim environments, ``\verb`` and URL arguments (before comment stripping).

    A ``comment`` environment is removed outright. A verbatim environment without its
    ``\end`` is left untouched (it cannot be told apart from a broken source).
    """
    if "\\begin" in text and _VERB_ENV_RE.search(text):
        out, pos = [], 0
        for m in _VERB_ENV_RE.finditer(text):
            if m.start() < pos:
                continue
            env = m.group(1)
            end_tok = "\\end{" + env + "}"
            j = text.find(end_tok, m.end())
            if j < 0:
                continue
            out.append(text[pos:m.start()])
            body = text[m.end():j]
            if env in C.REMOVED_ENVS:
                out.append(" ")
            else:
                out.append(lits.block(body))
            pos = j + len(end_tok)
        out.append(text[pos:])
        text = "".join(out)
    if "\\verb" in text or "\\lstinline" in text or "\\mintinline" in text or "\\spverb" in text:
        text = _VERB_INLINE_RE.sub(lambda m: lits.inline(m.group(2)), text)
        text = _LST_BRACE_RE.sub(lambda m: lits.inline(m.group(1)), text)
    if "\\url" in text or "\\path" in text or "\\nolinkurl" in text:
        text = _URL_RE.sub(lambda m: "\\" + m.group(1) + "{" + lits.inline(m.group(2)) + "}", text)
    if "\\href" in text:
        text = _HREF_RE.sub(lambda m: "\\href{" + lits.inline(m.group(1)) + "}", text)
    return text


# ---------------------------------------------------------------------------------------
# Comments
# ---------------------------------------------------------------------------------------
# A '%' preceded by an even number (incl. 0) of backslashes starts a comment. Like TeX, the
# comment swallows the end of line and the next line's leading blanks, unless the next line
# is blank (which must survive as a paragraph break).
_COMMENT_RE = re.compile(r"(?<!\\)((?:\\\\)*+)%[^\n]*+(?:\n[ \t]*+(?![\n]))?")


def normalize_newlines(text: str) -> str:
    if "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text


def strip_comments(text: str) -> str:
    if "%" not in text:
        return text
    return _COMMENT_RE.sub(r"\1", text)


_ENDINPUT_RE = re.compile(r"\\endinput(?![A-Za-z@])")


def cut_endinput(text: str) -> str:
    r"""Everything after ``\endinput`` is ignored by TeX."""
    m = _ENDINPUT_RE.search(text)
    return text[:m.start()] if m else text


def clean_file_text(raw: str, lits: Literals) -> str:
    """Per-file preprocessing done before include resolution."""
    text = normalize_newlines(raw)
    text = mask_literals(text, lits)
    text = strip_comments(text)
    return cut_endinput(text)


# ---------------------------------------------------------------------------------------
# Conditionals
# ---------------------------------------------------------------------------------------
# Macros named \if... that are *not* TeX conditionals (no matching \fi).
_NOT_CONDITIONALS = frozenset({
    "iff", "ifthenelse", "ifboolexpr", "ifbool", "iftoggle", "ifdef", "ifndef", "ifundef",
    "ifcsdef", "ifcsundef", "ifdefstring", "ifstrequal", "ifstrempty", "ifblank",
    "ifdefempty", "ifnumcomp", "ifnumequal", "ifdimcomp", "ifdefmacro", "ifdefparam",
    "ifdefprotected", "ifcsstring", "ifoot", "ifinlist", "ifdeftok", "ifnumodd",
    "ifdefvoid", "ifcsvoid", "ifcsempty", "ifdefcounter", "ifltxcounter", "ifdeflength",
    "ifnumless", "ifnumgreater", "ifdimequal", "ifdimless", "ifdimgreater", "ifboolexe",
    "ifstrcmp", "ifpackageloaded", "ifclassloaded", "iflanguage", "ifxetexorluatex_",
    "ifmtarg", "ifpdfstringunicode", "IfFileExists", "ifcommandkey", "ifnotempty",
    "ifempty", "ifsamestring", "ifisinteger", "ifinteger", "ifdecimal", "ifthen",
    "ifEmpty", "ifnum_", "ifcase_",
})
_ENGINE_VALUES = {"ifxetex": False, "ifluatex": False, "ifpdf": True, "ifpdftex": True,
                  "ifPDFTeX": True, "ifXeTeX": False, "ifLuaTeX": False, "ifvtex": False,
                  "iftutex": False, "ifptex": False, "ifuptex": False}
_KNOWN_DEFINED = frozenset({"pdfoutput", "pdftexversion", "pdfstrcmp", "pdfshellescape",
                            "documentclass", "usepackage", "section", "begin", "end"})
_COND_TOKEN_RE = re.compile(r"\\(if[A-Za-z@]*|else|fi|or|newif|let|def|gdef|edef|xdef|"
                            r"newcommand\*?|renewcommand\*?|providecommand\*?|csname)(?![A-Za-z@])"
                            r"|[{}]|\\[{}\\]")
_NEWIF_RE = re.compile(r"\\newif\s*\\if([A-Za-z@]+)")
_SETIF_RE = re.compile(r"\\([A-Za-z@]+?)(true|false)(?![A-Za-z@])")
_LETIF_RE = re.compile(r"\\let\s*\\if([A-Za-z@]+)\s*=?\s*\\if(true|false)(?![A-Za-z@])")
_CSNAME_RE = re.compile(r"\s*\\csname\s*([A-Za-z@]+)\s*\\endcsname")
_DEFINED_ARG_RE = re.compile(r"\s*\\([A-Za-z@]+|.)")
_IFX_ARGS_RE = re.compile(r"\s*(?:\\[A-Za-z@]+|\\.|.)\s*(?:\\[A-Za-z@]+|\\.|.)")
_IFNUM_ARGS_RE = re.compile(r"\s*[-+\\\w@.\s]{1,60}?[<=>]\s*[-+]?[\\\w@.]+")


@dataclass
class CondNotes:
    undecided: int = 0
    evaluated: int = 0
    notes: list[str] = field(default_factory=list)


def evaluate_conditionals(text: str, defined: set[str] | None = None,
                          notes: CondNotes | None = None) -> str:
    r"""Evaluate ``\if..\else..\fi`` blocks at brace depth 0.

    Decidable: ``\iftrue``, ``\iffalse``, ``\newif`` switches with ``\Xtrue``/``\Xfalse``
    (last assignment before the test), ``\ifdefined`` / ``\ifcsname`` against macros defined
    in the source, and engine tests (``\ifxetex``/``\ifluatex`` false, ``\ifpdf`` true).
    Undecidable conditionals keep their first branch (a note is recorded). Conditionals
    inside braces (macro bodies) are left untouched.
    """
    if "\\if" not in text:
        return text
    notes = notes if notes is not None else CondNotes()
    defined = defined or set()
    switches: dict[str, list[tuple[int, bool]]] = {}
    for m in _NEWIF_RE.finditer(text):
        switches.setdefault(m.group(1), []).append((m.start(), False))
    if switches:
        for m in _SETIF_RE.finditer(text):
            name = m.group(1)
            if name in switches:
                switches[name].append((m.start(), m.group(2) == "true"))
        for m in _LETIF_RE.finditer(text):
            if m.group(1) in switches:
                switches[m.group(1)].append((m.start(), m.group(2) == "true"))
        for lst in switches.values():
            lst.sort()

    def switch_value(name: str, pos: int) -> bool:
        val = False
        for p, v in switches[name]:
            if p > pos:
                break
            val = v
        return val

    tokens = list(_COND_TOKEN_RE.finditer(text))
    if not tokens:
        return text
    # Find matching structure: stack of open conditionals at any depth.
    out: list[str] = []
    pos = 0
    depth = 0
    i = 0
    n = len(tokens)
    prev_decl = False  # previous token was \newif/\let/\def... -> next \ifX is a name

    def skip_branch(start_idx: int) -> tuple[int, int, str]:
        """From token index, skip until matching \\else/\\or/\\fi at nesting 0.
        Returns (token index of terminator, its position, terminator name)."""
        level = 0
        k = start_idx
        pdecl = False
        while k < n:
            t = tokens[k]
            g = t.group(1)
            if g is None:
                k += 1
                pdecl = False
                continue
            if g.startswith("if") and g not in _NOT_CONDITIONALS and not pdecl:
                level += 1
            elif g == "fi":
                if level == 0:
                    return k, t.start(), "fi"
                level -= 1
            elif g in ("else", "or") and level == 0:
                return k, t.start(), g
            pdecl = g in ("newif", "let", "def", "gdef", "edef", "xdef") or g.startswith(
                ("newcommand", "renewcommand", "providecommand"))
            k += 1
        return -1, -1, ""

    while i < n:
        t = tokens[i]
        g = t.group(1)
        tok = t.group(0)
        if g is None:
            if tok == "{":
                depth += 1
            elif tok == "}":
                depth = max(0, depth - 1)
            prev_decl = False
            i += 1
            continue
        if g in ("newif", "let", "def", "gdef", "edef", "xdef", "csname") or g.startswith(
                ("newcommand", "renewcommand", "providecommand")):
            prev_decl = True
            i += 1
            continue
        if not g.startswith("if") or g in _NOT_CONDITIONALS or prev_decl or depth > 0:
            prev_decl = False
            i += 1
            continue
        prev_decl = False
        # Decide the condition.
        cond_end = t.end()
        value: bool | None
        name = g[2:]
        if g == "iftrue":
            value = True
        elif g == "iffalse":
            value = False
        elif g in _ENGINE_VALUES:
            value = _ENGINE_VALUES[g]
        elif name in switches:
            value = switch_value(name, t.start())
        elif g == "ifdefined":
            m = _DEFINED_ARG_RE.match(text, cond_end)
            if m:
                cond_end = m.end()
                value = m.group(1) in defined or m.group(1) in _KNOWN_DEFINED
            else:
                value = None
        elif g == "ifcsname":
            m = re.compile(r"\s*([A-Za-z@]+)\s*\\endcsname").match(text, cond_end)
            if m:
                cond_end = m.end()
                value = m.group(1) in defined or m.group(1) in _KNOWN_DEFINED
            else:
                value = None
        else:
            value = None
        # Locate branches.
        k1, p1, term1 = skip_branch(i + 1)
        if k1 < 0:
            # unbalanced: leave as text
            i += 1
            continue
        if value is None:
            notes.undecided += 1
            if len(notes.notes) < 20:
                notes.notes.append(f"undecided_if:\\{g}")
            # strip simple condition arguments of \ifx / \ifnum / \ifdim
            if g == "ifx":
                m = _IFX_ARGS_RE.match(text, cond_end)
                if m:
                    cond_end = m.end()
            elif g in ("ifnum", "ifdim", "ifodd"):
                m = _IFNUM_ARGS_RE.match(text, cond_end)
                if m:
                    cond_end = m.end()
            value = True
        else:
            notes.evaluated += 1
        # find the final \fi
        if term1 == "fi":
            k_fi, p_fi = k1, p1
            else_span = None
        else:
            k_cur, p_cur = k1, p1
            # skip remaining \else / \or branches until \fi
            spans = [(p1, tokens[k1].end())]
            while True:
                k2, p2, term2 = skip_branch(k_cur + 1)
                if k2 < 0:
                    break
                if term2 == "fi":
                    k_fi, p_fi = k2, p2
                    break
                spans.append((p2, tokens[k2].end()))
                k_cur = k2
            if k2 < 0:
                i += 1
                continue
            else_span = spans
        out.append(text[pos:t.start()])
        if value:
            first_end = p1
            out.append(text[cond_end:first_end])
        else:
            if else_span:
                # second branch: from end of first \else token to next terminator
                s = else_span[0][1]
                e = else_span[1][0] if len(else_span) > 1 else p_fi
                out.append(text[s:e])
        pos = tokens[k_fi].end()
        # continue scanning *inside* the kept text is not done (nested ifs in the kept
        # branch at depth 0 are re-evaluated by a second call, see evaluate_all).
        i = k_fi + 1
    out.append(text[pos:])
    return "".join(out)


def evaluate_all(text: str, defined: set[str] | None = None, notes: CondNotes | None = None,
                 max_passes: int = 4) -> str:
    """Repeat :func:`evaluate_conditionals` so that nested conditionals are handled."""
    for _ in range(max_passes):
        new = evaluate_conditionals(text, defined, notes)
        if new == text:
            break
        text = new
    return text


# ---------------------------------------------------------------------------------------
# Brace helpers
# ---------------------------------------------------------------------------------------
def match_brace(text: str, start: int, limit: int = C.MAX_BRACE_SCAN) -> int:
    """Index just after the ``}`` matching the ``{`` at ``start``; ``-1`` if not found.

    Escaped braces (``\\{``, ``\\}``) are ignored. The search is bounded by ``limit``.
    """
    depth = 0
    i = start
    end = min(len(text), start + limit)
    while i < end:
        j1 = text.find("{", i, end)
        j2 = text.find("}", i, end)
        if j2 < 0:
            return -1
        if 0 <= j1 < j2:
            j = j1
        else:
            j = j2
        # escaped?
        bs = 0
        k = j - 1
        while k >= 0 and text[k] == "\\":
            bs += 1
            k -= 1
        if bs % 2 == 1:
            i = j + 1
            continue
        if text[j] == "{":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return j + 1
        i = j + 1
    return -1


def match_bracket(text: str, start: int, limit: int = 2000) -> int:
    """Index after the ``]`` matching ``[`` at ``start`` (braces protect ``]``); -1 if none."""
    depth = 0
    i = start + 1
    end = min(len(text), start + limit)
    while i < end:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "]" and depth <= 0:
            return i + 1
        i += 1
    return -1


_WS_RE = re.compile(r"[ \t]*\n?[ \t]*")


def skip_ws(text: str, i: int) -> int:
    """Skip spaces and at most one newline (an empty line ends argument scanning)."""
    m = _WS_RE.match(text, i)
    return m.end() if m else i


def read_arg(text: str, i: int) -> tuple[str | None, int]:
    """Read one mandatory argument at ``i``: ``{...}`` or a single token."""
    i = skip_ws(text, i)
    if i >= len(text):
        return None, i
    c = text[i]
    if c == "{":
        j = match_brace(text, i)
        if j < 0:
            return None, i
        return text[i + 1:j - 1], j
    if c == "\\":
        m = re.compile(r"\\(?:[A-Za-z@]+|.)").match(text, i)
        if m:
            return m.group(0), m.end()
        return None, i
    if c in "}%":
        return None, i
    return c, i + 1


def read_opt(text: str, i: int) -> tuple[str | None, int]:
    """Read an optional ``[...]`` argument at ``i`` (after blanks)."""
    j = skip_ws(text, i)
    if j < len(text) and text[j] == "[":
        k = match_bracket(text, j)
        if k > 0:
            return text[j + 1:k - 1], k
    return None, i


# ---------------------------------------------------------------------------------------
# Macro definitions and expansion
# ---------------------------------------------------------------------------------------
# Names whose redefinition must never be expanded (structure, citations, core commands).
_PROTECTED = frozenset(
    set(C.SECTION_RANKS) | {s + "*" for s in C.SECTION_RANKS} | set(C.CITE_COMMANDS)
    | set(C.REF_COMMANDS) | {
        "begin", "end", "item", "caption", "label", "footnote", "emph", "textbf", "textit",
        "includegraphics", "url", "href", "title", "author", "maketitle", "bibitem",
        "bibliography", "input", "include", "documentclass", "usepackage", "appendix",
        "abstract", "newblock", "bibinfo", "thanks", "and", "date", "today", "par",
        "newcommand", "renewcommand", "def", "let", "section", "chapter", "paragraph",
        "frac", "left", "right", "text", "mathrm", "mathbf", "mathcal", "mathbb",
        "thebibliography", "bibfield", "verb", "item", "hline", "caption", "subcaption",
        "baselinestretch", "arraystretch", "thesection", "thesubsection", "thetable",
        "thefigure", "theequation", "thefootnote", "thepage", "refname", "bibname",
        "abstractname", "figurename", "tablename", "appendixname", "contentsname",
        "acknowledgments", "acknowledgements", "IEEEPARstart", "cite", "ref", "eqref",
        "newline", "linebreak", "familydefault", "rmdefault", "sfdefault", "ttdefault",
        "footnotesize", "small", "large", "normalsize", "UrlFont", "labelenumi",
        "labelitemi", "labelitemii", "theenumi", "algorithmicrequire", "algorithmicensure",
        "headrulewidth", "footrulewidth", "topfraction", "bottomfraction", "textfraction",
        "floatpagefraction", "dbltopfraction", "dblfloatpagefraction", "keywords",
    })
_NEWCMD_RE = re.compile(
    r"\\(?:new|renew|provide)command\*?\s*(?:\{\s*\\([A-Za-z@]+|.)\s*\}|\\([A-Za-z@]+|.))"
    r"|\\DeclareRobustCommand\*?\s*(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))"
    r"|\\(?:g|e|x)?def\s*\\([A-Za-z@]+)"
    r"|\\let\s*\\([A-Za-z@]+)\s*=?\s*(\\[A-Za-z@]+|\\.)"
    r"|\\(?:new|renew)environment\*?\s*\{([A-Za-z@*]+)\}"
    r"|\\DeclareMathOperator(\*?)\s*(?:\{\s*\\([A-Za-z@]+)\s*\}|\\([A-Za-z@]+))"
    r"|\\(?:new|renew|provide)expandafter_never(x)"
)
_DEF_PARAMS_RE = re.compile(r"((?:#\d)*)\s*\{")


@dataclass
class Macro:
    name: str
    nargs: int = 0
    default: str | None = None
    body: str = ""


@dataclass
class Env:
    name: str
    nargs: int = 0
    default: str | None = None
    begin: str = ""
    end: str = ""


@dataclass
class MacroTable:
    macros: dict[str, Macro] = field(default_factory=dict)
    envs: dict[str, Env] = field(default_factory=dict)
    defined: set[str] = field(default_factory=set)
    skipped: int = 0
    expansions: int = 0
    truncated: bool = False


def _body_ok(body: str) -> bool:
    # Bodies that define macros, use parameters of parameters, @-internals or TeX
    # conditionals are not "simple" and are left alone.
    if "##" in body or "@" in body or "\\csname" in body or "\\expandafter" in body:
        return False
    if re.search(r"\\(?:new|renew|provide)command|\\(?:g|e|x)?def(?![A-Za-z])|\\if[a-zA-Z]*|\\fi(?![A-Za-z])",
                 body):
        return False
    return True


def collect_definitions(text: str, table: MacroTable | None = None) -> tuple[str, MacroTable]:
    r"""Parse author macro definitions and remove them from ``text``.

    Handles ``\newcommand``/``\renewcommand``/``\providecommand``/``\DeclareRobustCommand``
    (0–9 args, optional default), ``\def`` with ``#1#2..`` parameters, ``\let\a\b``,
    ``\newenvironment`` and ``\DeclareMathOperator``. Unsupported definitions are removed
    from the text as well (they would only leak as garbage) but not expanded.
    """
    table = table or MacroTable()
    if "\\" not in text:
        return text, table
    out: list[str] = []
    pos = 0
    for m in _NEWCMD_RE.finditer(text):
        if m.start() < pos:
            continue
        g = m.groups()
        start = m.start()
        i = m.end()
        try:
            if g[0] is not None or g[1] is not None or g[2] is not None or g[3] is not None:
                name = g[0] or g[1] or g[2] or g[3]
                nargs_s, i2 = read_opt(text, i)
                nargs = 0
                if nargs_s is not None:
                    try:
                        nargs = int(nargs_s.strip())
                    except ValueError:
                        nargs = -1
                    i = i2
                default, i3 = read_opt(text, i)
                if default is not None:
                    i = i3
                body, i4 = read_arg(text, i)
                if body is None or nargs < 0 or nargs > 9:
                    continue
                i = i4
                table.defined.add(name)
                if (name not in _PROTECTED and _body_ok(body)
                        and not (text.startswith("\\provide", start) and name in table.macros)):
                    table.macros[name] = Macro(name, nargs, default, body)
                else:
                    table.skipped += 1
            elif g[4] is not None:  # \def
                name = g[4]
                pm = _DEF_PARAMS_RE.match(text, i)
                if not pm:
                    table.defined.add(name)
                    table.skipped += 1
                    continue
                params = pm.group(1)
                nargs = len(params) // 2
                if params and params != "".join(f"#{k}" for k in range(1, nargs + 1)):
                    table.skipped += 1
                    continue
                brace = pm.end() - 1
                j = match_brace(text, brace)
                if j < 0:
                    continue
                body = text[brace + 1:j - 1]
                i = j
                table.defined.add(name)
                if name not in _PROTECTED and _body_ok(body):
                    table.macros[name] = Macro(name, nargs, None, body)
                else:
                    table.skipped += 1
            elif g[5] is not None:  # \let\a\b
                name, target = g[5], g[6]
                table.defined.add(name)
                tname = target[1:]
                if name in _PROTECTED or name.startswith("if"):
                    table.skipped += 1
                elif tname in table.macros:
                    src = table.macros[tname]
                    table.macros[name] = Macro(name, src.nargs, src.default, src.body)
                elif tname in ("relax", "empty", "undefined"):
                    table.macros[name] = Macro(name, 0, None, "")
                else:
                    table.macros[name] = Macro(name, 0, None, target)
            elif g[7] is not None:  # \newenvironment{name}[n][default]{begin}{end}
                name = g[7]
                nargs_s, i2 = read_opt(text, i)
                nargs = 0
                if nargs_s is not None:
                    try:
                        nargs = int(nargs_s.strip())
                    except ValueError:
                        nargs = -1
                    i = i2
                default, i3 = read_opt(text, i)
                if default is not None:
                    i = i3
                b, i4 = read_arg(text, i)
                e, i5 = read_arg(text, i4) if b is not None else (None, i4)
                if b is None or e is None or nargs < 0:
                    continue
                i = i5
                if (name not in C.ABSTRACT_ENVS and name not in ("document", "thebibliography")
                        and _body_ok(b) and _body_ok(e) and "#" not in e):
                    table.envs[name] = Env(name, nargs, default, b, e)
                else:
                    table.skipped += 1
            elif g[9] is not None or g[10] is not None:  # \DeclareMathOperator
                name = g[9] or g[10]
                body, i2 = read_arg(text, i)
                if body is None:
                    continue
                i = i2
                table.defined.add(name)
                table.macros[name] = Macro(name, 0, None,
                                           "\\operatorname" + (g[8] or "") + "{" + body + "}")
            else:
                continue
        except (IndexError, ValueError):
            continue
        out.append(text[pos:start])
        pos = i
    out.append(text[pos:])
    return "".join(out), table


def _substitute(body: str, args: list[str]) -> str:
    if "#" not in body:
        return body
    return re.sub(r"#([1-9])", lambda m: args[int(m.group(1)) - 1]
                  if int(m.group(1)) <= len(args) else "", body)


def expand_macros(text: str, table: MacroTable, max_passes: int = C.MAX_MACRO_PASSES) -> str:
    """Expand author macros and environments, bounded in passes and growth."""
    if not table.macros and not table.envs:
        return text
    limit = int(len(text) * C.MAX_MACRO_GROWTH) + C.MAX_MACRO_GROWTH_ABS
    names = sorted(table.macros, key=len, reverse=True)
    mac_re = None
    if names:
        letters = [re.escape(n) for n in names if n[:1].isalpha() or n[:1] == "@"]
        others = [re.escape(n) for n in names if not (n[:1].isalpha() or n[:1] == "@")]
        parts = []
        if letters:
            parts.append(r"\\(" + "|".join(letters) + r")(?![A-Za-z@])")
        if others:
            parts.append(r"\\(" + "|".join(others) + r")")
        mac_re = re.compile("|".join(parts))
    env_re = None
    if table.envs:
        env_re = re.compile(r"\\(begin|end)\s*\{(" + "|".join(
            re.escape(e) for e in sorted(table.envs, key=len, reverse=True)) + r")\}")

    for _ in range(max_passes):
        changed = False
        if mac_re is not None:
            out: list[str] = []
            pos = 0
            for m in mac_re.finditer(text):
                if m.start() < pos:
                    continue
                name = m.group(1) or (m.group(2) if m.lastindex and m.lastindex >= 2 else None)
                mac = table.macros.get(name)
                if mac is None:
                    continue
                i = m.end()
                args: list[str] = []
                ok = True
                if mac.nargs:
                    k = 0
                    if mac.default is not None:
                        o, i2 = read_opt(text, i)
                        if o is not None:
                            args.append(o)
                            i = i2
                        else:
                            args.append(mac.default)
                        k = 1
                    for _a in range(k, mac.nargs):
                        a, i2 = read_arg(text, i)
                        if a is None:
                            ok = False
                            break
                        args.append(a)
                        i = i2
                if not ok:
                    continue
                body = _substitute(mac.body, args)
                # A control word followed by a letter needs a separator (\foo -> \alpha x).
                if not mac.nargs and i < len(text) and text[i:i + 1].isalpha() and re.search(
                        r"\\[A-Za-z@]+$", body):
                    body += " "
                out.append(text[pos:m.start()])
                out.append(body)
                pos = i
                table.expansions += 1
                changed = True
            if changed:
                out.append(text[pos:])
                text = "".join(out)
        if env_re is not None and env_re.search(text):
            out = []
            pos = 0
            for m in env_re.finditer(text):
                if m.start() < pos:
                    continue
                env = table.envs[m.group(2)]
                i = m.end()
                if m.group(1) == "begin":
                    args = []
                    ok = True
                    k = 0
                    if env.nargs and env.default is not None:
                        o, i2 = read_opt(text, i)
                        args.append(o if o is not None else env.default)
                        if o is not None:
                            i = i2
                        k = 1
                    for _a in range(k, env.nargs):
                        a, i2 = read_arg(text, i)
                        if a is None:
                            ok = False
                            break
                        args.append(a)
                        i = i2
                    if not ok:
                        continue
                    rep = " " + _substitute(env.begin, args) + " "
                else:
                    rep = " " + env.end + " "
                out.append(text[pos:m.start()])
                out.append(rep)
                pos = i
                table.expansions += 1
                changed = True
            out.append(text[pos:])
            text = "".join(out)
        if len(text) > limit:
            table.truncated = True
            break
        if not changed:
            break
    return text


# ---------------------------------------------------------------------------------------
# Theorem names
# ---------------------------------------------------------------------------------------
_NEWTHEOREM_RE = re.compile(
    r"\\(?:new|spnew|declare)theorem\*?\s*(?:\[[^\]]{0,200}\]\s*)?\{([A-Za-z@*]+)\}"
    r"|\\newtheoremstyle_never(x)"
    r"|\\declaretheorem\s*(?:\[[^\]]{0,400}\]\s*)?\{([A-Za-z@*,\s]+)\}"
    r"|\\newmdtheoremenv\s*(?:\[[^\]]{0,200}\]\s*)?\{([A-Za-z@*]+)\}"
    r"|\\newtcbtheorem\s*(?:\[[^\]]{0,200}\]\s*)?\{([A-Za-z@*]+)\}"
    r"|\\(?:newtcolorbox|newmdenv|newframedtheorem)\s*(?:\[[^\]]{0,200}\]\s*)?\{([A-Za-z@*]+)\}"
)
_TITLE_TAG_RE = re.compile(r"\\(?:new|spnew)theorem\*?\s*\{([A-Za-z@*]+)\}\s*(?:\[[^\]]*\]\s*)?\{([^{}]{0,80})\}")


def collect_theorem_names(text: str) -> tuple[set[str], dict[str, str]]:
    names: set[str] = set()
    for m in _NEWTHEOREM_RE.finditer(text):
        for g in m.groups():
            if g:
                for n in g.split(","):
                    n = n.strip()
                    if n:
                        names.add(n)
    titles = {m.group(1): m.group(2).strip() for m in _TITLE_TAG_RE.finditer(text)}
    return names, titles
