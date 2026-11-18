r"""LaTeX cleanup, text normalization, and ``\cite`` / ``\ref`` normalization.

:func:`clean_with_footnotes` turns a raw LaTeX prose fragment into plain text where

* non-semantic formatting is removed (``\centering``, sizes, spacing, rules, labels ...);
* content-bearing commands keep their text (``\textbf{x}`` -> ``x``, ``\href{u}{t}`` -> ``t``);
  an unknown command keeps its last brace argument, or disappears when it has none;
* accents and special characters become Unicode (NFC), ``~`` a space, ``--``/``---`` dashes;
* every citation variant becomes ``\cite{k1,k2}`` and every reference ``\ref{label}``;
* inline math becomes ``$...$`` (normalized by :mod:`mathnorm`);
* footnotes are cut out and returned separately; a marker ``\uE005<n>\uE006`` is left at the
  footnote position so that the caller can place the footnote sentences after their host.

The function is a single left-to-right scan driven by one pre-compiled regex; arguments are
read with bounded brace matching, so malformed input never raises or hangs.
"""

from __future__ import annotations

import re
import unicodedata

from . import config as C
from .mathnorm import normalize_inline
from .preprocess import (BL_CLOSE, BL_OPEN, IL_OPEN, Literals, match_brace, match_bracket,
                         read_arg)

FN_OPEN, FN_CLOSE = "\ue005", "\ue006"
FN_MARK_RE = re.compile("\ue005(\\d+)\ue006")

_TOK_RE = re.compile(
    r"\\(?P<cmd>[A-Za-z@]+)(?P<star>\*?)"
    r"|\\(?P<sym>[^A-Za-z@])"
    r"|(?P<ch>[{}$~&])"
    r"|(?P<dash>-{2,3})"
    r"|(?P<q>``|'')"
    r"|(?P<ph>\ue000\d+\ue001|\ue002\d+\ue003)"
)
_WS_RE = re.compile(r"\s+")
_SPACE_PUNCT_RE = re.compile(r" +([,.;:!?)\]])")
_OPEN_SPACE_RE = re.compile(r"([(\[]) +")

# Number of brace arguments of content-bearing commands; the last one is kept.
KEEP_ARITY = {
    "textcolor": 2, "colorbox": 2, "fcolorbox": 3, "href": 2, "resizebox": 3, "scalebox": 2,
    "rotatebox": 2, "multicolumn": 3, "multirow": 3, "parbox": 2, "raisebox": 2, "adjustbox": 2,
    "foreignlanguage": 2, "makebox": 1, "framebox": 1, "subfloat": 1, "IEEEPARstart": 2,
    "texorpdfstring": 2, "hyperlink": 2, "hyperref": 1, "Cline": 1, "shortstack": 1,
}
# texorpdfstring keeps its *first* argument (the TeX string), IEEEPARstart concatenates.
_KEEP_FIRST = {"texorpdfstring"}
_LETTER_ACCENTS = {"u", "v", "H", "c", "k", "d", "b", "r", "t"}
_URL_CMDS = {"url", "nolinkurl", "path"}
_SIUNITX = {"SI": 2, "qty": 2, "num": 1, "si": 1, "unit": 1, "ang": 1, "SIrange": 3,
            "qtyrange": 3, "numrange": 2, "numlist": 1, "qtylist": 2}
_ACRO = {"ac", "acs", "acl", "acf", "acp", "acsp", "aclp", "acfp", "Ac", "Acs", "Acl", "Acf",
         "Acp", "gls", "Gls", "glspl", "Glspl", "acrshort", "acrlong", "acrfull", "glsentrytext",
         "acro", "acrodef_x", "glssymbol", "GLS", "acused"}
_MATH_ENVS_INLINE = {"math"}
_CITE = C.CITE_COMMANDS
_REF = C.REF_COMMANDS


class _Ctx:
    __slots__ = ("lits", "footnotes", "keep_footnotes", "depth")

    def __init__(self, lits: Literals | None, keep_footnotes: bool):
        self.lits = lits
        self.footnotes: list[str] = []
        self.keep_footnotes = keep_footnotes
        self.depth = 0


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \t\n":
        i += 1
    return i


def _read_opts(text: str, i: int, max_n: int = 3) -> tuple[list[str], int]:
    opts = []
    for _ in range(max_n):
        j = _skip_ws(text, i)
        if j < len(text) and text[j] == "[":
            k = match_bracket(text, j)
            if k < 0:
                break
            opts.append(text[j + 1:k - 1])
            i = k
        else:
            break
    return opts, i


def _read_braces(text: str, i: int, n: int, adjacent: bool = False) -> tuple[list[str], int]:
    """Read up to ``n`` brace groups. ``adjacent``: only groups that follow without a blank line."""
    out = []
    for _ in range(n):
        j = i
        if adjacent:
            while j < len(text) and text[j] in " \t":
                j += 1
            if j < len(text) and text[j] == "\n":
                j += 1
                while j < len(text) and text[j] in " \t":
                    j += 1
                if j < len(text) and text[j] == "\n":
                    break
        else:
            j = _skip_ws(text, j)
        if j < len(text) and text[j] == "{":
            k = match_brace(text, j)
            if k < 0:
                break
            out.append(text[j + 1:k - 1])
            i = k
        else:
            break
    return out, i


def _find_math_end(text: str, i: int, delim: str) -> int:
    """Index of the closing delimiter of inline math starting after ``i`` (-1 if none)."""
    n = len(text)
    j = i
    limit = min(n, i + 20000)
    while j < limit:
        k = text.find(delim, j, limit)
        if k < 0:
            return -1
        # escaped?
        bs = 0
        p = k - 1
        while p >= i and text[p] == "\\":
            bs += 1
            p -= 1
        if bs % 2 == 0:
            # a blank line ends inline math (TeX error); stop there
            if "\n\n" in text[i:k] and re.search(r"\n[ \t]*\n", text[i:k]):
                return -1
            return k
        j = k + 1
    return -1


def _keys(s: str) -> list[str]:
    out = []
    for k in s.split(","):
        k = k.strip()
        if k and k not in out:
            out.append(k)
    return out


def _accent(base: str, acc: str) -> str:
    base = base.strip()
    if not base:
        return unicodedata.normalize("NFC", "\u00a0" + acc).strip()
    if base[0] == "ı":
        base = "i" + base[1:]
    elif base[0] == "ȷ":
        base = "j" + base[1:]
    return unicodedata.normalize("NFC", base[0] + acc) + base[1:]


def _clean(text: str, ctx: _Ctx) -> str:
    ctx.depth += 1
    if ctx.depth > 60:
        ctx.depth -= 1
        return re.sub(r"\\[A-Za-z@]+|[{}]", " ", text)
    out: list[str] = []
    i = 0
    n = len(text)
    search = _TOK_RE.search
    while i < n:
        m = search(text, i)
        if not m:
            out.append(text[i:])
            break
        if m.start() > i:
            out.append(text[i:m.start()])
        i = m.end()
        cmd = m.group("cmd")
        if cmd is not None:
            star = m.group("star")
            # ---- citations / references
            if cmd in _CITE:
                _o, i = _read_opts(text, i, 2)
                groups, i = _read_braces(text, i, 8 if cmd.endswith("s") else 1)
                if cmd.endswith("s") and cmd not in ("cites_",):
                    # biblatex multicite: \cites(pre)(post)[..]{a}[..]{b}
                    while True:
                        _o2, j = _read_opts(text, i, 2)
                        more, j = _read_braces(text, j, 1, adjacent=True)
                        if not more:
                            break
                        groups += more
                        i = j
                keys = []
                for g in groups:
                    keys += [k for k in _keys(g) if k not in keys]
                if keys:
                    out.append("\\cite{" + ",".join(keys) + "}")
                continue
            if cmd in _REF:
                _o, i = _read_opts(text, i, 1)
                g, i = _read_braces(text, i, 2 if cmd in ("crefrange", "Crefrange") else 1)
                labels = []
                for x in g:
                    labels += _keys(x)
                if labels:
                    out.append("\\ref{" + ",".join(labels) + "}")
                continue
            # ---- footnotes
            if cmd in ("footnote", "footnotetext", "endnote", "footnotefoot"):
                _o, i = _read_opts(text, i, 1)
                g, i = _read_braces(text, i, 1)
                if g and ctx.keep_footnotes:
                    sub = _Ctx(ctx.lits, False)
                    sub.depth = ctx.depth
                    ft = _finish(_clean(g[0], sub))
                    if ft:
                        ctx.footnotes.append(ft)
                        out.append(f"{FN_OPEN}{len(ctx.footnotes) - 1}{FN_CLOSE}")
                continue
            # ---- accents written as letters: \v{c}, \c c, \H{o}
            if cmd in _LETTER_ACCENTS and len(cmd) == 1:
                j = i
                if j < n and text[j] == "{":
                    k = match_brace(text, j)
                    if k > 0:
                        out.append(_accent(_clean(text[j + 1:k - 1], ctx), C.ACCENTS[cmd]))
                        i = k
                        continue
                j = _skip_ws(text, j)
                if j < n and text[j].isalpha():
                    out.append(_accent(text[j], C.ACCENTS[cmd]))
                    i = j + 1
                    continue
                if j < n and text[j] == "\\":
                    a, k = read_arg(text, j)
                    if a:
                        out.append(_accent(_clean(a, ctx), C.ACCENTS[cmd]))
                        i = k
                        continue
                continue
            # ---- URLs: literal argument
            if cmd in _URL_CMDS:
                g, i = _read_braces(text, i, 1)
                if g:
                    u = g[0]
                    if ctx.lits is not None:
                        u = ctx.lits.restore_inline(u)
                    out.append(u.strip())
                continue
            if cmd == "IEEEPARstart":
                g, i = _read_braces(text, i, 2)
                out.append("".join(_clean(x, ctx) for x in g))
                continue
            if cmd in _SIUNITX:
                _o, i = _read_opts(text, i, 1)
                g, i = _read_braces(text, i, _SIUNITX[cmd])
                out.append(" ".join(_clean(x, ctx) for x in g))
                continue
            if cmd in _ACRO:
                _o, i = _read_opts(text, i, 1)
                g, i = _read_braces(text, i, 1)
                if g:
                    out.append(_clean(g[0], ctx))
                continue
            if cmd == "ensuremath":
                g, i = _read_braces(text, i, 1)
                if g:
                    out.append(normalize_inline(g[0]))
                continue
            # ---- environments inside prose (captions, cells, ...)
            if cmd in ("begin", "end"):
                g, j = _read_braces(text, i, 1)
                env = g[0].strip() if g else ""
                i = j
                if cmd == "begin":
                    if env in _MATH_ENVS_INLINE or env in C.BLOCK_MATH_ENVS:
                        endtok = "\\end{" + env + "}"
                        k = text.find(endtok, i)
                        if k >= 0:
                            body = text[i:k]
                            if env in C.MULTILINE_MATH_ENVS:
                                body = body.replace("\\\\", " ").replace("&", " ")
                            out.append(" " + normalize_inline(body) + " ")
                            i = k + len(endtok)
                        continue
                    if env in C.TABULAR_ENVS:
                        _o, i = _read_opts(text, i, 1)
                        _g, i = _read_braces(text, i, 2 if env in ("tabular*", "tabularx", "tabulary") else 1)
                    else:
                        _o, i = _read_opts(text, i, 1)
                out.append(" ")
                continue
            if cmd == "item":
                _o, i = _read_opts(text, i, 1)
                if _o:
                    out.append(" " + _clean(_o[0], ctx) + " ")
                else:
                    out.append(" ")
                continue
            if cmd == "caption" or cmd == "captionof" or cmd == "subcaption":
                _o, i = _read_opts(text, i, 1)
                g, i = _read_braces(text, i, 2 if cmd == "captionof" else 1)
                if g:
                    out.append(" " + _clean(g[-1], ctx) + " ")
                continue
            # ---- symbols
            sym = C.TEXT_SYMBOLS.get(cmd)
            if sym is not None and cmd not in C.KEEP_CONTENT:
                out.append(sym)
                if i < n and text[i] == "{" and text[i + 1:i + 2] == "}":
                    i += 2  # \ss{}, \LaTeX{}
                elif sym and sym[-1].isalpha() and i < n and text[i] == " ":
                    pass
                continue
            # ---- content-bearing commands
            if cmd in C.KEEP_CONTENT or cmd in KEEP_ARITY:
                ar = KEEP_ARITY.get(cmd, 1)
                _o, i = _read_opts(text, i, 2)
                g, i = _read_braces(text, i, ar)
                if cmd == "subfloat" and _o and not g:
                    out.append(_clean(_o[0], ctx))
                elif g:
                    keep = g[0] if cmd in _KEEP_FIRST else g[-1]
                    out.append(_clean(keep, ctx))
                continue
            if cmd in C.DROP_ARG_IN_PROSE:
                _o, i = _read_opts(text, i, 1)
                _g, i = _read_braces(text, i, 1)
                continue
            na = C.DROP_WITH_ARGS.get(cmd)
            if na is not None:
                _o, i = _read_opts(text, i, 2)
                if na:
                    _g, i = _read_braces(text, i, na)
                continue
            if cmd in C.DROP_SWITCHES:
                # switches may be followed by a dimension, e.g. \vskip 2pt
                continue
            # ---- unknown command: keep the text of its last argument
            _o, j = _read_opts(text, i, 2) if i < n and text[i] in "[ " else ([], i)
            g, j2 = _read_braces(text, j, 6, adjacent=True)
            if g:
                out.append(_clean(g[-1], ctx))
                i = j2
            elif _o and j > i and text[i] == "[":
                i = j  # \foo[opt] without braces: drop the option too
            continue
        sym = m.group("sym")
        if sym is not None:
            if sym in C.ACCENTS and sym not in "~" or sym == "~":
                acc = C.ACCENTS.get(sym)
                if acc is not None:
                    # \'e  \'{e}  \'\i
                    if i < n and text[i] == "{":
                        k = match_brace(text, i)
                        if k > 0:
                            out.append(_accent(_clean(text[i + 1:k - 1], ctx), acc))
                            i = k
                            continue
                    if i < n and text[i] == "\\":
                        a, k = read_arg(text, i)
                        if a:
                            out.append(_accent(_clean(a, ctx), acc))
                            i = k
                            continue
                    if i < n and not text[i].isspace():
                        out.append(_accent(text[i], acc))
                        i += 1
                        continue
                    continue
            if sym == "\\":
                # line break, optional [len] or *
                if i < n and text[i] == "*":
                    i += 1
                if i < n and text[i] == "[":
                    k = match_bracket(text, i, 50)
                    if k > 0:
                        i = k
                out.append(" ")
                continue
            if sym == "(":
                k = _find_math_end(text, i, "\\)")
                if k >= 0:
                    out.append(normalize_inline(text[i:k]))
                    i = k + 2
                continue
            if sym == "[":
                k = _find_math_end(text, i, "\\]")
                if k >= 0:
                    out.append(" " + normalize_inline(text[i:k]) + " ")
                    i = k + 2
                continue
            if sym in (")", "]"):
                continue
            if sym in "%&_#${}":
                out.append(sym)
                continue
            if sym in " \n\t,;:!>":
                out.append(" ")
                continue
            if sym in "-/@":
                continue
            out.append(C.TEXT_SYMBOLS.get(sym, ""))
            continue
        ch = m.group("ch")
        if ch is not None:
            if ch == "$":
                if i < n and text[i] == "$":
                    k = text.find("$$", i + 1)
                    if k >= 0:
                        out.append(" " + normalize_inline(text[i + 1:k]) + " ")
                        i = k + 2
                    else:
                        i += 1
                    continue
                k = _find_math_end(text, i, "$")
                if k >= 0:
                    out.append(normalize_inline(text[i:k]))
                    i = k + 1
                continue
            if ch == "~":
                out.append(" ")
            elif ch == "&":
                out.append(" ")
            # braces are dropped
            continue
        d = m.group("dash")
        if d is not None:
            out.append("—" if len(d) == 3 else "–")
            continue
        if m.group("q") is not None:
            out.append('"')
            continue
        ph = m.group("ph")
        if ph is not None:
            if ph[0] == IL_OPEN and ctx.lits is not None:
                out.append(ctx.lits.restore_inline(ph))
            elif ph[0] == BL_OPEN:
                out.append(" ")
            continue
    ctx.depth -= 1
    return "".join(out)


def _finish(s: str) -> str:
    if "\n" in s or "\t" in s or "  " in s or "\u00a0" in s:
        s = _WS_RE.sub(" ", s)
    s = unicodedata.normalize("NFC", s.strip())
    if " " in s:
        s = _SPACE_PUNCT_RE.sub(r"\1", s)
        s = _OPEN_SPACE_RE.sub(r"\1", s)
    return s


def clean_with_footnotes(raw: str, lits: Literals | None = None) -> tuple[str, list[str]]:
    """Clean a prose fragment. Returns ``(text_with_footnote_markers, footnotes)``."""
    ctx = _Ctx(lits, True)
    return _finish(_clean(raw, ctx)), ctx.footnotes


def clean_inline(raw: str, lits: Literals | None = None) -> str:
    """Clean a fragment and drop footnotes (titles, captions, cells)."""
    ctx = _Ctx(lits, False)
    return _finish(_clean(raw, ctx))


def strip_markers(text: str) -> str:
    return _finish(FN_MARK_RE.sub("", text)) if FN_OPEN in text else text


_RESIDUAL_RE = re.compile(r"\\[a-zA-Z]+")
_MATH_SPAN_RE = re.compile(r"\$[^$]*\$")
_CITEREF_RE = re.compile(r"\\(?:cite|ref)\{[^}]*\}")


def has_residual_latex(sentence: str) -> bool:
    """True when a sentence still contains a control word outside ``$..$``, ``\\cite``, ``\\ref``."""
    if "\\" not in sentence:
        return False
    s = _MATH_SPAN_RE.sub(" ", sentence)
    s = _CITEREF_RE.sub(" ", s)
    return bool(_RESIDUAL_RE.search(s))
