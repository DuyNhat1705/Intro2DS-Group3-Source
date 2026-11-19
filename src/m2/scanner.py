r"""Hand-written, single-pass, error-tolerant LaTeX tokenizer with manual brace matching.
It never raises and never hangs.

The scanner exposes one pre-compiled regex (:data:`STRUCT_RE`) that finds the *structural*
tokens of a body (sectioning commands, ``\begin``/``\end``, ``\item``, math delimiters,
paragraph breaks and a few class-specific commands) plus helpers that read arguments and
find matching ``\end{env}`` with bounded searches. Everything between two tokens is prose and
is handed to :mod:`clean` by the tree builder (:mod:`hierarchy`).

No pattern can backtrack catastrophically: alternatives are literal prefixes, quantifiers are
bounded or possessive, and nested structures are matched by explicit counting loops.
"""

from __future__ import annotations

import re
from functools import lru_cache

from . import config as C
from .preprocess import match_brace, match_bracket

STRUCT_RE = re.compile(
    r"\\(?P<sec>part|chapter|section|subsection|subsubsection|paragraph|subparagraph)"
    r"(?P<sstar>\*?)(?![A-Za-z@])"
    r"|\\(?P<be>begin|end)[ \t]*+\{(?P<env>[^{}\s]{1,80})\}"
    r"|\\(?P<item>item)(?![A-Za-z@])"
    r"|(?P<dd>\$\$)"
    r"|(?P<d>\$)"
    r"|\\(?P<mopen>[(\[])"
    r"|\\(?P<misc>appendix|bibliography|printbibliography|nobibliography|abstract|acknowledgments|"
    r"acknowledgements|maketitle|twocolumn|IEEEraisesectionheading|bibitem|keywords|"
    r"IEEEkeywords|title|author|date)(?![A-Za-z@])"
    r"|(?P<par>\n[ \t]*+\n)"
)


def escaped(text: str, pos: int) -> bool:
    """True when the character at ``pos`` is preceded by an odd number of backslashes."""
    k = pos - 1
    n = 0
    while k >= 0 and text[k] == "\\":
        n += 1
        k -= 1
    return n % 2 == 1


@lru_cache(maxsize=512)
def _env_re(name: str) -> re.Pattern:
    e = re.escape(name)
    return re.compile(r"\\(begin|end)[ \t]*\{" + e + r"\}")


def find_env_end(text: str, start: int, name: str, limit: int = C.MAX_ENV_SCAN) -> tuple[int, int]:
    """Find the ``\\end{name}`` matching a ``\\begin{name}`` whose token ends at ``start``.

    Returns ``(body_end, after_end)``; ``(-1, -1)`` when not found within ``limit`` chars.
    """
    depth = 1
    stop = min(len(text), start + limit)
    for m in _env_re(name).finditer(text, start, stop):
        if m.group(1) == "begin":
            depth += 1
        else:
            depth -= 1
            if depth == 0:
                return m.start(), m.end()
    return -1, -1


def _brace_depth(text: str, i: int, k: int) -> int:
    """Brace depth at ``k`` relative to ``i`` (escaped braces ignored, never below 0)."""
    depth = 0
    j = i
    while j < k:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}" and depth:
            depth -= 1
        j += 1
    return depth


def dollar_at_top(text: str, i: int, k: int, stop: int) -> int:
    """Given the first unescaped ``$`` at ``k`` after an opening ``$`` at ``i - 1``, return the
    closing ``$`` at brace depth 0: ``$\\mbox{if $a\\cdot b$}$`` closes at the last ``$``, not
    at the one opening the text-mode math inside the box. ``k`` is kept when the braces before
    it are balanced (the common case) or when no depth-0 ``$`` exists (unbalanced source)."""
    if "{" not in text[i:k] or _brace_depth(text, i, k) == 0:
        return k
    depth = 0
    j = i
    while j < stop:
        c = text[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            if depth == 0:
                return k
            depth -= 1
        elif c == "$" and depth == 0:
            return j
        j += 1
    return k


def find_inline_math_end(text: str, i: int, delim: str, limit: int = 8000) -> int:
    """Closing delimiter of inline math opened before ``i``; -1 if none before a blank line."""
    stop = min(len(text), i + limit)
    j = i
    while j < stop:
        k = text.find(delim, j, stop)
        if k < 0:
            return -1
        if not escaped(text, k):
            if delim == "$":
                k = dollar_at_top(text, i, k, stop)
            seg = text[i:k]
            if "\n" in seg and re.search(r"\n[ \t]*\n", seg):
                return -1
            return k
        j = k + 1
    return -1


def find_display_end(text: str, i: int, delim: str, limit: int = 60000) -> int:
    stop = min(len(text), i + limit)
    j = i
    while j < stop:
        k = text.find(delim, j, stop)
        if k < 0:
            return -1
        if not escaped(text, k):
            return k
        j = k + 1
    return -1


_WS = " \t\n"


def skip_blank(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in _WS:
        i += 1
    return i


def read_optional(text: str, i: int) -> tuple[str | None, int]:
    j = skip_blank(text, i)
    if j < len(text) and text[j] == "[":
        k = match_bracket(text, j)
        if k > 0:
            return text[j + 1:k - 1], k
    return None, i


def read_group(text: str, i: int) -> tuple[str | None, int, bool]:
    """Read ``{...}`` after blanks. Returns ``(content, end, ok)``; ``ok`` is False when the
    brace never closes (the caller recovers and marks the version ``partial``)."""
    j = skip_blank(text, i)
    if j < len(text) and text[j] == "{":
        k = match_brace(text, j)
        if k > 0:
            return text[j + 1:k - 1], k, True
        return None, j, False
    return None, i, True


# Environment arguments that must be skipped so they do not leak into prose.
ENV_ARGS = {
    "minipage": 1, "multicols": 1, "multicols*": 1, "spacing": 1, "otherlanguage": 1,
    "otherlanguage*": 1, "adjustbox": 1, "CJK": 2, "CJK*": 2, "list": 2, "wrapfigure": 2,
    "wraptable": 2, "tabularx": 2, "tabular*": 2, "tabulary": 2, "tabular": 1, "array": 1,
    "longtable": 1, "subfigure": 1, "subtable": 1, "tcolorbox": 0, "mdframed": 0,
    "addmargin": 1, "addmargin*": 1, "turn": 1, "rotate": 1, "sidewaysfigure": 0,
    "selectlanguage": 1, "parcolumns": 1, "column": 1, "columns": 0, "block": 1,
    "exampleblock": 1, "alertblock": 1, "textblock": 1, "textblock*": 1, "overpic": 1,
    "appendices": 0, "subappendices": 0, "tblr": 1, "longtblr": 1, "NiceTabular": 1,
}


def skip_env_args(text: str, i: int, env: str) -> int:
    """Skip optional ``[..]`` arguments and the known mandatory arguments of ``env``."""
    for _ in range(2):
        o, j = read_optional(text, i)
        if o is None:
            break
        i = j
    for _ in range(ENV_ARGS.get(env, 0)):
        g, j, ok = read_group(text, i)
        if g is None:
            break
        i = j
        o, j = read_optional(text, i)
        if o is not None:
            i = j
    return i
