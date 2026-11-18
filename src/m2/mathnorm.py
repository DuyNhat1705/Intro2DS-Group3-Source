r"""Normalization of inline math to ``$...$`` and block math to ``\begin{equation}...\end{equation}``.

* labels, ``\nonumber``, ``\notag`` and ``\tag{..}`` are removed and whitespace is collapsed, so
  that the same formula always yields the same string (stable full-text deduplication);
* multi-line bodies (``align``, ``gather``, ``eqnarray`` ... or any top-level ``\\``) are wrapped
  as ``\begin{equation}\begin{aligned}...\end{aligned}\end{equation}`` so they stay valid LaTeX.
"""

from __future__ import annotations

import re

from . import config as C

_STRIP_RE = re.compile(r"\\label\s*\{[^{}]{0,300}\}|\\tag\*?\s*\{[^{}]{0,300}\}"
                       r"|\\(?:nonumber|notag)(?![A-Za-z@])")
_WS_RE = re.compile(r"\s+")
_STRAY_DELIM_RE = re.compile(r"(?<!\\)((?:\\\\)*)\\[()\[\]]")
_ENV_TOK_RE = re.compile(r"\\(begin|end)\s*\{[^}]*\}|\\\\")
# Leading argument of some environments: alignat{2}, IEEEeqnarray{rCl}, empheq[..]{align}
_LEAD_ARG_ENVS = {"alignat", "alignat*", "xalignat", "xalignat*", "xxalignat", "IEEEeqnarray",
                  "IEEEeqnarray*", "empheq", "darray"}
_LEAD_ARG_RE = re.compile(r"\s*(?:\[[^\]]{0,200}\])?\s*\{[^{}]{0,100}\}")

EQ_OPEN, EQ_CLOSE = "\\begin{equation}", "\\end{equation}"
AL_OPEN, AL_CLOSE = "\\begin{aligned}", "\\end{aligned}"


def clean_math(body: str) -> str:
    if "\\" in body:
        body = _STRIP_RE.sub(" ", body)
        if "\\(" in body or "\\)" in body or "\\[" in body or "\\]" in body:
            # stray math delimiters inside a formula (malformed source such as "$a\) b$")
            body = _STRAY_DELIM_RE.sub(r"\1 ", body)
    return _WS_RE.sub(" ", body).strip()


def normalize_inline(body: str) -> str:
    b = clean_math(body)
    return f"${b}$" if b else ""


def _top_level_linebreak(body: str) -> bool:
    if "\\\\" not in body:
        return False
    depth = 0
    for m in _ENV_TOK_RE.finditer(body):
        g = m.group(1)
        if g == "begin":
            depth += 1
        elif g == "end":
            depth = max(0, depth - 1)
        elif depth == 0:
            return True
    return False


def normalize_block(env: str, body: str) -> str | None:
    """Return the normalized block formula, or ``None`` when the body is empty."""
    if env in _LEAD_ARG_ENVS:
        m = _LEAD_ARG_RE.match(body)
        if m:
            body = body[m.end():]
    b = clean_math(body)
    if not b:
        return None
    if env in C.MULTILINE_MATH_ENVS or _top_level_linebreak(b):
        # trailing "\\" (empty last row) is dropped
        b = re.sub(r"(?:\s*\\\\)+\s*$", "", b)
        return f"{EQ_OPEN}{AL_OPEN}{b}{AL_CLOSE}{EQ_CLOSE}"
    return f"{EQ_OPEN}{b}{EQ_CLOSE}"


def braces_balanced(s: str) -> bool:
    depth = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth < 0:
                return False
        i += 1
    return depth == 0
