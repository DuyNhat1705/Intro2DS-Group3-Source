r"""Tree building from the scanner token stream with a rank stack.

* one ``document`` root per version; text before the first heading attaches to the root;
* a heading of rank ``r`` pops every open node of rank ``>= r`` (and every open environment
  container), which finds the right level even when a paper skips levels;
* abstract (env, ``\abstract{}``, preamble), acknowledgements (``\section*``, ``acks`` envs,
  ``\acknowledgments``) and appendices are kept; reference sections, ``thebibliography``,
  ``\bibliography`` and ``\printbibliography`` are excluded;
* lists become ``list`` -> ``item`` -> sentences; theorem-like environments become ``block``;
* leaves are sentences, block formulas, figures and tables (in document order: a block formula
  flushes the pending sentence);
* blank-line paragraphs do not create nodes (the paragraph structure is not part of the
  statement's hierarchy); a blank line only ends the pending sentence.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

from . import config as C
from .clean import FN_MARK_RE, clean_inline, clean_with_footnotes
from .mathnorm import normalize_block
from .preprocess import BL_OPEN, IL_OPEN, Literals, match_brace, match_bracket
from .scanner import (STRUCT_RE, escaped, find_display_end, find_env_end, find_inline_math_end,
                      read_group, read_optional, skip_env_args)
from .sentences import LRUBytes, SentenceSplitter

HEADING_KINDS = frozenset(C.SECTION_RANKS)
CONTAINER_KINDS = frozenset({"abstract", "list", "item", "block"})
_REF_TITLES = frozenset(C.REFERENCE_TITLES) | {"references and notes", "references cited",
                                               "list of references", "bibliography and references"}
_WHOLE_ENVS = C.FIGURE_ENVS | C.TABLE_ENVS | C.TABULAR_ENVS | C.DROPPED_ENVS | {"thebibliography", "math"}
_MATH_BLOCK = frozenset(e for e in C.BLOCK_MATH_ENVS if e != "subequations")
_CAPTION_RE = re.compile(r"\\(caption|captionof|subcaption|subcaptionbox|sidecaption|subfloat|"
                         r"tablecaption|figcaption)\*?(?![A-Za-z@])")
_GRAPHICS_RE = re.compile(r"\\(?:includegraphics|includesvg|epsfig|epsfbox|includeinkscape)\*?\s*"
                          r"(?:\[[^\]]{0,500}\])?\s*\{([^{}]{1,300})\}")
_TABULAR_BEGIN_RE = re.compile(r"\\begin\s*\{(" + "|".join(re.escape(e) for e in sorted(C.TABULAR_ENVS, key=len, reverse=True)) + r")\}")
_ROW_SPLIT_RE = re.compile(r"\\\\(?:\*)?(?:\s*\[[^\]]{0,40}\])?|\\tabularnewline(?![A-Za-z])|\\cr(?![A-Za-z])")
_RULES_RE = re.compile(r"\\(?:hline|toprule|midrule|bottomrule|hdashline|Xhline\{[^}]*\}|"
                       r"specialrule\{[^}]*\}\{[^}]*\}\{[^}]*\}|cmidrule(?:\([^)]*\))?\{[^}]*\}|"
                       r"cline\{[^}]*\}|cdashline\{[^}]*\}|addlinespace(?:\[[^\]]*\])?|"
                       r"rowcolor(?:\[[^\]]*\])?\{[^}]*\}|morecmidrules|endhead|endfirsthead|"
                       r"endfoot|endlastfoot|noalign\{[^}]*\})")
_LABEL_RE = re.compile(r"\\label\s*\{([^{}]{0,200})\}")


@dataclass(slots=True)
class Node:
    kind: str
    content: str = ""
    children: list = field(default_factory=list)
    rank: int = 99

    def walk(self):
        stack = [self]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))


@dataclass
class TreeStats:
    n_nodes: int = 0
    n_sentences: int = 0
    n_formulas: int = 0
    n_figures: int = 0
    n_tables: int = 0
    n_lists: int = 0
    n_blocks: int = 0
    max_depth: int = 0
    n_verbatim: int = 0
    n_section_cmds: int = 0
    n_section_nodes: int = 0
    n_unbalanced_formulas: int = 0
    recoveries: int = 0
    refs_skipped: int = 0


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z ]+", "", t.lower()).strip()


class TreeBuilder:
    """Builds one version tree from the (macro-expanded) body."""

    def __init__(self, lits: Literals, theorem_names: set[str] | None = None,
                 splitter: SentenceSplitter | None = None, para_cache: LRUBytes | None = None,
                 safe_mode: bool = False):
        self.lits = lits
        self.block_envs = C.BLOCK_ENVS | set(theorem_names or ())
        self.splitter = splitter or SentenceSplitter()
        self.para_cache = para_cache
        self.safe_mode = safe_mode
        self.root = Node("document", rank=-99)
        self.stack: list[Node] = [self.root]
        self.buf: list[str] = []
        self.skip_rank: int | None = None
        self.in_appendix = False
        self.stats = TreeStats()
        self.has_abstract = False

    # ------------------------------------------------------------------ helpers
    @property
    def top(self) -> Node:
        return self.stack[-1]

    def _add(self, node: Node) -> None:
        if self.skip_rank is None:
            self.top.children.append(node)

    def flush(self) -> None:
        if not self.buf:
            return
        raw = "".join(self.buf)
        self.buf = []
        if self.skip_rank is not None or not raw.strip():
            return
        for s in self.paragraph_sentences(raw):
            self.top.children.append(Node("sentence", s))

    def paragraph_sentences(self, raw: str) -> list[str]:
        """Clean + split one paragraph (cached); footnote sentences follow their host."""
        key = raw
        if IL_OPEN in raw:
            key = raw + "\x00" + self.lits.restore_inline(raw)
        if self.para_cache is not None:
            hit = self.para_cache.get(key)
            if hit is not None:
                return list(hit)
        if BL_OPEN in raw:
            self.stats.n_verbatim += raw.count(BL_OPEN)
        text, fns = clean_with_footnotes(raw, self.lits)
        out: list[str] = []
        if text:
            for s in self.splitter.split(text):
                if "\ue005" in s:
                    marks = [int(x) for x in FN_MARK_RE.findall(s)]
                    s2 = re.sub(r"\s+", " ", FN_MARK_RE.sub("", s)).strip()
                    s2 = re.sub(r" +([,.;:!?])", r"\1", s2)
                    if s2 and re.search(r"\w", s2):
                        out.append(s2)
                    for k in marks:
                        if 0 <= k < len(fns):
                            out.extend(x for x in self.splitter.split(fns[k]) if re.search(r"\w", x))
                elif re.search(r"\w", s):
                    out.append(s)
        if self.para_cache is not None:
            self.para_cache.put(key, tuple(out))
        return out

    def _pop_to_heading(self) -> None:
        while len(self.stack) > 1 and self.top.kind not in HEADING_KINDS:
            self.stack.pop()

    def open_heading(self, kind: str, title: str) -> None:
        rank = C.SECTION_RANKS[kind]
        self.flush()
        while len(self.stack) > 1 and (self.top.kind not in HEADING_KINDS or self.top.rank >= rank):
            self.stack.pop()
        node = Node(kind, title, rank=rank)
        self.top.children.append(node)
        self.stack.append(node)
        self.stats.n_section_nodes += kind == "section"

    def open_container(self, kind: str, env: str) -> None:
        self.flush()
        node = Node(kind, "", rank=100)
        node.content = env  # temporarily remember the env name (replaced in finalize)
        self._add(node)
        if self.skip_rank is None:
            self.stack.append(node)

    def close_container(self, env: str) -> None:
        self.flush()
        for k in range(len(self.stack) - 1, 0, -1):
            n = self.stack[k]
            if n.kind in CONTAINER_KINDS and n.content == env:
                del self.stack[k:]
                return
            if n.kind == "item" and env in C.LIST_ENVS:
                continue
            if n.kind in HEADING_KINDS and n.content.startswith("\x00ack:") and n.content[5:] == env:
                del self.stack[k:]
                return

    # ------------------------------------------------------------------ leaves
    def add_formula(self, env: str, body: str) -> None:
        self.flush()
        f = normalize_block(env, body)
        if f:
            from .mathnorm import braces_balanced
            if not braces_balanced(f):
                self.stats.n_unbalanced_formulas += 1
            self._add(Node("formula", f))

    def _captions(self, body: str) -> list[str]:
        caps = []
        for m in _CAPTION_RE.finditer(body):
            cmd = m.group(1)
            i = m.end()
            opt, i2 = read_optional(body, i)
            if cmd == "subfloat":
                if opt:
                    caps.append(opt)
                continue
            g, i3, _ok = read_group(body, i2)
            if g is None:
                continue
            if cmd == "captionof":
                g2, _i4, _ok2 = read_group(body, i3)
                g = g2 if g2 is not None else ""
            caps.append(g)
        out = []
        for c in caps:
            t = clean_inline(c, self.lits)
            if t:
                out.append(t)
        return out

    def _tabular_rows(self, body: str) -> list[str]:
        rows_out: list[str] = []
        for m in _TABULAR_BEGIN_RE.finditer(body):
            env = m.group(1)
            s = skip_env_args(body, m.end(), env)
            e, _after = find_env_end(body, m.end(), env)
            if e < 0:
                e = len(body)
            inner = _RULES_RE.sub(" ", body[s:e])
            for row in _split_top(inner, _ROW_SPLIT_RE):
                cells = [clean_inline(c, self.lits) for c in _split_cells(row)]
                if any(cells):
                    rows_out.append(" | ".join(c for c in cells))
        return rows_out

    def add_float(self, env: str, body: str) -> None:
        self.flush()
        is_table = env in C.TABLE_ENVS or env in C.TABULAR_ENVS
        if env in C.TABULAR_ENVS:
            # standalone tabular: body starts with the column spec
            body = "\\begin{" + env + "}" + body + "\\end{" + env + "}"
        caps = self._captions(body)
        if is_table:
            rows = self._tabular_rows(body)
            parts = []
            if caps:
                parts.append(" ".join(caps))
            if rows:
                parts.append("\n".join(rows))
            content = "\n".join(parts)
            kind = "table"
        else:
            content = " ".join(caps)
            kind = "figure"
        if not content:
            names = [posixpath.basename(g.strip()) for g in _GRAPHICS_RE.findall(body)]
            content = " ".join(n for n in names if n)
        if not content:
            lab = _LABEL_RE.search(body)
            content = lab.group(1).strip() if lab else env.rstrip("*")
        self._add(Node(kind, content))

    # ------------------------------------------------------------------ main loop
    def build(self, body: str, preamble: str = "") -> Node:
        if preamble:
            self._preamble_abstract(preamble)
        self._parse(body)
        self.flush()
        self._finalize()
        return self.root

    def _preamble_abstract(self, pre: str) -> None:
        m = re.search(r"\\begin\s*\{abstract\}", pre)
        if m:
            e, _a = find_env_end(pre, m.end(), "abstract")
            if e > 0:
                self._abstract_text(pre[m.end():e])
                return
        m = re.search(r"\\abstract\s*\{", pre)
        if m:
            k = match_brace(pre, m.end() - 1)
            if k > 0:
                self._abstract_text(pre[m.end():k - 1])

    def _abstract_text(self, text: str) -> None:
        if not text.strip():
            return
        self.open_container("abstract", "abstract")
        self.has_abstract = True
        self._parse(text)
        self.close_container("abstract")

    def _parse(self, text: str) -> None:
        i = 0
        n = len(text)
        search = STRUCT_RE.search
        while i < n:
            m = search(text, i)
            if not m:
                self.buf.append(text[i:])
                break
            s = m.start()
            if s > i:
                self.buf.append(text[i:s])
            i = m.end()
            g = m.lastgroup
            if g == "sstar":
                g = "sec"
            if m.group("sec"):
                i = self._heading(text, m.group("sec"), i)
            elif m.group("be"):
                i = self._env(text, m.group("be"), m.group("env"), i, s)
            elif m.group("item"):
                i = self._item(text, i)
            elif m.group("dd"):
                if escaped(text, s):
                    self.buf.append("$$")
                    continue
                k = find_display_end(text, i, "$$")
                if k < 0:
                    self.stats.recoveries += 1
                    continue
                self.add_formula("equation", text[i:k])
                i = k + 2
            elif m.group("d"):
                if escaped(text, s):
                    self.buf.append("$")
                    continue
                k = find_inline_math_end(text, i, "$")
                if k < 0:
                    continue  # stray '$' is dropped
                self.buf.append(text[s:k + 1])
                i = k + 1
            elif m.group("mopen"):
                if escaped(text, s):
                    self.buf.append(m.group(0))
                    continue
                if m.group("mopen") == "(":
                    k = find_inline_math_end(text, i, "\\)")
                    if k < 0:
                        continue
                    self.buf.append(text[s:k + 2])
                    i = k + 2
                else:
                    k = find_display_end(text, i, "\\]")
                    if k < 0:
                        self.stats.recoveries += 1
                        continue
                    self.add_formula("equation", text[i:k])
                    i = k + 2
            elif m.group("misc"):
                i = self._misc(text, m.group("misc"), i)
            elif m.group("par") is not None:
                self.flush()

    # ------------------------------------------------------------------ token handlers
    def _heading(self, text: str, kind: str, i: int) -> int:
        _opt, i2 = read_optional(text, i)
        g, i3, ok = read_group(text, i2)
        if not ok:
            self.stats.recoveries += 1
            # recover: title until end of line
            j = text.find("\n", i2)
            j = len(text) if j < 0 else j
            g, i3 = text[i2:j].lstrip("{"), j
        if g is None:
            return i
        if kind == "section":
            self.stats.n_section_cmds += 1
        title = clean_inline(g.replace("\\\\", " "), self.lits)
        rank = C.SECTION_RANKS[kind]
        if self.skip_rank is not None:
            if rank <= self.skip_rank:
                self.skip_rank = None
            else:
                return i3
        nt = _norm_title(title)
        if nt in _REF_TITLES or (nt.startswith("references") and len(nt) < 30):
            self.flush()
            self.skip_rank = rank
            self.stats.refs_skipped += 1
            return i3
        self.open_heading(kind, title)
        return i3

    def _env(self, text: str, be: str, env: str, i: int, s: int) -> int:
        if be == "end":
            if env in C.LIST_ENVS:
                self.close_container(env)
            elif env in C.ABSTRACT_ENVS:
                self.close_container("abstract")
            elif env in C.ACK_ENVS:
                self.close_container(env)
            elif env in self.block_envs:
                self.close_container(env)
            return i
        # ---- begin
        if env in _MATH_BLOCK:
            e, after = find_env_end(text, i, env)
            if e < 0:
                self.stats.recoveries += 1
                j = text.find("\n\n", i)
                e = after = (len(text) if j < 0 else j)
            self.add_formula(env, text[i:e])
            return after
        if env in _WHOLE_ENVS:
            e, after = find_env_end(text, i, env)
            if e < 0:
                self.stats.recoveries += 1
                return i
            body = text[i:e]
            if env == "math":
                self.buf.append("$" + body + "$")
            elif env in C.FIGURE_ENVS or env in C.TABLE_ENVS or env in C.TABULAR_ENVS:
                if self.skip_rank is None:
                    self.add_float(env, body)
            else:
                self.flush()
                if env == "thebibliography":
                    self.stats.refs_skipped += 1
            return after
        if env in C.LIST_ENVS:
            i = skip_env_args(text, i, env)
            self.open_container("list", env)
            return i
        if env in C.ABSTRACT_ENVS:
            self.open_container("abstract", "abstract")
            self.has_abstract = True
            return i
        if env in C.ACK_ENVS:
            self.flush()
            if self.skip_rank is not None:
                self.skip_rank = None
            while len(self.stack) > 1 and (self.top.kind not in HEADING_KINDS or self.top.rank >= 1):
                self.stack.pop()
            node = Node("section", "\x00ack:" + env, rank=1)
            self.top.children.append(node)
            self.stack.append(node)
            return skip_env_args(text, i, env)
        if env in self.block_envs:
            i = skip_env_args(text, i, env)
            self.open_container("block", env)
            return i
        if env in C.APPENDIX_ENVS:
            self.flush()
            self.in_appendix = True
            return i
        # transparent environment: drop the token and its arguments
        return skip_env_args(text, i, env)

    def _item(self, text: str, i: int) -> int:
        lst = None
        for k in range(len(self.stack) - 1, 0, -1):
            if self.stack[k].kind == "list":
                lst = k
                break
            if self.stack[k].kind in HEADING_KINDS:
                break
        if lst is None:
            return i
        self.flush()
        del self.stack[lst + 1:]
        item = Node("item", "", rank=101)
        self.top.children.append(item)
        self.stack.append(item)
        label, j = read_optional(text, i)
        if label is not None:
            self.buf.append(label + " ")
            return j
        return i

    def _misc(self, text: str, cmd: str, i: int) -> int:
        if cmd == "appendix":
            self.flush()
            self.in_appendix = True
            self.skip_rank = None
            return i
        if cmd in ("bibliography", "nobibliography"):
            self.flush()
            _g, j, _ok = read_group(text, i)
            return j
        if cmd == "printbibliography":
            self.flush()
            _o, j = read_optional(text, i)
            return j
        if cmd == "abstract":
            g, j, ok = read_group(text, i)
            if g is not None and ok:
                self.flush()
                self._abstract_text(g)
                return j
            return i
        if cmd in ("acknowledgments", "acknowledgements"):
            self.flush()
            self.skip_rank = None
            while len(self.stack) > 1 and (self.top.kind not in HEADING_KINDS or self.top.rank >= 1):
                self.stack.pop()
            node = Node("section", "Acknowledgments", rank=1)
            self.top.children.append(node)
            self.stack.append(node)
            return i
        if cmd == "twocolumn":
            o, j = read_optional(text, i)
            return j
        if cmd in ("title", "author", "date", "keywords"):
            _o, j = read_optional(text, i)
            g, j2, ok = read_group(text, j)
            return j2 if g is not None else i
        if cmd == "bibitem":
            # a stray \bibitem outside thebibliography: stop prose until next structure
            self.flush()
            return i
        return i  # maketitle, IEEEraisesectionheading, IEEEkeywords

    # ------------------------------------------------------------------ finalize
    def _finalize(self) -> None:
        st = self.stats
        stack = [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            st.n_nodes += 1
            st.max_depth = max(st.max_depth, d)
            k = node.kind
            if k in CONTAINER_KINDS:
                node.content = ""
            elif k in HEADING_KINDS and node.content.startswith("\x00ack:"):
                node.content = "Acknowledgments"
            if k == "sentence":
                st.n_sentences += 1
            elif k == "formula":
                st.n_formulas += 1
            elif k == "figure":
                st.n_figures += 1
            elif k == "table":
                st.n_tables += 1
            elif k == "list":
                st.n_lists += 1
            elif k == "block":
                st.n_blocks += 1
            # drop empty containers
            node.children = [c for c in node.children
                             if not (c.kind in CONTAINER_KINDS and not _has_leaf(c))]
            for c in node.children:
                stack.append((c, d + 1))


def _has_leaf(n: Node) -> bool:
    for x in n.walk():
        if x.kind in ("sentence", "formula", "figure", "table"):
            return True
    return False


def _split_top(s: str, sep: re.Pattern) -> list[str]:
    """Split on ``sep`` at brace depth 0."""
    out, depth, last = [], 0, 0
    i = 0
    n = len(s)
    for m in sep.finditer(s):
        # compute depth up to m.start()
        while i < m.start():
            c = s[i]
            if c == "\\":
                i += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        if depth <= 0:
            out.append(s[last:m.start()])
            last = m.end()
        i = max(i, m.end())
    out.append(s[last:])
    return out


def _split_cells(row: str) -> list[str]:
    cells, depth, cur = [], 0, []
    i = 0
    n = len(row)
    while i < n:
        c = row[i]
        if c == "\\" and i + 1 < n:
            cur.append(row[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == "&" and depth <= 0:
            cells.append("".join(cur))
            cur = []
            i += 1
            continue
        cur.append(c)
        i += 1
    cells.append("".join(cur))
    return cells
