"""Interfaces for per-version parsing and publication-wide element tables.

Colab CPU: install ``wtpsplit==2.2.2`` and ``onnxruntime==1.31.1`` (or
install this project's requirements), then import and call construct_hierarchy.
The first nonempty prose passage downloads sat-3l-sm into the Hugging Face
cache; subsequent passages reuse the model in the Python process. Inference
runs locally on CPU and requires no inference service or API key.
"""

from dataclasses import dataclass
from functools import lru_cache
import re
import warnings
from typing import TypedDict


class HierarchyDeduplicationWarning(UserWarning):
    """Review-only schema ambiguity; all elements and occurrence edges are retained."""


@dataclass
class Node:
    local_id: str
    parent_local_id: str | None
    kind: str
    content: str


class ElementRow(TypedDict):
    element_id: str
    arxiv_id: str
    kind: str
    content: str


class HierarchyRow(TypedDict):
    arxiv_id: str
    version: int
    element_id: str
    parent_id: str


_HEADINGS = {name: i for i, name in enumerate(
    ("part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph")
)}
_LISTS = {"itemize", "enumerate", "description"}
_MATH = {"equation", "align", "alignat", "flalign", "gather", "multline", "eqnarray", "displaymath", "dmath"}
_FIGURES = {"figure", "table", "tabular", "tabularx", "longtable"}
_COMMAND = re.compile(r"\\([A-Za-z@]+\*?|[^A-Za-z@])")
_URL = re.compile(r"(?:https?://|www\.)[^\s{}<>]+", re.I)
_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_RULE_END = re.compile(r'''[.!?]+["'\u201d\u2019)\]]*\s+(?=["'\u201c\u2018(\[]*[A-Z0-9])''')
_ABBREVIATION_END = re.compile(
    r"\b(?:figs?|eqs?|secs?|refs?|dr|mr|mrs|ms|prof|no|vs|etc|e\.g|i\.e)\.$", re.I
)
_PROSE_ENV = re.compile(
    r"(?:(?:main|sub|inner)?(?:theorem|thm|lemma|lem|proposition|prop|definition|defi|defn|"
    r"remark|rem|rmk|corollary|cor|coro|claim|example|exam|assumption|conjecture|"
    r"observation|question|problem)|proof|IEEEproof|quote|quotation|abstract|document|"
    r"titlepage|frontmatter|keywords|IEEEkeywords|acknowledg(?:e)?ments)\*?", re.I
)
_LITERAL_ENVS = {"verbatim", "verbatim*", "Verbatim", "Verbatim*", "BVerbatim",
                 "lstlisting", "minted", "alltt"}


def _literal_end(text: str, match: re.Match[str]) -> int | None:
    """Skip literal TeX without interpreting its contents as document structure."""
    if match.group(1).rstrip("*") == "verb":
        p = match.end()
        if p < len(text) and not text[p].isspace():
            end = text.find(text[p], p + 1)
            return end + 1 if end >= 0 else len(text)
    if match.group(1) == "begin":
        p = _space(text, match.end())
        end = _group(text, p)
        if end is not None:
            name = text[p + 1:end - 1].strip()
            if name in _LITERAL_ENVS:
                close = re.search(r"\\end\s*\{" + re.escape(name) + r"\}", text[end:])
                return end + close.end() if close else len(text)
    return None


def _group(text: str, pos: int, opening: str = "{", closing: str = "}") -> int | None:
    """Return the offset after a balanced group, respecting escaped characters."""
    if pos >= len(text) or text[pos] != opening:
        return None
    depth = 1
    braces = 0
    i = pos + 1
    while i < len(text):
        ch = text[i]
        if ch == "\\":
            i += 2
            continue
        if opening != "{" and ch == "{":
            braces += 1
        elif opening != "{" and ch == "}" and braces:
            braces -= 1
        elif not braces and ch == opening:
            depth += 1
        elif not braces and ch == closing:
            depth -= 1
            if not depth:
                return i + 1
        i += 1
    return None


def _space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _arguments(text: str, pos: int, *, optional: bool = True) -> int:
    """Skip adjacent braced arguments and, outside math, optional arguments."""
    end = pos
    while True:
        p = _space(text, end)
        if p >= len(text) or text[p] not in ("[{" if optional else "{"):
            return end
        q = _group(text, p, text[p], "]" if text[p] == "[" else "}")
        if q is None:
            warnings.warn("Unclosed command argument; preserving remaining text", UserWarning, stacklevel=3)
            return end
        end = q


def _environment_end(text: str, start: int, name: str) -> int | None:
    """Find the closing token without interpreting commands inside arguments."""
    depth = 1
    i = start
    while i < len(text):
        if text[i] != "\\":
            i += 1
            continue
        match = _COMMAND.match(text, i)
        if match is None:
            i += 1
            continue
        literal_end = _literal_end(text, match)
        if literal_end is not None:
            i = literal_end
            continue
        if match.group(1) in {"begin", "end"}:
            p = _space(text, match.end())
            end = _group(text, p)
            if end is not None:
                if text[p + 1:end - 1].strip() == name:
                    depth += 1 if match.group(1) == "begin" else -1
                    if not depth:
                        return end
                i = end
                continue
        # In math, square brackets can delimit half-open intervals rather than
        # command options, e.g. \operatorname{Unif}[0,2\pi) or \in[0,1/2).
        # Only brace groups shield structural tokens while scanning math.
        i = _arguments(text, match.end(), optional=name.rstrip("*") not in _MATH)
    return None


@lru_cache(maxsize=1)
def _sentence_model():
    """Load the CPU model once per process, only when prose is present."""
    try:
        from wtpsplit import SaT
        return SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise RuntimeError(
            "Could not load SaT sat-3l-sm on ONNX CPU; install the project "
            "requirements and ensure the model is cached or downloadable."
        ) from exc


# Only these formatting commands expose their prose arguments to SaT. Unknown
# macros and their arguments remain opaque, since their meaning is not known.
_PROSE_COMMANDS = {"textit", "textbf", "texttt", "textrm", "textsf", "textsl", "textsc", "emph", "mbox"}


def _sentence_projection(text: str) -> tuple[str, list[int | None]]:
    """Project prose for SaT and map each projected boundary to its source."""
    chars: list[str] = []
    offsets = [0]

    def copy(pos: int, end: int, masked: bool = False) -> None:
        for j in range(pos, end):
            chars.append(" " if masked else text[j])
            offsets.append(j + 1)

    def protect(end: int, placeholder: str) -> None:
        # Interior boundaries of a placeholder have no source counterpart.
        chars.extend(placeholder)
        offsets.extend([None] * (len(placeholder) - 1))
        offsets.append(end)

    def scan(start: int, end: int) -> None:
        i = start
        while i < end:
            ch = text[i]
            url = _URL.match(text, i, end) if ch in "hHwW" else None
            if url:
                j = url.end()
                while j > i and text[j - 1] in ".!?,;:)]}\"'":
                    j -= 1
                if j > i:
                    protect(j, "url")
                    i = j
                    continue
            if ch == "$":
                j = i + 1
                while j < end:
                    if text[j] == "\\":
                        j += 2
                    elif text[j] == "$":
                        j += 1
                        break
                    else:
                        j += 1
                i = min(j, end)
                protect(i, "formula")
                continue
            if ch == "\\":
                if text.startswith(r"\(", i):
                    j = text.find(r"\)", i + 2, end)
                    i = j + 2 if j >= 0 else end
                    protect(i, "formula")
                    continue
                match = _COMMAND.match(text, i)
                if match:
                    literal_end = _literal_end(text, match)
                    if literal_end is not None:
                        i = min(literal_end, end)
                        protect(i, "literal")
                        continue
                    p = _space(text, match.end())
                    if match.group(1) in _PROSE_COMMANDS and p < end and text[p] == "{":
                        j = _group(text, p)
                        if j is not None and j <= end:
                            copy(i, p + 1, masked=True)
                            scan(p + 1, j - 1)
                            copy(j - 1, j, masked=True)
                            i = j
                            continue
                    i = min(_arguments(text, match.end()), end)
                    protect(i, "reference" if "cite" in match.group(1) or "ref" in match.group(1) else "latex")
                    continue
            if ch == "{":
                j = _group(text, i)
                if j is not None and j <= end:
                    copy(i, i + 1, masked=True)
                    scan(i + 1, j - 1)
                    copy(j - 1, j, masked=True)
                    i = j
                    continue
            copy(i, i + 1)
            i += 1

    scan(0, len(text))
    return "".join(chars), offsets


def _blocked_sentence_boundary(text: str, end: int) -> bool:
    """Guard explicit rules/model cuts at common abbreviations and initials."""
    # A model boundary must not split the digits or decimal point of a number.
    if 0 < end < len(text) and (
        (text[end - 1].isdigit() and (text[end].isdigit() or
         (text[end] == "." and end + 1 < len(text) and text[end + 1].isdigit()))) or
        (text[end - 1] == "." and text[end].isdigit() and end >= 2 and text[end - 2].isdigit())
    ):
        return True
    prefix = text[max(0, end - 200):end].rstrip().rstrip("\"'\u201d\u2019)]").rstrip()
    if _ABBREVIATION_END.search(prefix):
        return True
    following = re.match(r'''["'\u201c\u2018(\[]*([A-Za-z]+)(\.?)''', text[end:end + 200].lstrip())
    if not following:
        return False
    word, dot = following.groups()
    if re.search(r"\bet\s+al\.$", prefix, re.I):
        return word[0].islower()
    initial = re.search(r"\b[A-Z]\.$", prefix)
    if not initial:
        return False
    if len(word) == 1 and dot:
        return True
    before = prefix[:initial.start()].rstrip()
    previous = re.search(r"([A-Za-z]+)\.?$", before)
    # Part I. / method A. may end a sentence; a title-cased word before an
    # initial usually belongs to a name. This is a heuristic, not NER.
    labels = {"part", "section", "appendix", "method", "case", "type", "model", "algorithm", "option"}
    if previous and previous[1].lower() in labels:
        return False
    if previous and (re.fullmatch(r"[A-Z][a-z]+", previous[1]) or
                     (len(previous[1]) == 1 and before.endswith('.'))):
        return word[0].isupper()
    return not before and word[0].isupper() and word not in {"The", "This", "Next", "We", "It", "See"}


def _sentences(text: str):
    """Combine SaT and guarded punctuation rules; slice original TeX only."""
    if not text.strip():
        return
    projected, offsets = _sentence_projection(text)
    # wtpsplit 2.2.2 emits this informational warning on every call, even
    # without newlines. Other warnings must remain visible.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="split_on_input_newlines=False will lead to newlines.*", category=UserWarning)
        parts = list(_sentence_model().split(projected, split_on_input_newlines=False, strip_whitespace=False))
    if not all(isinstance(part, str) for part in parts) or "".join(parts) != projected:
        raise ValueError("SaT output does not reconstruct the projected input; cannot map sentence boundaries")
    ends = {len(projected)}
    projected_end = 0
    for part in parts:
        projected_end += len(part)
        if projected_end == len(projected) or not _blocked_sentence_boundary(projected, projected_end):
            ends.add(projected_end)
    for match in _RULE_END.finditer(projected):
        if not match.group().rstrip().endswith('...') and not _blocked_sentence_boundary(projected, match.end()):
            ends.add(match.end())
    source_start = 0
    for projected_end in sorted(ends):
        source_end = offsets[projected_end]
        if source_end is None or source_end < source_start:
            # Merge any model split inside a protected token or already
            # consumed closing brace into the next source slice.
            continue
        # A masked formatting-group close belongs to the preceding sentence.
        while source_end < len(text) and text[source_end] == "}":
            source_end += 1
        piece = text[source_start:source_end].strip()
        if piece:
            yield piece
        source_start = source_end


def construct_hierarchy(cleaned_tex: str, key_map: dict[str, str] | None = None) -> list[Node]:
    r"""Build a deterministic preorder tree from one cleaned LaTeX version.

    Empty or whitespace-only input returns []; otherwise n0 is the document
    root. Integration kinds are document, heading, list, item, sentence,
    equation and figure (including standalone tabular). List content is empty;
    item content is its optional label or empty. Abstract is a transparent
    container. Only heading title text is stored. SaT proposes prose sentence
    boundaries with guarded punctuation rules on a projection protecting TeX. Unknown
    commands and their arguments remain in prose. Bibliography headings are
    omitted until a heading at the same or higher level. Malformed constructs
    emit UserWarning and are preserved where practical. Typical cost is linear
    in input plus output, excluding pathological deeply nested TeX groups.
    Pass one version's bibliography key_map to canonicalize citations before
    building the tree; omit it to retain the original citation keys.
    """
    if not isinstance(cleaned_tex, str):
        raise TypeError("cleaned_tex must be a str")
    if not cleaned_tex.strip():
        return []

    # Canonicalize before parsing without changing the tree-building algorithm.
    text = _rewrite_citation_text(cleaned_tex, key_map) if key_map else cleaned_tex
    nodes = [Node("n0", None, "document", "")]
    headings: list[tuple[int, str]] = []
    # Each entry is (environment name, list ID, current item ID).
    lists: list[tuple[str, str, str | None]] = []
    skipping: int | None = None

    def add(kind: str, content: str, parent: str) -> str:
        node_id = f"n{len(nodes)}"
        nodes.append(Node(node_id, parent, kind, content))
        return node_id

    def parent() -> str:
        if lists:
            return lists[-1][2] or lists[-1][1]
        return headings[-1][1] if headings else "n0"

    def flush(start: int, end: int) -> None:
        if skipping is None:
            for sentence in _sentences(text[start:end]):
                add("sentence", sentence, parent())

    i = start = 0
    n = len(text)
    while i < n:
        # Paragraph boundaries also end an unfinished sentence fragment.
        paragraph_break = _PARAGRAPH_BREAK.match(text, i) if text[i] == "\n" else None
        if paragraph_break:
            flush(start, i)
            i = paragraph_break.end()
            start = i
            continue
        if text.startswith("$$", i) or text.startswith(r"\[", i):
            opener = text[i:i + 2]
            close = "$$" if opener == "$$" else r"\]"
            end = text.find(close, i + 2)
            if end >= 0:
                flush(start, i)
                end += 2
                if skipping is None:
                    add("equation", text[i:end], parent())
                i = start = end
                continue
            warnings.warn("Unclosed display math; preserving remaining text", UserWarning, stacklevel=2)
            i += 2
            continue
        if text[i] == "$":
            j = i + 1
            while j < n and text[j] != "$":
                j += 2 if text[j] == "\\" else 1
            if j >= n:
                warnings.warn("Unclosed inline math", UserWarning, stacklevel=2)
            i = min(j + 1, n)
            continue
        if text[i] == "{":
            end = _group(text, i)
            if end is None:
                warnings.warn("Unclosed group; preserving remaining text", UserWarning, stacklevel=2)
            # A bare group can contain document structure. Command arguments
            # are still skipped by _arguments below and remain opaque.
            i += 1
            continue
        if text[i] != "\\":
            i += 1
            continue
        if text.startswith(r"\(", i):
            end = text.find(r"\)", i + 2)
            if end < 0:
                warnings.warn("Unclosed inline math", UserWarning, stacklevel=2)
                i = n
            else:
                i = end + 2
            continue
        match = _COMMAND.match(text, i)
        if not match:
            i += 1
            continue
        command = match.group(1)
        base = command.rstrip("*")
        literal_end = _literal_end(text, match)
        if literal_end is not None:
            i = literal_end
            continue
        if base in _HEADINGS:
            p = _space(text, match.end())
            if p < n and text[p] == "[":
                end = _group(text, p, "[", "]")
                if end is None:
                    warnings.warn("Unclosed heading option", UserWarning, stacklevel=2)
                    i = match.end()
                    continue
                p = _space(text, end)
            end = _group(text, p)
            if end is None:
                warnings.warn("Heading missing balanced title", UserWarning, stacklevel=2)
                i = match.end()
                continue
            level = _HEADINGS[base]
            title = text[p + 1:end - 1].strip()
            flush(start, i)
            if skipping is not None and level <= skipping:
                skipping = None
            if skipping is None:
                while headings and headings[-1][0] >= level:
                    headings.pop()
                if lists:
                    warnings.warn("Heading closes an unclosed list", UserWarning, stacklevel=2)
                    lists.clear()
                if re.fullmatch(r"(?:references|bibliography|literature cited|\\refname|\\bibname)\s*", title, re.I):
                    skipping = level
                else:
                    node_id = add("heading", title, parent())
                    headings.append((level, node_id))
            i = start = end
            continue
        if base in {"begin", "end"}:
            p = _space(text, match.end())
            end = _group(text, p)
            if end is not None:
                env = text[p + 1:end - 1].strip()
                plain = env.rstrip("*")
                if base == "begin" and (plain in _MATH or plain in _FIGURES or env == "thebibliography"):
                    close = _environment_end(text, end, env)
                    if close is not None:
                        flush(start, i)
                        if skipping is None and env != "thebibliography":
                            add("equation" if plain in _MATH else "figure", text[i:close], parent())
                        i = start = close
                        continue
                    warnings.warn(f"Unclosed environment {env}; preserving remaining text", UserWarning, stacklevel=2)
                    if env == "thebibliography":
                        # A damaged bibliography must still be excluded. A later
                        # peer heading or appendix can resume normal parsing.
                        skipping = headings[-1][0] if headings else max(_HEADINGS.values())
                        i = start = end
                        continue
                if env in _LISTS:
                    flush(start, i)
                    if skipping is None:
                        if base == "begin":
                            list_id = add("list", "", parent())
                            lists.append((env, list_id, None))
                        elif lists and lists[-1][0] == env:
                            lists.pop()
                        else:
                            warnings.warn(f"Unmatched list end {env}", UserWarning, stacklevel=2)
                    i = start = end
                    continue
                if _PROSE_ENV.fullmatch(env):
                    flush(start, i)
                    i = start = end
                    if base == "begin":
                        p = _space(text, end)
                        if p < n and text[p] == "[":
                            q = _group(text, p, "[", "]")
                            if q is not None:
                                i = q  # keep the optional label as prose, shield its syntax
                    continue
        if base == "item" and lists:
            flush(start, i)
            p = _space(text, match.end())
            label = ""
            end = match.end()
            if p < n and text[p] == "[":
                q = _group(text, p, "[", "]")
                if q is not None:
                    label, end = text[p + 1:q - 1], q
                else:
                    warnings.warn("Unclosed item label", UserWarning, stacklevel=2)
            if skipping is None:
                env, list_id, _ = lists[-1]
                lists[-1] = (env, list_id, add("item", label, list_id))
            i = start = end
            continue
        if base in {"appendix", "bibliography", "addbibresource", "addglobalbib", "nobibliography", "printbibliography", "bibliographystyle", "bibitem"}:
            flush(start, i)
            if base == "appendix":
                skipping = None
            if base == "bibitem" and skipping is None:
                skipping = headings[-1][0] if headings else max(_HEADINGS.values())
            i = start = _arguments(text, match.end())
            continue
        i = _arguments(text, match.end())
    flush(start, n)
    if lists:
        warnings.warn("Unclosed list environment", UserWarning, stacklevel=2)
    return nodes


def rewrite_citations(nodes: list[Node], key_map: dict[str, str]) -> list[Node]:
    """Apply one version's cite-key mapping while preserving local IDs and edges."""
    return [Node(node.local_id, node.parent_local_id, node.kind,
                 _rewrite_citation_text(node.content, key_map)) for node in nodes]


_CITE_COMMANDS = {
    "cite", "citep", "citet", "citealp", "citealt", "citeauthor", "citeyear",
    "citeyearpar", "Citep", "Citet", "Citealp", "Citealt", "Citeauthor",
    "parencite", "Parencite", "textcite", "Textcite", "autocite", "Autocite",
    "footcite", "footcitetext", "smartcite", "Smartcite", "supercite",
    "fullcite", "footfullcite", "nocite", "citenum", "citetitle", "citedate",
    "cites", "Cites", "parencites", "Parencites", "textcites", "Textcites",
    "autocites", "Autocites", "footcites", "footcitetexts", "smartcites",
    "Smartcites", "supercites",
}


def _rewrite_citation_text(text: str, key_map: dict[str, str]) -> str:
    """Rewrite citation arguments only; retain options, spacing and key order."""
    edits: list[tuple[int, int, str]] = []
    i = 0
    while i < len(text):
        if text[i] == "%":
            end = text.find("\n", i)
            i = end if end >= 0 else len(text)
            continue
        if text[i] != "\\":
            i += 1
            continue
        match = _COMMAND.match(text, i)
        if match is None:
            i += 1
            continue
        i = match.end()
        command = match.group(1).rstrip("*")
        literal_end = _literal_end(text, match)
        if literal_end is not None:
            i = literal_end
            continue
        if command in {"url", "path", "href"}:
            # The URL argument may contain percent escapes or literal cite text.
            p = _space(text, i)
            end = _group(text, p)
            if end is not None:
                i = end
            continue
        if command not in _CITE_COMMANDS:
            continue
        multiple = command.endswith("s")
        while True:
            p = _space(text, i)
            # biblatex multicites may have global notes in parentheses.
            while p < len(text) and text[p] in ("[(" if multiple else "["):
                end = _group(text, p, text[p], "]" if text[p] == "[" else ")")
                if end is None:
                    break
                p = _space(text, end)
            end = _group(text, p)
            if end is None:
                break
            body = text[p + 1:end - 1]
            # A key list is flat; malformed/nested arguments remain intact.
            if "{" not in body and "}" not in body:
                rewritten = re.sub(
                    r"[^,\s]+",
                    lambda key: key_map.get(key.group(), key.group()),
                    body,
                )
                edits.append((p + 1, end - 1, rewritten))
            i = end
            if not multiple:
                break
    parts = []
    start = 0
    for start_edit, end_edit, replacement in edits:
        parts.extend((text[start:start_edit], replacement))
        start = end_edit
    parts.append(text[start:])
    return "".join(parts)


def build_element_tables(
    arxiv_id: str, nodes_by_version: dict[int, list[Node]]
) -> tuple[list[ElementRow], list[HierarchyRow]]:
    """Share element IDs across versions and retain version-specific parent edges."""
    return build_element_table(arxiv_id, nodes_by_version)


def build_element_table(
    arxiv_id: str, nodes_by_version: dict[int, list[Node]]
) -> tuple[list[ElementRow], list[HierarchyRow]]:
    """Build schema rows from canonicalized nodes, comparing exact kind/content.

    Every occurrence contributes its edge, including repeated occurrences.
    HierarchyDeduplicationWarning flags lossless ambiguities for review.
    Invalid local IDs or missing parents raise ValueError.
    """
    if not isinstance(arxiv_id, str) or not arxiv_id.strip():
        raise ValueError("arxiv_id must be a nonempty str")
    if any(not isinstance(version, int) or isinstance(version, bool) for version in nodes_by_version):
        raise TypeError("version must be an int")
    # Validate the occurrence trees before content deduplication can obscure
    # their identities. Parent order need not be preorder for this API.
    for version, nodes in nodes_by_version.items():
        parent_by_id = {}
        for node in nodes:
            if node.local_id in parent_by_id:
                raise ValueError(f"Duplicate local ID {node.local_id!r} in version {version}")
            parent_by_id[node.local_id] = node.parent_local_id
        complete = set()
        for local_id in parent_by_id:
            path = set()
            current = local_id
            while current is not None and current not in complete:
                if current not in parent_by_id:
                    raise ValueError(f"Missing parent {current!r} in version {version}")
                if current in path:
                    raise ValueError(f"Parent cycle at {current!r} in version {version}")
                path.add(current)
                current = parent_by_id[current]
            complete.update(path)
    elements: list[ElementRow] = []
    hierarchy: list[HierarchyRow] = []
    shared: dict[tuple[str, str], str] = {}
    local_ids: dict[tuple[int, str], str] = {}
    content_kinds: dict[str, set[str]] = {}
    for version in sorted(nodes_by_version):
        for node in nodes_by_version[version]:
            local_key = (version, node.local_id)
            identity = (node.kind, node.content)
            if identity not in shared:
                kinds = content_kinds.setdefault(node.content, set())
                if kinds and node.kind not in kinds:
                    warnings.warn("Identical content occurs with different node kinds; keeping separate elements",
                                  HierarchyDeduplicationWarning, stacklevel=2)
                kinds.add(node.kind)
                element_id = f"{arxiv_id}#e{len(elements) + 1:06d}"
                shared[identity] = element_id
                elements.append(ElementRow(element_id=element_id, arxiv_id=arxiv_id,
                                           kind=node.kind, content=node.content))
            local_ids[local_key] = shared[identity]

    parents: dict[tuple[int, str], set[str]] = {}
    for version in sorted(nodes_by_version):
        for node in nodes_by_version[version]:
            if node.parent_local_id is None:
                continue
            parent_key = (version, node.parent_local_id)
            if parent_key not in local_ids:
                raise ValueError(f"Missing parent {node.parent_local_id!r} in version {version}")
            element_id = local_ids[(version, node.local_id)]
            parent_id = local_ids[parent_key]
            previous = parents.setdefault((version, element_id), set())
            if previous and parent_id not in previous:
                warnings.warn(f"Element {element_id} has multiple parents in version {version}; retaining all edges",
                              HierarchyDeduplicationWarning, stacklevel=2)
            if element_id == parent_id:
                warnings.warn(f"Deduplication creates a self-edge for {element_id} in version {version}",
                              HierarchyDeduplicationWarning, stacklevel=2)
            previous.add(parent_id)
            hierarchy.append(HierarchyRow(arxiv_id=arxiv_id, version=version,
                                          element_id=element_id, parent_id=parent_id))
    return elements, hierarchy
