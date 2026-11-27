"""Match citation titles within a single publication."""
import re
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import unquote

ID = r"(?:\d{4}[.-]\d{4,5}|[a-z][a-z0-9.-]*/\d{7})(?:v\d+)?"
PREFIX = (r"(?:https?://)?(?:(?:www|export)\.)?arxiv\.org/(?:abs|pdf|html)/"
          r"|(?:https?://(?:dx\.)?doi\.org/)?10\.48550/arxiv\.|arxiv\s*:\s*")
DEFAULT_TITLE_SCORER = "rapidfuzz"
TITLE_SCORERS = ("sequence", DEFAULT_TITLE_SCORER)

def arxiv_id(value):
    text = unquote(str(value or "")).strip().lower()
    text = re.sub(r"^(?:" + PREFIX + ")", "", text)
    text = re.split(r"[?#]", text)[0].removesuffix(".pdf").rstrip("/")
    if not re.fullmatch(ID, text):
        raise ValueError(f"ID arXiv không hợp lệ: {value!r}")
    text = re.sub(r"v\d+$", "", text)
    return text if "/" in text else text.replace("-", ".")


def normalize(value):
    text = str(value or "")
    text = re.sub(r"""\\['"`^~=.]\s*\{?([a-zA-Z])\}?""", r"\1", text)
    text = re.sub(r"\\([a-zA-Z]+)", lambda m: {
        "tex": "TeX", "latex": "LaTeX", "ae": "ae", "ss": "ss", "o": "o",
        "textit": "", "textbf": "", "emph": "", "mathrm": "", "text": "",
    }.get(m[1], m[1]), text, flags=re.I)
    text = unicodedata.normalize("NFKD", text.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(re.findall(r"[^\W_]+", text))


def sequence_title_similarity(first, second):
    return SequenceMatcher(None, first, second, autojunk=False).ratio()


def get_title_similarity(scorer):
    if scorer == "sequence":
        return sequence_title_similarity
    if scorer != "rapidfuzz":
        raise ValueError(f"Bộ so tiêu đề không hợp lệ: {scorer}")
    try:
        from rapidfuzz.fuzz import ratio
    except ModuleNotFoundError as exc:
        if exc.name != "rapidfuzz":
            raise
        raise ValueError("Thiếu RapidFuzz. Cài bằng python -m pip install rapidfuzz.") from exc

    def rapidfuzz_title_similarity(first, second):
        return ratio(first, second) / 100

    return rapidfuzz_title_similarity


def decision(status, method, reason, ranked=(), target=None):
    result = dict(label_status=status, match_method=method, reason=reason,
                  groundtruth_arxiv_id=target, match_score=ranked[0][1] if ranked else None)
    for n, (ident, value) in enumerate(ranked[:2], 1):
        result[f"candidate_{n}_arxiv_id"] = ident
        result[f"candidate_{n}_score"] = value
    return result


def match(row, candidates, threshold, margin, *, title_similarity=None):
    if not candidates:
        return decision("no_match", "empty_candidates", "confirmed_empty_candidates")

    title = normalize(row.get("title"))
    if not title:
        return decision("no_match", "title_similarity", "missing_bibtex_title")

    if title_similarity is None:
        title_similarity = get_title_similarity(DEFAULT_TITLE_SCORER)
    scored = []
    incomplete = False
    for ident, variants in candidates.items():
        usable = [variant for variant in variants if variant]
        if not usable:
            incomplete = True
            continue
        value = max(title_similarity(title, variant) for variant in usable)
        scored.append((ident, value))
    if not scored:
        return decision("no_match", "title_similarity", "missing_candidate_titles")

    scored.sort(key=lambda item: (-item[1], item[0]))
    ranked = scored[:2]
    ident, value = ranked[0]
    if value < threshold:
        return decision("no_match", "title_similarity", "low_title_similarity", ranked)
    if len(ranked) > 1 and (value == ranked[1][1] or value - ranked[1][1] < margin):
        return decision("no_match", "title_similarity", "ambiguous_candidates", ranked)
    if incomplete:
        return decision("no_match", "title_similarity", "incomplete_candidate_titles", ranked)
    return decision("matched", "title_similarity", "title_similarity_above_threshold", ranked, ident)
