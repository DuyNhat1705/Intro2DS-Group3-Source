"""Metadata feature extraction and dataset partitioning for Reference Matching (Milestone 3).

Evaluated by: MRR@5 (Mean Reciprocal Rank at 5)
Repo: Intro2DS-Group3-Source

Provides:
1. Author, Year, and Identifier feature extraction functions (atomic and composite).
2. Robust publication-level dataset partitioning (train / valid / test) avoiding data leakage.
3. Candidate pair matrix constructor (build_pair_table) supporting pluggable feature functions.
4. Parquet / JSON IO and CLI for pipeline execution and self-testing.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants & Label Schema Configuration
# ---------------------------------------------------------------------------

LABEL_COL_ARXIV_ID = "arxiv_id"
LABEL_COL_CITATION_KEY = "citation_key"
LABEL_COL_GROUNDTRUTH = "groundtruth_arxiv_id"
LABEL_COL_SOURCE = "label_source"

VALID_PARTITIONS = {"train", "valid", "test"}

FEATURE_DOCS: dict[str, str] = {
    # Author features (8)
    "first_author_match": "1.0 if normalized last name of first author matches exactly between BibTeX and candidate reference; strongest single author signal.",
    "author_overlap_ratio": "Jaccard similarity |A ∩ B| / |A ∪ B| of normalized author surnames; measures overall author list consistency.",
    "author_overlap_min_ratio": "Overlap normalized by min(|A|, |B|); protects against truncation when BibTeX or reference cuts authors with 'et al.'.",
    "last_author_match": "1.0 if normalized last name of last author matches; useful for lab/senior PI matching.",
    "first_author_initial_match": "1.0 if first author surname matches AND both have given name with matching first initial; disambiguates common surnames.",
    "author_count_diff": "Absolute difference in author count |nA - nB| (default 99.0 if missing one side); detects co-author list completeness discrepancies.",
    "has_bib_authors": "1.0 if BibTeX entry contains at least one parsed author, else 0.0; indicators of metadata presence.",
    "has_ref_authors": "1.0 if reference entry contains at least one parsed author, else 0.0; indicators of reference metadata completeness.",
    # Year features (5)
    "year_diff": "Absolute difference |year_bib - year_ref| in publication/submission years (default 99.0 if missing); strong sanity filter.",
    "is_same_year": "1.0 if both years are present and identical, else 0.0; exact publication year match indicator.",
    "year_missing": "1.0 if either BibTeX or reference lacks a valid year, else 0.0; indicator for imputation/missingness handling.",
    "year_diff_signed": "Signed difference (year_bib - year_ref); captures arXiv preprint vs formal conference/journal latency (default 0.0 if missing).",
    "year_within_1": "1.0 if absolute year difference <= 1; tolerant to 1-year preprint submission vs publication latency.",
    # ID features (4)
    "doi_match": "1.0 if both BibTeX and reference have a normalized DOI and they are identical; definitive exact match when present.",
    "arxiv_hint_match": "1.0 if arXiv ID extracted/hinted in BibTeX entry matches candidate ref_arxiv_id; highly confident match.",
    "doi_arxiv_match": "1.0 if BibTeX DOI equals '10.48550/arxiv.<ref_arxiv_id>'; matches standard arXiv DOI format.",
    "arxiv_in_url_match": "1.0 if candidate ref_arxiv_id appears inside the BibTeX url field; detects explicit arXiv links.",
}

# Surnames prefixes in academic citations
PREFIXES = {
    "van", "von", "de", "del", "da", "di", "le", "la", "bin", "ibn", "der", "den", "du", "dos", "das"
}


# ---------------------------------------------------------------------------
# Helper Functions: Robust Field Extraction and Type Conversions
# ---------------------------------------------------------------------------

def _pick(d: Any, *names: str, default: Any = None) -> Any:
    """Safely extract the first existing non-null, non-NaN field from a mapping or Series."""
    if d is None:
        return default
    if isinstance(d, (dict, Mapping)):
        for name in names:
            if name in d:
                val = d[name]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return val
        return default
    for name in names:
        if hasattr(d, "get"):
            val = d.get(name)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return val
        elif hasattr(d, name):
            val = getattr(d, name)
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return val
        elif hasattr(d, "__getitem__"):
            try:
                val = d[name]
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    return val
            except (KeyError, IndexError, TypeError):
                pass
    return default


def _as_list(val: Any) -> list[str]:
    """Safely convert various representations of string sequences to a list of strings.

    Handles None, NaN, list, tuple, np.ndarray, and BibTeX strings ('A and B').
    Avoids truthiness check directly on numpy arrays.
    """
    if val is None:
        return []
    if isinstance(val, float) and math.isnan(val):
        return []
    if isinstance(val, (list, tuple)):
        res = []
        for x in val:
            if x is not None and not (isinstance(x, float) and math.isnan(x)):
                s = str(x).strip()
                if s:
                    res.append(s)
        return res
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes, dict)):
        res = []
        for x in val:
            if x is not None and not (isinstance(x, float) and math.isnan(x)):
                s = str(x).strip()
                if s:
                    res.append(s)
        return res
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if " and " in s:
            return [p.strip() for p in s.split(" and ") if p.strip()]
        if ";" in s:
            return [p.strip() for p in s.split(";") if p.strip()]
        return [s]
    return [str(val).strip()]


def _parse_year(val: Any) -> int | None:
    """Safely extract an integer year (1900-2099) from int, float, str, or datetime."""
    if val is None:
        return None
    if isinstance(val, float):
        if math.isnan(val):
            return None
        return int(val)
    if isinstance(val, int):
        return int(val)
    s = str(val).strip()
    if not s:
        return None
    m = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    if m:
        return int(m.group(1))
    return None


# ---------------------------------------------------------------------------
# Text & Identifier Normalization
# ---------------------------------------------------------------------------

def clean_latex_name(s: str) -> str:
    """Normalize LaTeX markup, accents, ligatures, and unicode characters in names."""
    if not s:
        return ""
    # Standard ligatures and symbols
    s = s.replace(r"\ss", "ss").replace(r"{\ss}", "ss")
    s = s.replace(r"\AA", "A").replace(r"\aa", "a")
    s = s.replace(r"\AE", "AE").replace(r"\ae", "ae")
    s = s.replace(r"\O", "O").replace(r"\o", "o").replace(r"{\o}", "o")
    s = s.replace(r"\L", "L").replace(r"\l", "l")
    # Commands like \c{c}, \'e, \"{u}, \v{s}, \~{a}
    s = re.sub(r"\\[a-zA-Z]+\{([a-zA-Z])\}", r"\1", s)
    s = re.sub(r"\\[\'\"`^~=.\u007fcdbkrutHv]\{?([a-zA-Z])\}?", r"\1", s)
    s = re.sub(r"[{}\\]", "", s)
    # Unicode NFKD decomposition to strip combining accents (Erdős -> Erdos)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.strip()


def parse_author_name(raw_name: str) -> tuple[str, str, str]:
    """Parse author string into (normalized_last_name, normalized_first_name, first_initial).

    Handles:
    - 'Last, First' vs 'First Last'
    - Compound surnames with prefixes: van, von, de, del, etc.
    - Omission of 'et al' / 'others'
    """
    s = clean_latex_name(raw_name)
    if not s:
        return "", "", ""

    s_lower = s.lower().strip()
    if s_lower in {"et al", "et al.", "others", "and others"}:
        return "", "", ""

    if "," in s:
        parts = s.split(",", 1)
        last_str = parts[0].strip()
        first_str = parts[1].strip()
    else:
        tokens = s.split()
        if len(tokens) == 0:
            return "", "", ""
        if len(tokens) == 1:
            last_str = tokens[0]
            first_str = ""
        else:
            idx = len(tokens) - 1
            while idx > 0 and tokens[idx - 1].lower() in PREFIXES:
                idx -= 1
            last_str = " ".join(tokens[idx:])
            first_str = " ".join(tokens[:idx])

    last_norm = re.sub(r"[^\w\s]", "", last_str).strip().lower()
    last_norm = re.sub(r"\s+", " ", last_norm)

    first_clean = re.sub(r"[^\w\s]", "", first_str).strip().lower()
    first_clean = re.sub(r"\s+", " ", first_clean)

    initial = ""
    for ch in first_clean:
        if ch.isalpha():
            initial = ch
            break

    return last_norm, first_clean, initial


def normalize_doi(doi: Any) -> str:
    """Normalize DOI string by lowercasing and stripping URL/protocol prefixes."""
    if doi is None:
        return ""
    if isinstance(doi, float) and math.isnan(doi):
        return ""
    s = str(doi).strip().lower()
    s = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", s)
    s = re.sub(r"^doi:\s*", "", s)
    return s.strip()


def normalize_arxiv_id(aid: Any) -> str:
    """Normalize arXiv identifier to canonical short form (lowercase, strip version suffix).

    Handles both new style (YYMM.NNNNN) and legacy style (hep-th/9901001), as well as
    embedded URLs and arXiv: prefixes.
    """
    if aid is None:
        return ""
    if isinstance(aid, float) and math.isnan(aid):
        return ""
    s = str(aid).strip().lower()
    # Strip version suffix if at the very end
    s = re.sub(r"v\d+$", "", s)
    # Extract canonical arXiv ID pattern
    m = re.search(
        r"(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*|10\.48550/arxiv\.)?([a-z\-]+(?:\.[a-z\-]+)?/\d{7}|\d{4}\.\d{4,5})",
        s,
    )
    if m:
        return m.group(1).lower()
    return s.strip()


# ---------------------------------------------------------------------------
# Pre-caching: Prepare BibTeX and Reference Entries
# ---------------------------------------------------------------------------

def prepare_bib(entry: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Precompute normalized metadata fields for a BibTeX entry. Idempotent."""
    if isinstance(entry, dict) and entry.get("_prepared"):
        return entry
    d = dict(entry)
    if d.get("_prepared"):
        return d

    authors_raw = _as_list(_pick(d, "authors", "author", "ref_authors"))
    parsed_authors = [parse_author_name(a) for a in authors_raw]
    valid_authors = [p for p in parsed_authors if p[0]]

    last_names = {p[0] for p in valid_authors}
    first_author_last = valid_authors[0][0] if valid_authors else ""
    first_author_initial = valid_authors[0][2] if valid_authors else ""
    last_author_last = valid_authors[-1][0] if valid_authors else ""

    year_val = _parse_year(_pick(d, "year", "ref_year", "ref_submission_date"))
    doi_val = normalize_doi(_pick(d, "doi", "ref_doi"))

    hint_candidate = _pick(d, "arxiv_id_hint", "eprint", "arxiv", "arxiv_id")
    if not hint_candidate:
        if doi_val and "10.48550/arxiv." in doi_val:
            hint_candidate = doi_val
        elif _pick(d, "url"):
            hint_candidate = _pick(d, "url")
        elif _pick(d, "journal"):
            hint_candidate = _pick(d, "journal")
    arxiv_hint = normalize_arxiv_id(hint_candidate) if hint_candidate else ""

    url_val = str(_pick(d, "url") or "").lower()

    d["_authors"] = authors_raw
    d["_author_last_names"] = last_names
    d["_first_author_last"] = first_author_last
    d["_first_author_initial"] = first_author_initial
    d["_last_author_last"] = last_author_last
    d["_author_count"] = len(valid_authors)
    d["_year"] = year_val
    d["_doi"] = doi_val
    d["_arxiv_hint"] = arxiv_hint
    d["_url"] = url_val
    d["_prepared"] = True
    return d


def prepare_ref(ref: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    """Precompute normalized metadata fields for a reference entry. Idempotent."""
    if isinstance(ref, dict) and ref.get("_prepared"):
        return ref
    d = dict(ref)
    if d.get("_prepared"):
        return d

    authors_raw = _as_list(_pick(d, "ref_authors", "authors", "author"))
    parsed_authors = [parse_author_name(a) for a in authors_raw]
    valid_authors = [p for p in parsed_authors if p[0]]

    last_names = {p[0] for p in valid_authors}
    first_author_last = valid_authors[0][0] if valid_authors else ""
    first_author_initial = valid_authors[0][2] if valid_authors else ""
    last_author_last = valid_authors[-1][0] if valid_authors else ""

    year_val = _parse_year(
        _pick(d, "ref_submission_date", "submission_date", "year", "ref_year", "publicationDate")
    )
    doi_val = normalize_doi(_pick(d, "ref_doi", "doi"))
    ref_arxiv_id = normalize_arxiv_id(_pick(d, "ref_arxiv_id", "arxiv_id"))

    d["_authors"] = authors_raw
    d["_author_last_names"] = last_names
    d["_first_author_last"] = first_author_last
    d["_first_author_initial"] = first_author_initial
    d["_last_author_last"] = last_author_last
    d["_author_count"] = len(valid_authors)
    d["_year"] = year_val
    d["_doi"] = doi_val
    d["_arxiv_id"] = ref_arxiv_id
    d["_prepared"] = True
    return d


# ---------------------------------------------------------------------------
# Atomic Metadata Feature Functions
# ---------------------------------------------------------------------------

def first_author_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_first_author_last"] or not r["_first_author_last"]:
        return 0.0
    return 1.0 if b["_first_author_last"] == r["_first_author_last"] else 0.0


def author_overlap_ratio(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    s1 = b["_author_last_names"]
    s2 = r["_author_last_names"]
    if not s1 or not s2:
        return 0.0
    union_len = len(s1 | s2)
    if union_len == 0:
        return 0.0
    return float(len(s1 & s2) / union_len)


def author_overlap_min_ratio(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    s1 = b["_author_last_names"]
    s2 = r["_author_last_names"]
    min_len = min(len(s1), len(s2))
    if min_len == 0:
        return 0.0
    return float(len(s1 & s2) / min_len)


def last_author_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_last_author_last"] or not r["_last_author_last"]:
        return 0.0
    return 1.0 if b["_last_author_last"] == r["_last_author_last"] else 0.0


def first_author_initial_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_first_author_last"] or not r["_first_author_last"]:
        return 0.0
    if b["_first_author_last"] != r["_first_author_last"]:
        return 0.0
    if b["_first_author_initial"] and r["_first_author_initial"]:
        return 1.0 if b["_first_author_initial"] == r["_first_author_initial"] else 0.0
    return 0.0


def author_count_diff(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if b["_author_count"] == 0 or r["_author_count"] == 0:
        return 99.0
    return float(abs(b["_author_count"] - r["_author_count"]))


def has_bib_authors(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    return 1.0 if b["_author_count"] > 0 else 0.0


def has_ref_authors(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    return 1.0 if r["_author_count"] > 0 else 0.0


def year_diff(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if b["_year"] is None or r["_year"] is None:
        return 99.0
    return float(abs(b["_year"] - r["_year"]))


def is_same_year(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if b["_year"] is None or r["_year"] is None:
        return 0.0
    return 1.0 if b["_year"] == r["_year"] else 0.0


def year_missing(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    return 1.0 if (b["_year"] is None or r["_year"] is None) else 0.0


def year_diff_signed(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if b["_year"] is None or r["_year"] is None:
        return 0.0
    return float(b["_year"] - r["_year"])


def year_within_1(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if b["_year"] is None or r["_year"] is None:
        return 0.0
    return 1.0 if abs(b["_year"] - r["_year"]) <= 1 else 0.0


def doi_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_doi"] or not r["_doi"]:
        return 0.0
    return 1.0 if b["_doi"] == r["_doi"] else 0.0


def arxiv_hint_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_arxiv_hint"] or not r["_arxiv_id"]:
        return 0.0
    return 1.0 if b["_arxiv_hint"] == r["_arxiv_id"] else 0.0


def doi_arxiv_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_doi"] or not r["_arxiv_id"]:
        return 0.0
    expected_doi = f"10.48550/arxiv.{r['_arxiv_id']}"
    return 1.0 if b["_doi"] == expected_doi else 0.0


def arxiv_in_url_match(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> float:
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    if not b["_url"] or not r["_arxiv_id"]:
        return 0.0
    return 1.0 if r["_arxiv_id"] in b["_url"] else 0.0


# ---------------------------------------------------------------------------
# Composite Grouped Feature Functions
# ---------------------------------------------------------------------------

def compute_author_features(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float]:
    """Compute all 8 author-based metadata similarity features."""
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    return {
        "first_author_match": first_author_match(b, r),
        "author_overlap_ratio": author_overlap_ratio(b, r),
        "author_overlap_min_ratio": author_overlap_min_ratio(b, r),
        "last_author_match": last_author_match(b, r),
        "first_author_initial_match": first_author_initial_match(b, r),
        "author_count_diff": author_count_diff(b, r),
        "has_bib_authors": has_bib_authors(b, r),
        "has_ref_authors": has_ref_authors(b, r),
    }


def compute_year_features(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float]:
    """Compute all 5 publication/submission year metadata features."""
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    return {
        "year_diff": year_diff(b, r),
        "is_same_year": is_same_year(b, r),
        "year_missing": year_missing(b, r),
        "year_diff_signed": year_diff_signed(b, r),
        "year_within_1": year_within_1(b, r),
    }


def compute_id_features(bib: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float]:
    """Compute all 4 DOI and arXiv identifier matching features."""
    b = bib if bib.get("_prepared") else prepare_bib(bib)
    r = ref if ref.get("_prepared") else prepare_ref(ref)
    return {
        "doi_match": doi_match(b, r),
        "arxiv_hint_match": arxiv_hint_match(b, r),
        "doi_arxiv_match": doi_arxiv_match(b, r),
        "arxiv_in_url_match": arxiv_in_url_match(b, r),
    }


DEFAULT_FEATURE_FNS = [
    compute_author_features,
    compute_year_features,
    compute_id_features,
]


# ---------------------------------------------------------------------------
# Dataset Partitioning (Publication-Level Split)
# ---------------------------------------------------------------------------

def split_publications(
    labeled_pubs: Mapping[str, Any] | Iterable[tuple[Any, ...]] | pd.DataFrame,
    *,
    k: int = 1,
    seed: int = 42,
    min_eval_entries: int = 5,
) -> dict[str, str]:
    """Partition publications into train, valid, and test sets.

    Guarantees:
    - Split is by PUBLICATION (all citations of a publication stay together, zero leakage).
    - Exactly k manual and k auto publications in Test.
    - Exactly k manual and k auto publications in Valid.
    - All remaining publications assigned to Train.
    - Strictly deterministic given seed, completely invariant to input collection order.

    Parameters
    ----------
    labeled_pubs : Mapping, Iterable of tuples, or DataFrame
        Maps arxiv_id to 'manual'|'auto', or mapping to dict with {'source': ..., 'count': ...}.
    k : int, default 1
        Number of manual and auto publications per evaluation set.
        NOTE: According to course project specification, k is the number of members in the group!
        The default k=1 is only for initial team convention. The group MUST set k to their member count.
    seed : int, default 42
        Random seed for deterministic shuffling.
    min_eval_entries : int, default 5
        Target minimum number of labeled entries per paper for valid/test.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")

    # Standardize input to dict: arxiv_id -> {"source": "manual"|"auto", "count": int}
    pub_dict: dict[str, dict[str, Any]] = {}
    if isinstance(labeled_pubs, pd.DataFrame):
        for row in labeled_pubs.to_dict(orient="records"):
            pid = str(row[LABEL_COL_ARXIV_ID])
            src = str(row.get(LABEL_COL_SOURCE, "auto")).lower()
            cnt = int(row.get("n_entries", row.get("count", 0)))
            pub_dict[pid] = {"source": src, "count": cnt}
    elif isinstance(labeled_pubs, Mapping):
        for pid, val in labeled_pubs.items():
            pid_str = str(pid)
            if isinstance(val, str):
                pub_dict[pid_str] = {"source": val.lower(), "count": 0}
            elif isinstance(val, Mapping):
                src = str(val.get("source", val.get(LABEL_COL_SOURCE, "auto"))).lower()
                cnt = int(val.get("count", val.get("n_entries", 0)))
                pub_dict[pid_str] = {"source": src, "count": cnt}
            elif isinstance(val, (tuple, list)):
                src = str(val[0]).lower()
                cnt = int(val[1]) if len(val) > 1 else 0
                pub_dict[pid_str] = {"source": src, "count": cnt}
            else:
                pub_dict[pid_str] = {"source": str(val).lower(), "count": 0}
    elif isinstance(labeled_pubs, Iterable):
        for item in labeled_pubs:
            if isinstance(item, (tuple, list)):
                pid_str = str(item[0])
                src = str(item[1]).lower()
                cnt = int(item[2]) if len(item) > 2 else 0
                pub_dict[pid_str] = {"source": src, "count": cnt}

    manual_all = sorted([p for p, info in pub_dict.items() if info["source"] == "manual"])
    auto_all = sorted([p for p, info in pub_dict.items() if info["source"] == "auto"])

    required = 2 * k
    if len(manual_all) < required or len(auto_all) < required:
        raise ValueError(
            f"Insufficient publications for split with k={k}: need at least {required} manual "
            f"(got {len(manual_all)}) and {required} auto (got {len(auto_all)}). "
            f"Deficit: manual={max(0, required - len(manual_all))}, auto={max(0, required - len(auto_all))}."
        )

    def select_eval_pool(candidates: list[str]) -> list[str]:
        high_entry = [p for p in candidates if pub_dict[p]["count"] >= min_eval_entries]
        if len(high_entry) >= required:
            pool = list(high_entry)
        else:
            if any(pub_dict[p]["count"] > 0 for p in candidates):
                warnings.warn(
                    f"Only {len(high_entry)} papers have >= {min_eval_entries} entries (need {required}). "
                    "Relaxing min_eval_entries requirement."
                )
            pool = list(candidates)
        return pool

    manual_pool = select_eval_pool(manual_all)
    auto_pool = select_eval_pool(auto_all)

    # Deterministic shuffling
    rng = random.Random(seed)
    rng.shuffle(manual_pool)
    rng.shuffle(auto_pool)

    test_manual = manual_pool[:k]
    valid_manual = manual_pool[k:2 * k]

    test_auto = auto_pool[:k]
    valid_auto = auto_pool[k:2 * k]

    test_set = set(test_manual + test_auto)
    valid_set = set(valid_manual + valid_auto)
    train_set = set(pub_dict.keys()) - test_set - valid_set

    if len(train_set) == 0:
        warnings.warn("Train partition is empty after selecting test and valid publications.")

    split_result: dict[str, str] = {}
    for p in pub_dict:
        if p in test_set:
            split_result[p] = "test"
        elif p in valid_set:
            split_result[p] = "valid"
        else:
            split_result[p] = "train"

    return split_result


# ---------------------------------------------------------------------------
# Candidate Pair Table Construction
# ---------------------------------------------------------------------------

def validate_label_schema(labels_df: pd.DataFrame) -> None:
    """Validate that labels_df contains required columns."""
    required = [LABEL_COL_ARXIV_ID, LABEL_COL_CITATION_KEY, LABEL_COL_GROUNDTRUTH, LABEL_COL_SOURCE]
    missing = [c for c in required if c not in labels_df.columns]
    if missing:
        raise ValueError(
            f"Labels DataFrame is missing required columns: {missing}. "
            f"Found columns: {list(labels_df.columns)}. "
            f"Expected schema: {required}"
        )


def build_pair_table(
    bibtex_df: pd.DataFrame,
    references_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    partition_of: Mapping[str, str] | Callable[[str], str],
    feature_fns: Sequence[Callable[[Mapping[str, Any], Mapping[str, Any]], dict[str, float]]] | None = None,
    max_neg_per_query: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct full candidate pair matrix with computed metadata features.

    Parameters
    ----------
    bibtex_df : pd.DataFrame
        Milestone 2 unified bibtex table (must contain arxiv_id, citation_key, etc.).
    references_df : pd.DataFrame
        Milestone 1 references table (must contain arxiv_id, ref_arxiv_id, etc.).
    labels_df : pd.DataFrame
        Groundtruth labels (must contain arxiv_id, citation_key, groundtruth_arxiv_id, label_source).
    partition_of : Mapping or Callable
        Maps arxiv_id to 'train'|'valid'|'test'.
    feature_fns : Sequence of callables, optional
        List of feature functions returning dict[str, float].
        Defaults to [compute_author_features, compute_year_features, compute_id_features].
    max_neg_per_query : int, optional
        Downsample negatives per query for 'train' partition only. Default None preserves all.
    seed : int, default 42
        Deterministic random seed for negative sampling.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        (pairs_df, stats_summary)
    """
    validate_label_schema(labels_df)

    if feature_fns is None:
        feature_fns = DEFAULT_FEATURE_FNS

    # 1. Pre-cache and prepare BibTeX entries: (arxiv_id, citation_key) -> prepared_bib
    bib_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for row in bibtex_df.to_dict(orient="records"):
        aid = str(_pick(row, "arxiv_id"))
        ckey = str(_pick(row, "citation_key"))
        key = (aid, ckey)
        if key not in bib_cache:
            bib_cache[key] = prepare_bib(row)

    # 2. Pre-cache and prepare Reference entries: arxiv_id -> list of deduped prepared_ref
    ref_groups: dict[str, list[dict[str, Any]]] = {}
    for row in references_df.to_dict(orient="records"):
        aid = str(_pick(row, "arxiv_id"))
        raw_ref_aid = _pick(row, "ref_arxiv_id", "arxiv_id")
        ref_aid = normalize_arxiv_id(raw_ref_aid)
        if not ref_aid:
            continue
        prepared = prepare_ref(row)
        if aid not in ref_groups:
            ref_groups[aid] = []
        # Deduplicate candidate references by canonical ref_arxiv_id within publication
        if not any(r["_arxiv_id"] == ref_aid for r in ref_groups[aid]):
            ref_groups[aid].append(prepared)

    # 3. Construct candidate pairs for each labeled entry
    rows_out: list[dict[str, Any]] = []
    skipped_bib_not_found = 0
    gt_not_in_refs = 0
    all_neg_qids = 0

    partition_stats = {
        p: {"n_pubs": set(), "n_qids": set(), "n_pairs": 0, "n_pos": 0}
        for p in VALID_PARTITIONS
    }

    for label_row in labels_df.to_dict(orient="records"):
        aid = str(label_row[LABEL_COL_ARXIV_ID])
        ckey = str(label_row[LABEL_COL_CITATION_KEY])
        raw_gt = label_row[LABEL_COL_GROUNDTRUTH]
        source = str(label_row.get(LABEL_COL_SOURCE, "unknown"))

        bib_key = (aid, ckey)
        if bib_key not in bib_cache:
            skipped_bib_not_found += 1
            continue

        bib_prepared = bib_cache[bib_key]
        cand_refs = ref_groups.get(aid, [])

        # Groundtruth normalization
        gt_norm = None
        if raw_gt is not None and not (isinstance(raw_gt, float) and math.isnan(raw_gt)):
            gt_str = str(raw_gt).strip()
            if gt_str:
                gt_norm = normalize_arxiv_id(gt_str)

        if callable(partition_of):
            part = str(partition_of(aid))
        elif isinstance(partition_of, Mapping):
            part = str(partition_of.get(aid, "train"))
        else:
            part = "train"

        if part not in VALID_PARTITIONS:
            part = "train"

        qid = f"{aid}::{ckey}"

        # Check if groundtruth is present in candidate references
        if gt_norm is not None:
            has_gt = any(r["_arxiv_id"] == gt_norm for r in cand_refs)
            if not has_gt:
                gt_not_in_refs += 1

        # Build candidate pairs
        query_candidates: list[tuple[dict[str, Any], int]] = []
        pos_found = 0
        for r in cand_refs:
            is_pos = 1 if (gt_norm is not None and r["_arxiv_id"] == gt_norm) else 0
            if is_pos:
                pos_found += 1
            query_candidates.append((r, is_pos))

        if pos_found == 0:
            all_neg_qids += 1

        # Negative downsampling for train partition only
        if part == "train" and max_neg_per_query is not None and max_neg_per_query >= 0:
            pos_candidates = [c for c in query_candidates if c[1] == 1]
            neg_candidates = [c for c in query_candidates if c[1] == 0]
            if len(neg_candidates) > max_neg_per_query:
                qid_seed = (seed + (abs(hash(qid)) % 1000003)) & 0xFFFFFFFF
                local_rng = random.Random(qid_seed)
                sampled_negs = local_rng.sample(neg_candidates, max_neg_per_query)
                query_candidates = pos_candidates + sampled_negs

        # Feature extraction
        for r_prepared, is_pos in query_candidates:
            pair_data: dict[str, Any] = {
                "qid": qid,
                "arxiv_id": aid,
                "citation_key": ckey,
                "ref_arxiv_id": r_prepared["_arxiv_id"],
                "label": int(is_pos),
                "partition": part,
                "label_source": source,
            }
            for fn in feature_fns:
                feats = fn(bib_prepared, r_prepared)
                pair_data.update(feats)

            rows_out.append(pair_data)

            # Record stats
            partition_stats[part]["n_pubs"].add(aid)
            partition_stats[part]["n_qids"].add(qid)
            partition_stats[part]["n_pairs"] += 1
            if is_pos:
                partition_stats[part]["n_pos"] += 1

    # 4. Construct DataFrame and enforce schema/sorting
    if rows_out:
        pairs_df = pd.DataFrame(rows_out)
        # Ensure contiguous grouping per qid and deterministic partition sorting
        pairs_df.sort_values(
            by=["partition", "arxiv_id", "citation_key", "ref_arxiv_id"],
            inplace=True,
            ignore_index=True,
        )
    else:
        # Create empty DataFrame with standard columns
        dummy_bib = prepare_bib({"authors": [], "year": 2000})
        dummy_ref = prepare_ref({"ref_authors": [], "year": 2000})
        sample_feats: dict[str, float] = {}
        for fn in feature_fns:
            sample_feats.update(fn(dummy_bib, dummy_ref))

        empty_cols = [
            "qid", "arxiv_id", "citation_key", "ref_arxiv_id",
            "label", "partition", "label_source",
        ] + list(sample_feats.keys())
        pairs_df = pd.DataFrame(columns=empty_cols)

    # Format stats dictionary
    total_pairs = len(pairs_df)
    total_pos = sum(p["n_pos"] for p in partition_stats.values())
    stats = {
        "total_pairs": total_pairs,
        "total_qids": len({r["qid"] for r in rows_out}) if rows_out else 0,
        "total_publications": len({r["arxiv_id"] for r in rows_out}) if rows_out else 0,
        "by_partition": {
            part: {
                "n_pubs": len(info["n_pubs"]),
                "n_qids": len(info["n_qids"]),
                "n_pairs": info["n_pairs"],
                "n_pos": info["n_pos"],
                "pos_ratio": (info["n_pos"] / info["n_pairs"]) if info["n_pairs"] > 0 else 0.0,
            }
            for part, info in partition_stats.items()
        },
        "positive_ratio_overall": (total_pos / total_pairs) if total_pairs > 0 else 0.0,
        "all_negative_qids": all_neg_qids,
        "gt_not_in_refs": gt_not_in_refs,
        "skipped_bib_not_found": skipped_bib_not_found,
    }

    return pairs_df, stats


# ---------------------------------------------------------------------------
# I/O Utilities
# ---------------------------------------------------------------------------

def load_table(path: str | Path) -> pd.DataFrame:
    """Read Parquet, JSON, or CSV file/directory into a pandas DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")
    if p.is_dir() or p.suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return pd.read_json(p)
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return pd.read_parquet(p)


def collect_references_from_json(data_dir: str | Path = "data") -> pd.DataFrame:
    """Gather all individual data/*/references.json files into a single DataFrame."""
    root = Path(data_dir)
    records = []
    if not root.exists():
        return pd.DataFrame(columns=[
            "arxiv_id", "ref_arxiv_id", "ref_title", "ref_authors", "ref_submission_date", "ref_s2_id"
        ])
    for ref_file in root.glob("*/references.json"):
        try:
            with open(ref_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    records.extend(data)
        except Exception:
            continue
    if not records:
        return pd.DataFrame(columns=[
            "arxiv_id", "ref_arxiv_id", "ref_title", "ref_authors", "ref_submission_date", "ref_s2_id"
        ])
    return pd.DataFrame(records)


def save_pair_table(
    pairs_df: pd.DataFrame,
    split_dict: dict[str, str],
    out_dir: str | Path = "data/matching",
) -> tuple[Path, Path]:
    """Save pair table to Parquet with explicit PyArrow schema and split mapping to JSON.

    Parameters
    ----------
    pairs_df : pd.DataFrame
        Pair table with metadata and features.
    split_dict : dict[str, str]
        Mapping from arxiv_id to partition name.
    out_dir : str or Path, default 'data/matching'
        Target output directory.

    Returns
    -------
    tuple[Path, Path]
        (parquet_filepath, split_json_filepath)
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    parquet_file = out_path / "pairs.parquet"
    split_file = out_path / "split.json"

    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2, ensure_ascii=False)

    feature_cols = [
        c for c in pairs_df.columns
        if c not in {"qid", "arxiv_id", "citation_key", "ref_arxiv_id", "label", "partition", "label_source"}
    ]

    base_fields = [
        pa.field("qid", pa.string(), nullable=False),
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("ref_arxiv_id", pa.string(), nullable=False),
        pa.field("label", pa.int8(), nullable=False),
        pa.field("partition", pa.dictionary(pa.int8(), pa.string())),
        pa.field("label_source", pa.string()),
    ]
    feat_fields = [pa.field(fc, pa.float32()) for fc in feature_cols]
    schema = pa.schema(base_fields + feat_fields)

    if len(pairs_df) == 0:
        empty_table = pa.Table.from_batches([], schema=schema)
        pq.write_table(empty_table, parquet_file, compression="zstd")
    else:
        df = pairs_df.copy()
        df["qid"] = df["qid"].astype(str)
        df["arxiv_id"] = df["arxiv_id"].astype(str)
        df["citation_key"] = df["citation_key"].astype(str)
        df["ref_arxiv_id"] = df["ref_arxiv_id"].astype(str)
        df["label"] = df["label"].astype("int8")
        df["partition"] = df["partition"].astype(str).astype("category")
        df["label_source"] = df["label_source"].astype(str)
        for fc in feature_cols:
            df[fc] = df[fc].astype("float32")

        table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
        pq.write_table(table, parquet_file, compression="zstd")

    return parquet_file, split_file


# ---------------------------------------------------------------------------
# Self-Test Suite (10 Automated Verification Tests)
# ---------------------------------------------------------------------------

def _selftest() -> None:
    """Execute complete self-test suite covering all required specifications."""
    print("Running metadata & dataset split self-test suite...")

    # Test 1: first_author_match & normalization
    m1 = first_author_match({"authors": ["A. Vaswani"]}, {"ref_authors": ["Ashish Vaswani"]})
    assert m1 == 1.0, f"Expected 1.0, got {m1}"

    m2 = first_author_match({"authors": ["Smith, John"]}, {"ref_authors": ["John Smith"]})
    assert m2 == 1.0, f"Expected 1.0, got {m2}"

    m3 = first_author_match({"authors": ["Erdős"]}, {"ref_authors": ["Erdos"]})
    assert m3 == 1.0, f"Expected 1.0, got {m3}"

    m4 = first_author_match({"authors": [r"Fran\c{c}ois Chollet"]}, {"ref_authors": ["François Chollet"]})
    assert m4 == 1.0, f"Expected 1.0, got {m4}"

    m5 = first_author_match({"authors": ["Vaswani"]}, {"ref_authors": ["Smith"]})
    assert m5 == 0.0, f"Expected 0.0, got {m5}"

    b_none = {"authors": []}
    r_val = {"ref_authors": ["Vaswani"]}
    assert first_author_match(b_none, r_val) == 0.0
    assert has_bib_authors(b_none, r_val) == 0.0
    assert has_ref_authors(b_none, r_val) == 1.0
    print("  [1/10] first_author_match & normalization: PASSED")

    # Test 2: author_overlap_ratio & author_overlap_min_ratio
    bib_authors = ["Vaswani", "Shazeer", "Parmar"]
    ref_authors = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"]
    overlap = author_overlap_ratio({"authors": bib_authors}, {"ref_authors": ref_authors})
    assert abs(overlap - 0.75) < 1e-5, f"Expected 0.75, got {overlap}"

    bib_cut = ["Vaswani"]
    ref_8 = [
        "Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit",
        "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin",
    ]
    jaccard_cut = author_overlap_ratio({"authors": bib_cut}, {"ref_authors": ref_8})
    min_cut = author_overlap_min_ratio({"authors": bib_cut}, {"ref_authors": ref_8})
    assert abs(jaccard_cut - 0.125) < 1e-5, f"Expected 0.125, got {jaccard_cut}"
    assert abs(min_cut - 1.0) < 1e-5, f"Expected 1.0, got {min_cut}"
    print("  [2/10] author_overlap_ratio & min_ratio: PASSED")

    # Test 3: Year features
    y_feat1 = compute_year_features({"year": 2017}, {"ref_submission_date": "2017-06-12"})
    assert y_feat1["year_diff"] == 0.0
    assert y_feat1["is_same_year"] == 1.0
    assert y_feat1["year_missing"] == 0.0

    y_feat2 = compute_year_features({"year": 2018}, {"ref_submission_date": "2017"})
    assert y_feat2["year_diff"] == 1.0
    assert y_feat2["is_same_year"] == 0.0
    assert y_feat2["year_diff_signed"] == 1.0
    assert y_feat2["year_within_1"] == 1.0

    y_feat3 = compute_year_features({"year": None}, {"ref_submission_date": "2017"})
    assert y_feat3["year_diff"] == 99.0
    assert y_feat3["is_same_year"] == 0.0
    assert y_feat3["year_missing"] == 1.0
    assert y_feat3["year_diff_signed"] == 0.0

    y_feat4 = compute_year_features({"year": float("nan")}, {"ref_submission_date": None})
    assert y_feat4["year_diff"] == 99.0
    assert y_feat4["year_missing"] == 1.0
    print("  [3/10] year features & missingness handling: PASSED")

    # Test 4: Identifier matching
    id_feat1 = compute_id_features(
        {"doi": "https://doi.org/10.48550/arXiv.1706.03762"},
        {"ref_arxiv_id": "1706.03762"},
    )
    assert id_feat1["doi_arxiv_match"] == 1.0

    id_feat2 = compute_id_features(
        {"arxiv_id_hint": "1706.03762v5"},
        {"ref_arxiv_id": "1706.03762"},
    )
    assert id_feat2["arxiv_hint_match"] == 1.0

    id_feat3 = compute_id_features(
        {"arxiv_id_hint": "hep-th/9901001"},
        {"ref_arxiv_id": "hep-th/9901001v2"},
    )
    assert id_feat3["arxiv_hint_match"] == 1.0

    id_feat4 = compute_id_features(
        {"arxiv_id_hint": "1706.03762"},
        {"ref_arxiv_id": "1801.00001"},
    )
    assert id_feat4["arxiv_hint_match"] == 0.0

    id_feat5 = compute_id_features(
        {"doi": "10.1145/12345"},
        {"ref_doi": "10.1145/67890"},
    )
    assert id_feat5["doi_match"] == 0.0
    print("  [4/10] ID features & arXiv/DOI normalization: PASSED")

    # Test 5: numpy ndarray author robustness
    try:
        import numpy as np
        arr_bib = np.array(["Vaswani", "Shazeer"])
        arr_ref = np.array(["Ashish Vaswani", "Noam Shazeer"])
    except ImportError:
        arr_bib = ["Vaswani", "Shazeer"]
        arr_ref = ["Ashish Vaswani", "Noam Shazeer"]

    nd_match = first_author_match({"authors": arr_bib}, {"ref_authors": arr_ref})
    assert nd_match == 1.0
    nd_feats = compute_author_features({"authors": arr_bib}, {"ref_authors": arr_ref})
    assert nd_feats["author_overlap_ratio"] == 1.0
    print("  [5/10] numpy ndarray safety: PASSED")

    # Test 6: split_publications
    pubs = {f"man_{i}": "manual" for i in range(7)}
    pubs.update({f"aut_{j}": "auto" for j in range(6)})
    split1 = split_publications(pubs, k=1, seed=42)

    test_pubs = [p for p, s in split1.items() if s == "test"]
    valid_pubs = [p for p, s in split1.items() if s == "valid"]
    train_pubs = [p for p, s in split1.items() if s == "train"]

    assert len(test_pubs) == 2, f"Expected 2 test pubs, got {len(test_pubs)}"
    assert sum(1 for p in test_pubs if p.startswith("man_")) == 1
    assert sum(1 for p in test_pubs if p.startswith("aut_")) == 1

    assert len(valid_pubs) == 2, f"Expected 2 valid pubs, got {len(valid_pubs)}"
    assert sum(1 for p in valid_pubs if p.startswith("man_")) == 1
    assert sum(1 for p in valid_pubs if p.startswith("aut_")) == 1

    assert len(train_pubs) == 9
    assert set(split1.keys()) == set(pubs.keys())

    # Invariance to input order
    reversed_pubs = dict(reversed(list(pubs.items())))
    split2 = split_publications(reversed_pubs, k=1, seed=42)
    assert split1 == split2, "Split result must be strictly invariant to input dictionary order"

    # Deficit error
    try:
        split_publications({"man_0": "manual", "aut_0": "auto"}, k=1)
        assert False, "Should have raised ValueError on insufficient publications"
    except ValueError as e:
        assert "Insufficient publications" in str(e)
    print("  [6/10] split_publications determinism & validation: PASSED")

    # Test 7: build_pair_table structure & contiguity
    bib_df = pd.DataFrame([
        {"arxiv_id": "pub1", "citation_key": "e1", "title": "Paper 1", "authors": ["Vaswani"], "year": 2017},
        {"arxiv_id": "pub1", "citation_key": "e2", "title": "Paper 2", "authors": ["Devlin"], "year": 2018},
        {"arxiv_id": "pub1", "citation_key": "e3", "title": "Paper 3", "authors": ["He"], "year": 2016},
    ])
    ref_df = pd.DataFrame([
        {"arxiv_id": "pub1", "ref_arxiv_id": "ref_A", "ref_title": "Ref A", "ref_authors": ["Vaswani"], "ref_submission_date": "2017"},
        {"arxiv_id": "pub1", "ref_arxiv_id": "ref_B", "ref_title": "Ref B", "ref_authors": ["Devlin"], "ref_submission_date": "2018"},
        {"arxiv_id": "pub1", "ref_arxiv_id": "ref_C", "ref_title": "Ref C", "ref_authors": ["He"], "ref_submission_date": "2016"},
    ])
    lbl_df = pd.DataFrame([
        {"arxiv_id": "pub1", "citation_key": "e1", "groundtruth_arxiv_id": "ref_B", "label_source": "manual"},
        {"arxiv_id": "pub1", "citation_key": "e2", "groundtruth_arxiv_id": None, "label_source": "manual"},
    ])
    pairs_df, stats = build_pair_table(
        bib_df, ref_df, lbl_df,
        partition_of={"pub1": "valid"},
    )
    assert len(pairs_df) == 6, f"Expected 6 rows, got {len(pairs_df)}"
    # e1 should have exactly one label 1 (for ref_B)
    e1_labels = pairs_df[pairs_df["citation_key"] == "e1"]["label"].tolist()
    assert sum(e1_labels) == 1
    # e2 should have all 0 labels
    e2_labels = pairs_df[pairs_df["citation_key"] == "e2"]["label"].tolist()
    assert sum(e2_labels) == 0
    # e3 not in labels -> absent
    assert "e3" not in pairs_df["citation_key"].values
    # Check qid contiguity
    qids = pairs_df["qid"].tolist()
    assert qids[:3] == ["pub1::e1"] * 3 or qids[:3] == ["pub1::e2"] * 3
    print("  [7/10] build_pair_table candidate matrix & qid contiguity: PASSED")

    # Test 8: Groundtruth not in candidate references
    lbl_missing_gt = pd.DataFrame([
        {"arxiv_id": "pub1", "citation_key": "e1", "groundtruth_arxiv_id": "nonexistent_ref", "label_source": "manual"},
    ])
    _, stats_missing = build_pair_table(
        bib_df, ref_df, lbl_missing_gt,
        partition_of={"pub1": "train"},
    )
    assert stats_missing["gt_not_in_refs"] == 1
    print("  [8/10] groundtruth outside reference set handled gracefully: PASSED")

    # Test 9: Reference deduplication within publication
    ref_dup_df = pd.DataFrame([
        {"arxiv_id": "pub1", "ref_arxiv_id": "ref_B", "ref_title": "Ref B copy 1", "ref_authors": ["Devlin"], "ref_submission_date": "2018"},
        {"arxiv_id": "pub1", "ref_arxiv_id": "ref_B", "ref_title": "Ref B copy 2", "ref_authors": ["Devlin"], "ref_submission_date": "2018"},
    ])
    lbl_single = pd.DataFrame([
        {"arxiv_id": "pub1", "citation_key": "e1", "groundtruth_arxiv_id": "ref_B", "label_source": "manual"},
    ])
    pairs_dup, _ = build_pair_table(
        bib_df, ref_dup_df, lbl_single,
        partition_of={"pub1": "train"},
    )
    assert len(pairs_dup) == 1, f"Duplicate reference should be deduplicated to 1 candidate, got {len(pairs_dup)}"
    assert pairs_dup.iloc[0]["label"] == 1
    print("  [9/10] reference deduplication: PASSED")

    # Test 10: Pluggable feature functions
    dummy_fn = lambda b, r: {"dummy_score": 1.0}
    pairs_custom, _ = build_pair_table(
        bib_df, ref_df, lbl_df,
        partition_of={"pub1": "test"},
        feature_fns=[dummy_fn],
    )
    assert "dummy_score" in pairs_custom.columns
    assert (pairs_custom["dummy_score"] == 1.0).all()
    print("  [10/10] pluggable custom feature functions: PASSED")

    print("\nSELFTEST PASSED (10/10)")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metadata feature extraction & candidate pair generation for Milestone 3 Reference Matching."
    )
    parser.add_argument("--bibtex", type=str, help="Path to bibtex table (parquet/dir).")
    parser.add_argument("--references", type=str, help="Path to references table (parquet/dir or json).")
    parser.add_argument("--labels", type=str, help="Path to labels table (parquet/csv/json).")
    parser.add_argument("--out", type=str, default="data/matching", help="Output directory (default data/matching).")
    parser.add_argument("--k", type=int, default=1, help="Number of manual and auto publications per eval set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic split and sampling.")
    parser.add_argument("--max-neg-per-query", type=int, default=None, help="Max negatives per query on train partition.")
    parser.add_argument("--selftest", action="store_true", help="Run internal self-test suite and exit.")

    args = parser.parse_args()

    if args.selftest:
        _selftest()
        return

    if not args.bibtex or not args.labels:
        parser.print_help()
        print("\nError: --bibtex and --labels are required when not running --selftest.")
        return

    print("=== Milestone 3: Reference Matching - Metadata & Dataset Split ===")
    print(f"Loading bibtex from: {args.bibtex}")
    bib_df = load_table(args.bibtex)

    print(f"Loading labels from: {args.labels}")
    labels_df = load_table(args.labels)

    if args.references:
        print(f"Loading references from: {args.references}")
        ref_df = load_table(args.references)
    else:
        print("No --references path specified. Gathering from data/*/references.json...")
        ref_df = collect_references_from_json("data")

    # Extract distinct labeled publications
    labeled_pubs_df = (
        labels_df[[LABEL_COL_ARXIV_ID, LABEL_COL_SOURCE]]
        .drop_duplicates()
        .copy()
    )
    # Count entries per publication
    counts = labels_df.groupby(LABEL_COL_ARXIV_ID).size().to_dict()
    labeled_pubs_df["n_entries"] = labeled_pubs_df[LABEL_COL_ARXIV_ID].map(counts)

    print(f"Partitioning publications (k={args.k}, seed={args.seed})...")
    split_map = split_publications(labeled_pubs_df, k=args.k, seed=args.seed)

    print("Building candidate pair table and extracting metadata features...")
    pairs_df, stats = build_pair_table(
        bib_df,
        ref_df,
        labels_df,
        partition_of=split_map,
        max_neg_per_query=args.max_neg_per_query,
        seed=args.seed,
    )

    print(f"Saving output to: {args.out}")
    pq_path, split_path = save_pair_table(pairs_df, split_map, out_dir=args.out)

    print("\n--- Summary Statistics ---")
    print(f"Output Parquet: {pq_path} ({len(pairs_df):,} rows)")
    print(f"Output Split:   {split_path} ({len(split_map):,} publications)")
    print(f"Total Publications: {stats['total_publications']}")
    print(f"Total Queries (qids): {stats['total_qids']}")
    print(f"Positive Ratio: {stats['positive_ratio_overall']:.4%}")
    print(f"Groundtruth not in crawled refs: {stats['gt_not_in_refs']}")
    print(f"Skipped queries (bib entry not in bibtex_df): {stats['skipped_bib_not_found']}")
    print("\nPartition Breakdown:")
    for part, p_stat in stats["by_partition"].items():
        print(f"  [{part.upper()}] Pubs: {p_stat['n_pubs']}, Queries: {p_stat['n_qids']}, Pairs: {p_stat['n_pairs']:,}, Positives: {p_stat['n_pos']} ({p_stat['pos_ratio']:.2%})")


if __name__ == "__main__":
    main()
