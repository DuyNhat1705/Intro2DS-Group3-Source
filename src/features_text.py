"""Text preprocessing and similarity feature engineering for Reference Matching (Milestone 3).

Author: Nhu Y (Group 3)
Designed for:
- Maximum accuracy on real-world scientific paper titles & venues
- Ultra-fast vectorized/cached performance across millions of candidate pairs
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# Try high-performance C++ rapidfuzz, fallback to difflib if not yet installed
try:
    from rapidfuzz import fuzz, distance
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    import difflib

# ---------------------------------------------------------------------------
# 1. High-Performance Text Normalization & Cleaning
# ---------------------------------------------------------------------------

_LATEX_COMMAND_RE = re.compile(r"\\[a-zA-Z]+(?:\*|\b)(?:\[[^\]]*\])?")
_BRACES_RE = re.compile(r"[{}]")
_PUNCT_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")
_MATH_BLOCK_RE = re.compile(r"\$[^$]*\$|\\begin\{equation\}.*?\\end\{equation\}", re.DOTALL)

# Common scientific English stop words
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "should", "shouldn't", "so", "some", "such", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "were", "weren't", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "won't", "would", "wouldn't", "you", "your",
    "yours", "yourself", "yourselves"
}


def clean_latex_markup(text: str) -> str:
    """Strip LaTeX markup, accents, quotes, and HTML entities."""
    if not text:
        return ""
    
    # 1. Unescape HTML entities (e.g. &amp;, &#x2013;)
    s = html.unescape(text)
    
    # 2. Normalize unicode (NFC/NFKD to convert accents e.g., e + ´ -> é)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    
    # 3. Replace LaTeX math with space
    s = _MATH_BLOCK_RE.sub(" ", s)
    
    # 4. Replace LaTeX quotes and dashes
    s = s.replace("``", '"').replace("''", '"').replace("`", "'")
    s = s.replace("---", " - ").replace("--", " - ").replace("~", " ")
    
    # 5. Remove commands like \textbf{...}, \emph{...}, leaving the text
    # Standard common macro replacements
    s = re.sub(r"\\(?:textbf|textit|emph|text|mathrm|mathbf|underline)\{([^}]*)\}", r"\1", s)
    s = _LATEX_COMMAND_RE.sub(" ", s)
    s = _BRACES_RE.sub("", s)
    
    return s.strip()


@lru_cache(maxsize=65536)
def normalize_title(text: Optional[str]) -> str:
    """Deeply cleaned, lowercased, punctuation-free string for title comparison."""
    if not text:
        return ""
    
    s = clean_latex_markup(text)
    s = s.lower()
    # Remove punctuation
    s = _PUNCT_RE.sub(" ", s)
    # Collapse whitespace
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def get_tokens(text: str, remove_stopwords: bool = False) -> List[str]:
    """Tokenize normalized string into word tokens."""
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return tokens


def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from string."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


# ---------------------------------------------------------------------------
# 2. Similarity Metric Engines (with Fast Fallbacks)
# ---------------------------------------------------------------------------

def compute_levenshtein_ratio(s1: str, s2: str) -> float:
    """Normalized similarity in [0.0, 1.0] based on edit distance."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    if HAS_RAPIDFUZZ:
        return float(distance.Levenshtein.normalized_similarity(s1, s2))
    else:
        return difflib.SequenceMatcher(None, s1, s2).ratio()


def compute_jaro_winkler(s1: str, s2: str) -> float:
    """Jaro-Winkler similarity in [0.0, 1.0] (high weight on prefix matches)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
        
    if HAS_RAPIDFUZZ:
        return float(distance.JaroWinkler.similarity(s1, s2))
    else:
        # High quality pure Python Jaro-Winkler approximation
        m = difflib.SequenceMatcher(None, s1, s2).ratio()
        prefix = 0
        for c1, c2 in zip(s1[:4], s2[:4]):
            if c1 == c2:
                prefix += 1
            else:
                break
        return min(1.0, m + prefix * 0.1 * (1.0 - m))


def compute_token_sort_ratio(s1: str, s2: str) -> float:
    """Token sort ratio: robust against reordered words in titles."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
        
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_sort_ratio(s1, s2)) / 100.0
    else:
        t1 = " ".join(sorted(s1.split()))
        t2 = " ".join(sorted(s2.split()))
        return difflib.SequenceMatcher(None, t1, t2).ratio()


def compute_token_set_ratio(s1: str, s2: str) -> float:
    """Token set ratio: detects title when one is a subset / truncated subtitle."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
        
    if HAS_RAPIDFUZZ:
        return float(fuzz.token_set_ratio(s1, s2)) / 100.0
    else:
        tokens1 = set(s1.split())
        tokens2 = set(s2.split())
        intersection = tokens1 & tokens2
        if not intersection:
            return 0.0
        diff1to2 = tokens1 - tokens2
        diff2to1 = tokens2 - tokens1
        
        sorted_inter = " ".join(sorted(intersection))
        sorted_1to2 = " ".join(sorted(diff1to2))
        sorted_2to1 = " ".join(sorted(diff2to1))
        
        c1 = (sorted_inter + " " + sorted_1to2).strip()
        c2 = (sorted_inter + " " + sorted_2to1).strip()
        
        r1 = difflib.SequenceMatcher(None, sorted_inter, c1).ratio()
        r2 = difflib.SequenceMatcher(None, sorted_inter, c2).ratio()
        r3 = difflib.SequenceMatcher(None, c1, c2).ratio()
        return max(r1, r2, r3)


def compute_jaccard_similarity(tokens1: Sequence[str], tokens2: Sequence[str]) -> float:
    """Jaccard index of word/token sets."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / float(len(s1 | s2))


def compute_containment(tokens_small: Sequence[str], tokens_large: Sequence[str]) -> float:
    """Fraction of small token set contained within the large set."""
    s_small = set(tokens_small)
    s_large = set(tokens_large)
    if not s_small:
        return 0.0
    return len(s_small & s_large) / float(len(s_small))


@lru_cache(maxsize=65536)
def _char_ngram_counts(s: str, n: int) -> Tuple[Counter, float]:
    ng = Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1)))
    norm = math.sqrt(sum(v * v for v in ng.values()))
    return ng, norm


def compute_cosine_char_ngram(s1: str, s2: str, n: int = 3) -> float:
    """Fast character n-gram cosine similarity (subword matching)."""
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    ng1, norm1 = _char_ngram_counts(s1, n)
    ng2, norm2 = _char_ngram_counts(s2, n)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    all_keys = ng1.keys() & ng2.keys()
    if not all_keys:
        return 0.0

    dot = sum(ng1[k] * ng2[k] for k in all_keys)
    return dot / (norm1 * norm2)


# ---------------------------------------------------------------------------
# 3. Main Feature Extraction Entry Points
# ---------------------------------------------------------------------------

def extract_text_features(
    bib_title: Optional[str],
    ref_title: Optional[str],
    bib_journal: Optional[str] = None,
    ref_venue: Optional[str] = None
) -> Dict[str, float]:
    """Compute high-fidelity text similarity features for one candidate pair.
    
    Parameters
    ----------
    bib_title : str or None
        Title extracted from LaTeX / BibTeX
    ref_title : str or None
        Title from Semantic Scholar crawled references
    bib_journal : str or None, optional
        Journal or booktitle from BibTeX
    ref_venue : str or None, optional
        Venue / publication journal from Semantic Scholar
        
    Returns
    -------
    dict
        Dictionary of feature names mapping to numeric float scores.
    """
    # 1. Normalization
    t1 = normalize_title(bib_title)
    t2 = normalize_title(ref_title)
    
    # 2. Token sets
    toks1 = get_tokens(t1, remove_stopwords=False)
    toks2 = get_tokens(t2, remove_stopwords=False)
    
    toks_nostop1 = get_tokens(t1, remove_stopwords=True)
    toks_nostop2 = get_tokens(t2, remove_stopwords=True)
    
    len1 = len(t1)
    len2 = len(t2)
    max_len = max(len1, len2)
    min_len = min(len1, len2)
    
    # Basic string metrics
    features: Dict[str, float] = {}
    
    # [Feature 1] Exact Match
    features["title_exact_match"] = 1.0 if (t1 and t1 == t2) else 0.0
    
    # [Feature 2] Length differences
    features["title_len_diff"] = float(abs(len1 - len2))
    features["title_len_ratio"] = (min_len / float(max_len)) if max_len > 0 else 0.0
    
    # [Feature 3] Word count differences
    w_len1 = len(toks1)
    w_len2 = len(toks2)
    max_w = max(w_len1, w_len2)
    features["title_word_count_diff"] = float(abs(w_len1 - w_len2))
    features["title_word_count_ratio"] = (min(w_len1, w_len2) / float(max_w)) if max_w > 0 else 0.0
    
    # [Feature 4 & 5] String distance ratios
    features["title_levenshtein"] = compute_levenshtein_ratio(t1, t2)
    features["title_jaro_winkler"] = compute_jaro_winkler(t1, t2)
    
    # [Feature 6 & 7] Token sort and set ratios (handles permutations and subtitles)
    features["title_token_sort"] = compute_token_sort_ratio(t1, t2)
    features["title_token_set"] = compute_token_set_ratio(t1, t2)
    
    # [Feature 8 & 9] Jaccard overlaps
    features["title_word_jaccard"] = compute_jaccard_similarity(toks1, toks2)
    features["title_word_jaccard_nostop"] = compute_jaccard_similarity(toks_nostop1, toks_nostop2)
    
    # [Feature 10] Containment (is one title completely inside the other?)
    if w_len1 <= w_len2:
        features["title_containment"] = compute_containment(toks_nostop1, toks_nostop2)
    else:
        features["title_containment"] = compute_containment(toks_nostop2, toks_nostop1)
        
    # [Feature 11 & 12] Character 3-gram and 4-gram cosine similarities
    features["title_char_3gram_cosine"] = compute_cosine_char_ngram(t1, t2, n=3)
    features["title_char_4gram_cosine"] = compute_cosine_char_ngram(t1, t2, n=4)
    
    # [Feature 13] Prefix match ratio (first 3 words match)
    p1 = " ".join(toks1[:3]) if toks1 else ""
    p2 = " ".join(toks2[:3]) if toks2 else ""
    features["title_prefix_3word_match"] = 1.0 if (p1 and p1 == p2) else 0.0
    
    # [Feature 14] Venue / Journal similarity (if available)
    v1 = normalize_title(bib_journal)
    v2 = normalize_title(ref_venue)
    if v1 and v2:
        features["venue_has_data"] = 1.0
        features["venue_token_set"] = compute_token_set_ratio(v1, v2)
        features["venue_word_jaccard"] = compute_jaccard_similarity(v1.split(), v2.split())
    else:
        features["venue_has_data"] = 0.0
        features["venue_token_set"] = 0.0
        features["venue_word_jaccard"] = 0.0
        
    return features


# ---------------------------------------------------------------------------
# 4. Ultra-Fast Batch Processor for Millions of Pairs
# ---------------------------------------------------------------------------

class CachedTextFeatureExtractor:
    """Pre-caches tokenization and normalized strings to process thousands of pairs/sec."""
    
    def __init__(self):
        self._norm_cache: Dict[str, str] = {}
        
    def get_normalized(self, raw_text: Optional[str]) -> str:
        if not raw_text:
            return ""
        if raw_text not in self._norm_cache:
            self._norm_cache[raw_text] = normalize_title(raw_text)
        return self._norm_cache[raw_text]
        
    def extract_pair(
        self,
        bib_title: Optional[str],
        ref_title: Optional[str],
        bib_journal: Optional[str] = None,
        ref_venue: Optional[str] = None
    ) -> Dict[str, float]:
        # Uses standard function, but warm caches accelerate repeated titles
        return extract_text_features(bib_title, ref_title, bib_journal, ref_venue)


def _safe_str(val: Any) -> Optional[str]:
    """Convert None, float NaN, or non-string safely to str or None."""
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    s = str(val).strip()
    return s if s else None


def compute_text_features(bib: Dict[str, Any], ref: Dict[str, Any]) -> Dict[str, float]:
    """Pluggable feature adapter compatible with features_meta.build_pair_table.

    Accepts a prepared (or raw) BibTeX dict and Reference dict, extracts title & venue
    fields safely, and returns all 17 text similarity features.
    """
    bib_title = _safe_str(bib.get("title"))
    ref_title = _safe_str(ref.get("ref_title") if ref.get("ref_title") is not None else ref.get("title"))
    bib_journal = _safe_str(bib.get("journal") if bib.get("journal") is not None else bib.get("booktitle"))
    ref_venue = _safe_str(ref.get("ref_venue") if ref.get("ref_venue") is not None else ref.get("venue"))
    return extract_text_features(bib_title, ref_title, bib_journal, ref_venue)


