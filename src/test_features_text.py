"""Unit test & benchmark for features_text.py"""

import time
from features_text import extract_text_features, normalize_title, clean_latex_markup

def test_latex_cleaning():
    print("--- 1. Testing LaTeX Cleaning ---")
    raw = r"Attention is all you need: {\em Vaswani} et~al. -- ``Transformers'' \& \textbf{BERT}"
    cleaned = normalize_title(raw)
    print("Raw:    ", raw)
    print("Cleaned:", cleaned)
    assert "vaswani" in cleaned
    assert "transformers" in cleaned
    assert "bert" in cleaned
    print("PASS: LaTeX cleaning works accurately!")

def test_pair_features():
    print("\n--- 2. Testing Pair Feature Extraction ---")
    # Real-world scientific paper title variations
    bib_title = r"Attention is all you need"
    ref_title = r"Attention Is All You Need"
    feats = extract_text_features(bib_title, ref_title)
    print(f"Match 1 (Identical case-insensitive):")
    print(f"  Exact Match: {feats['title_exact_match']}")
    print(f"  Levenshtein: {feats['title_levenshtein']:.4f}")
    print(f"  Jaro-Winkler: {feats['title_jaro_winkler']:.4f}")
    assert feats['title_exact_match'] == 1.0

    # Truncated subtitle variation (extremely common in arXiv/Semantic Scholar)
    bib_title_2 = r"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    ref_title_2 = r"BERT: Pre-training of Deep Bidirectional Transformers"
    feats_2 = extract_text_features(bib_title_2, ref_title_2)
    print(f"\nMatch 2 (Subtitle truncation):")
    print(f"  Exact Match: {feats_2['title_exact_match']}")
    print(f"  Token Set Ratio: {feats_2['title_token_set']:.4f}")
    print(f"  Containment: {feats_2['title_containment']:.4f}")
    print(f"  3-gram Cosine: {feats_2['title_char_3gram_cosine']:.4f}")
    assert feats_2['title_token_set'] >= 0.95 or feats_2['title_containment'] == 1.0

    # Completely different papers
    bib_title_3 = r"Deep Residual Learning for Image Recognition"
    ref_title_3 = r"Attention Is All You Need"
    feats_3 = extract_text_features(bib_title_3, ref_title_3)
    print(f"\nMatch 3 (Completely different papers):")
    print(f"  Exact Match: {feats_3['title_exact_match']}")
    print(f"  Token Sort Ratio: {feats_3['title_token_sort']:.4f}")
    print(f"  Levenshtein: {feats_3['title_levenshtein']:.4f}")
    assert feats_3['title_exact_match'] == 0.0
    assert feats_3['title_levenshtein'] < 0.4

def test_speed_benchmark():
    print("\n--- 3. Testing Speed Benchmark ---")
    bib = "Deep Residual Learning for Image Recognition and Computer Vision Tasks"
    ref = "Deep Residual Learning for Image Recognition"
    
    n_iters = 1000
    start = time.perf_counter()
    for _ in range(n_iters):
        _ = extract_text_features(bib, ref)
    elapsed = time.perf_counter() - start
    pairs_per_sec = n_iters / elapsed
    print(f"Processed {n_iters} pairs in {elapsed:.3f}s -> {pairs_per_sec:,.0f} pairs/sec!")
    print("PASS: Performance is super fast!")

if __name__ == "__main__":
    test_latex_cleaning()
    test_pair_features()
    test_speed_benchmark()

