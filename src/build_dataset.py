"""Unified Dataset Builder for Milestone 3 Reference Matching.

Combines:
- Text & NLP similarity features (src/features_text.py - Nhu Y)
- Metadata features, candidate pair matrix & train/valid/test split (src/features_meta.py - Nhu Ngoc)

Usage:
    python src/build_dataset.py --bibtex <path> --references <path> --labels <path> --out data/matching
    python src/build_dataset.py --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from features_meta import (
    LABEL_COL_ARXIV_ID,
    LABEL_COL_SOURCE,
    build_pair_table,
    collect_references_from_json,
    compute_author_features,
    compute_id_features,
    compute_year_features,
    load_table,
    save_pair_table,
    split_publications,
)
from features_text import compute_text_features

ALL_FEATURE_FNS = (
    compute_text_features,
    compute_author_features,
    compute_year_features,
    compute_id_features,
)


def build_and_save_dataset(
    bibtex_df: pd.DataFrame,
    references_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    out_dir: str | Path = "data/matching",
    k: int = 1,
    seed: int = 42,
    max_neg_per_query: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Run full feature extraction (text + meta), split publications, and save parquet files."""
    t0 = time.perf_counter()

    # 1. Extract distinct labeled publications and entry counts
    labeled_pubs_df = (
        labels_df[[LABEL_COL_ARXIV_ID, LABEL_COL_SOURCE]]
        .drop_duplicates()
        .copy()
    )
    counts = labels_df.groupby(LABEL_COL_ARXIV_ID).size().to_dict()
    labeled_pubs_df["n_entries"] = labeled_pubs_df[LABEL_COL_ARXIV_ID].map(counts)

    # 2. Split publications into train / valid / test (k manual + k auto in valid & test)
    split_map = split_publications(labeled_pubs_df, k=k, seed=seed)

    # 3. Build candidate pair matrix & compute all text + meta features
    pairs_df, stats = build_pair_table(
        bibtex_df=bibtex_df,
        references_df=references_df,
        labels_df=labels_df,
        partition_of=split_map,
        feature_fns=ALL_FEATURE_FNS,
        max_neg_per_query=max_neg_per_query,
        seed=seed,
    )

    # 4. Save unified pairs.parquet + split.json + individual partition parquet files
    out_path = Path(out_dir)
    pq_path, split_path = save_pair_table(pairs_df, split_map, out_dir=out_path)

    for part in ("train", "valid", "test"):
        part_df = pairs_df[pairs_df["partition"] == part].reset_index(drop=True)
        part_df.to_parquet(out_path / f"{part}.parquet", index=False)

    elapsed = time.perf_counter() - t0
    stats["elapsed_sec"] = elapsed
    stats["pq_path"] = str(pq_path)
    stats["split_path"] = str(split_path)
    return pairs_df, stats


def _run_integration_selftest() -> None:
    """Verify full integration between features_text.py and features_meta.py."""
    print("Running full integration selftest (features_text + features_meta)...")

    bibtex_df = pd.DataFrame([
        {
            "arxiv_id": "2511.00001",
            "citation_key": "vaswani2017",
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "journal": "NeurIPS",
            "arxiv_id_hint": "1706.03762",
        },
        {
            "arxiv_id": "2511.00002",
            "citation_key": "devlin2018",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            "authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "year": 2018,
            "journal": "NAACL",
        },
        {
            "arxiv_id": "2511.00003",
            "citation_key": "he2016",
            "title": "Deep Residual Learning for Image Recognition",
            "authors": ["Kaiming He", "Xiangyu Zhang"],
            "year": 2016,
            "journal": "CVPR",
        },
        {
            "arxiv_id": "2511.00004",
            "citation_key": "lecun2015",
            "title": "Deep learning",
            "authors": ["Yann LeCun", "Yoshua Bengio"],
            "year": 2015,
            "journal": "Nature",
        },
    ])

    references_df = pd.DataFrame([
        {
            "arxiv_id": "2511.00001",
            "ref_arxiv_id": "1706.03762",
            "ref_title": "Attention is all you need",
            "ref_authors": ["A. Vaswani", "N. Shazeer"],
            "ref_submission_date": "2017-06-12",
            "ref_venue": "Neural Information Processing Systems",
        },
        {
            "arxiv_id": "2511.00001",
            "ref_arxiv_id": "1810.04805",
            "ref_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "ref_authors": ["J. Devlin", "M. Chang"],
            "ref_submission_date": "2018-10-11",
        },
        {
            "arxiv_id": "2511.00002",
            "ref_arxiv_id": "1810.04805",
            "ref_title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "ref_authors": ["Jacob Devlin", "Ming-Wei Chang"],
            "ref_submission_date": "2018-10-11",
        },
        {
            "arxiv_id": "2511.00003",
            "ref_arxiv_id": "1512.03385",
            "ref_title": "Deep Residual Learning for Image Recognition",
            "ref_authors": ["Kaiming He"],
            "ref_submission_date": "2015-12-10",
        },
        {
            "arxiv_id": "2511.00004",
            "ref_arxiv_id": "1505.00001",
            "ref_title": "Deep learning",
            "ref_authors": ["Yann LeCun"],
            "ref_submission_date": "2015-05-01",
        },
    ])

    labels_df = pd.DataFrame([
        {"arxiv_id": "2511.00001", "citation_key": "vaswani2017", "groundtruth_arxiv_id": "1706.03762", "label_source": "manual"},
        {"arxiv_id": "2511.00002", "citation_key": "devlin2018", "groundtruth_arxiv_id": "1810.04805", "label_source": "manual"},
        {"arxiv_id": "2511.00003", "citation_key": "he2016", "groundtruth_arxiv_id": "1512.03385", "label_source": "auto"},
        {"arxiv_id": "2511.00004", "citation_key": "lecun2015", "groundtruth_arxiv_id": "1505.00001", "label_source": "auto"},
    ])

    labeled_pubs = labels_df[["arxiv_id", "label_source"]].drop_duplicates().copy()
    labeled_pubs["n_entries"] = 1
    split_map = split_publications(labeled_pubs, k=1, seed=42)

    pairs_df, stats = build_pair_table(
        bibtex_df=bibtex_df,
        references_df=references_df,
        labels_df=labels_df,
        partition_of=split_map,
        feature_fns=ALL_FEATURE_FNS,
    )

    assert len(pairs_df) == 5
    # Verify both text and meta feature columns exist
    assert "title_token_set" in pairs_df.columns
    assert "title_exact_match" in pairs_df.columns
    assert "first_author_match" in pairs_df.columns
    assert "year_diff" in pairs_df.columns
    assert "arxiv_hint_match" in pairs_df.columns

    # Check positive row for devlin2018 (subtitle truncation)
    devlin_row = pairs_df[pairs_df["citation_key"] == "devlin2018"].iloc[0]
    assert devlin_row["label"] == 1
    assert devlin_row["title_token_set"] == 1.0
    assert devlin_row["first_author_match"] == 1.0

    print(f"  Generated {len(pairs_df)} pairs with {len(pairs_df.columns)} total columns.")
    print(f"  Feature columns count: {len(pairs_df.columns) - 7}")
    print("INTEGRATION SELFTEST PASSED!")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified Milestone 3 dataset (Text + Metadata features & Train/Valid/Test split)."
    )
    parser.add_argument("--bibtex", type=str, help="Path to unified bibtex table (parquet file or dir).")
    parser.add_argument("--references", type=str, help="Path to references table (parquet file or dir).")
    parser.add_argument("--labels", type=str, help="Path to labels table from Manh Dat (parquet/csv/json).")
    parser.add_argument("--out", type=str, default="data/matching", help="Output directory (default: data/matching).")
    parser.add_argument("--k", type=int, default=1, help="Number of manual & auto pubs in valid/test (default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--max-neg-per-query", type=int, default=None, help="Optional negative sampling for train.")
    parser.add_argument("--selftest", action="store_true", help="Run integration selftest and exit.")

    args = parser.parse_args()

    if args.selftest:
        _run_integration_selftest()
        return

    if not args.bibtex or not args.labels:
        parser.print_help()
        print("\nError: --bibtex and --labels are required when not running --selftest.")
        return

    print("=== Milestone 3: Building Unified Dataset (Text + Metadata) ===")
    bib_df = load_table(args.bibtex)
    labels_df = load_table(args.labels)
    if args.references:
        ref_df = load_table(args.references)
    else:
        print("No --references specified, gathering from data/*/references.json...")
        ref_df = collect_references_from_json("data")

    pairs_df, stats = build_and_save_dataset(
        bibtex_df=bib_df,
        references_df=ref_df,
        labels_df=labels_df,
        out_dir=args.out,
        k=args.k,
        seed=args.seed,
        max_neg_per_query=args.max_neg_per_query,
    )

    print(f"\nCompleted in {stats['elapsed_sec']:.2f}s!")
    print(f"Saved unified pairs: {stats['pq_path']} ({len(pairs_df):,} rows, {len(pairs_df.columns)} cols)")
    print(f"Saved train.parquet, valid.parquet, test.parquet in: {args.out}")
    print("\nPartition Breakdown:")
    for part, p_stat in stats["by_partition"].items():
        print(
            f"  [{part.upper()}] Pubs: {p_stat['n_pubs']}, Queries: {p_stat['n_qids']}, "
            f"Pairs: {p_stat['n_pairs']:,}, Positives: {p_stat['n_pos']} ({p_stat['pos_ratio']:.2%})"
        )


if __name__ == "__main__":
    main()

