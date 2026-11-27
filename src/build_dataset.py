"""Unified Dataset Builder for Milestone 3 Reference Matching.

Combines:
- Labels from Manh Dat (src/m3/manual_truth.csv & src/m3/auto_truth.csv)
- Text & NLP similarity features (src/features_text.py - Nhu Y)
- Metadata features, candidate pair matrix & train/valid/test split (src/features_meta.py - Nhu Ngoc)

Usage:
    python src/build_dataset.py --bibtex <path_or_glob> --references <path_or_glob> --out data/matching
    python src/build_dataset.py --selftest
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from features_meta import (
    LABEL_COL_ARXIV_ID,
    LABEL_COL_CITATION_KEY,
    LABEL_COL_GROUNDTRUTH,
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


def load_and_merge_labels(
    manual_path: str | Path = "src/m3/manual_truth.csv",
    auto_path: str | Path = "src/m3/auto_truth.csv",
) -> pd.DataFrame:
    """Load Manh Dat's manual and automatic label CSVs into unified schema.

    - Keeps only valid matched entries (label_status == 'matched' and non-empty groundtruth_arxiv_id).
    - Standardizes label_source to 'manual' / 'auto' (required by features_meta.split_publications).
    - Deduplicates (arxiv_id, citation_key), giving priority to manual labels.
    """
    frames: list[pd.DataFrame] = []

    if manual_path and Path(manual_path).is_file():
        df_m = pd.read_csv(manual_path, dtype=str)
        df_m[LABEL_COL_SOURCE] = "manual"
        frames.append(df_m)

    if auto_path and Path(auto_path).is_file():
        df_a = pd.read_csv(auto_path, dtype=str)
        if LABEL_COL_SOURCE in df_a.columns:
            df_a[LABEL_COL_SOURCE] = (
                df_a[LABEL_COL_SOURCE]
                .fillna("auto")
                .str.strip()
                .str.lower()
                .replace({"automatic": "auto"})
            )
        else:
            df_a[LABEL_COL_SOURCE] = "auto"
        frames.append(df_a)

    if not frames:
        raise FileNotFoundError(f"No label files found at {manual_path} or {auto_path}")

    df = pd.concat(frames, ignore_index=True)

    # Keep only valid matched rows per M3 specification
    if "label_status" in df.columns:
        df = df[df["label_status"].astype(str).str.strip().str.lower() == "matched"]

    df[LABEL_COL_GROUNDTRUTH] = df[LABEL_COL_GROUNDTRUTH].fillna("").astype(str).str.strip()
    df = df[df[LABEL_COL_GROUNDTRUTH] != ""]

    df[LABEL_COL_ARXIV_ID] = df[LABEL_COL_ARXIV_ID].astype(str).str.strip()
    df[LABEL_COL_CITATION_KEY] = df[LABEL_COL_CITATION_KEY].astype(str).str.strip()

    # Manual comes first in `frames`, so keep="first" prefers manual on overlap
    df = df.drop_duplicates(subset=[LABEL_COL_ARXIV_ID, LABEL_COL_CITATION_KEY], keep="first")

    return df[[LABEL_COL_ARXIV_ID, LABEL_COL_CITATION_KEY, LABEL_COL_GROUNDTRUTH, LABEL_COL_SOURCE]].reset_index(drop=True)


def load_many_tables(paths: list[str], table_subdir: str | None = None) -> pd.DataFrame:
    """Load and concatenate one or more parquet files/directories."""
    dfs = []
    for p_str in paths:
        p = Path(p_str)
        if p.is_dir() and table_subdir and (p / table_subdir).exists():
            dfs.append(load_table(p / table_subdir))
        else:
            dfs.append(load_table(p))
    return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]


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
    labeled_ids = set(labels_df[LABEL_COL_ARXIV_ID].astype(str).str.strip())
    bibtex_df = bibtex_df[bibtex_df["arxiv_id"].astype(str).str.strip().isin(labeled_ids)].copy()
    references_df = references_df[references_df["arxiv_id"].astype(str).str.strip().isin(labeled_ids)].copy()
    print(
        f"Filtered to {len(labeled_ids):,} labeled publications -> "
        f"{len(bibtex_df):,} bibtex rows, {len(references_df):,} reference rows.",
        flush=True,
    )

    labeled_pubs_df = (
        labels_df[[LABEL_COL_ARXIV_ID, LABEL_COL_SOURCE]]
        .drop_duplicates()
        .copy()
    )
    counts = labels_df.groupby(LABEL_COL_ARXIV_ID).size().to_dict()
    labeled_pubs_df["n_entries"] = labeled_pubs_df[LABEL_COL_ARXIV_ID].map(counts)

    # 2. Split publications into train / valid / test (k manual + k auto in valid & test)
    split_map = split_publications(labeled_pubs_df, k=k, seed=seed)

    # 3. Build candidate pair matrix & compute all text + meta features in batches for live progress
    all_pub_ids = sorted(labeled_ids)
    batch_size = 250
    chunk_dfs = []
    agg_stats = {
        "total_pairs": 0,
        "total_qids": 0,
        "total_publications": 0,
        "by_partition": {
            p: {"n_pubs": 0, "n_qids": 0, "n_pairs": 0, "n_pos": 0, "pos_ratio": 0.0}
            for p in ("train", "valid", "test")
        },
        "positive_ratio_overall": 0.0,
        "all_negative_qids": 0,
        "gt_not_in_refs": 0,
        "skipped_bib_not_found": 0,
    }

    bib_groups = {aid: df_g for aid, df_g in bibtex_df.groupby("arxiv_id", sort=False)}
    ref_groups_df = {aid: df_g for aid, df_g in references_df.groupby("arxiv_id", sort=False)}
    lbl_groups = {aid: df_g for aid, df_g in labels_df.groupby(LABEL_COL_ARXIV_ID, sort=False)}

    total_batches = (len(all_pub_ids) + batch_size - 1) // batch_size
    for b_idx in range(total_batches):
        batch_ids = all_pub_ids[b_idx * batch_size : (b_idx + 1) * batch_size]
        b_bib = pd.concat([bib_groups[i] for i in batch_ids if i in bib_groups], ignore_index=True) if any(i in bib_groups for i in batch_ids) else bibtex_df.iloc[0:0]
        b_ref = pd.concat([ref_groups_df[i] for i in batch_ids if i in ref_groups_df], ignore_index=True) if any(i in ref_groups_df for i in batch_ids) else references_df.iloc[0:0]
        b_lbl = pd.concat([lbl_groups[i] for i in batch_ids if i in lbl_groups], ignore_index=True)

        c_df, c_stats = build_pair_table(
            bibtex_df=b_bib,
            references_df=b_ref,
            labels_df=b_lbl,
            partition_of=split_map,
            feature_fns=ALL_FEATURE_FNS,
            max_neg_per_query=max_neg_per_query,
            seed=seed,
        )
        if len(c_df) > 0:
            chunk_dfs.append(c_df)

        agg_stats["total_pairs"] += c_stats["total_pairs"]
        agg_stats["total_qids"] += c_stats["total_qids"]
        agg_stats["total_publications"] += c_stats["total_publications"]
        agg_stats["all_negative_qids"] += c_stats["all_negative_qids"]
        agg_stats["gt_not_in_refs"] += c_stats["gt_not_in_refs"]
        agg_stats["skipped_bib_not_found"] += c_stats["skipped_bib_not_found"]
        for p in ("train", "valid", "test"):
            for k_stat in ("n_pubs", "n_qids", "n_pairs", "n_pos"):
                agg_stats["by_partition"][p][k_stat] += c_stats["by_partition"][p][k_stat]

        done_pubs = min((b_idx + 1) * batch_size, len(all_pub_ids))
        print(
            f"  [Batch {b_idx + 1}/{total_batches}] {done_pubs}/{len(all_pub_ids)} pubs | "
            f"pairs so far: {agg_stats['total_pairs']:,} | elapsed: {time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    pairs_df = pd.concat(chunk_dfs, ignore_index=True) if chunk_dfs else pd.DataFrame()
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values(
            by=["partition", "arxiv_id", "citation_key", "ref_arxiv_id"],
            ignore_index=True,
        )

    total_pos = sum(agg_stats["by_partition"][p]["n_pos"] for p in ("train", "valid", "test"))
    agg_stats["positive_ratio_overall"] = (total_pos / agg_stats["total_pairs"]) if agg_stats["total_pairs"] > 0 else 0.0
    for p in ("train", "valid", "test"):
        npairs = agg_stats["by_partition"][p]["n_pairs"]
        agg_stats["by_partition"][p]["pos_ratio"] = (agg_stats["by_partition"][p]["n_pos"] / npairs) if npairs > 0 else 0.0
    stats = agg_stats

    # 4. Save unified pairs.parquet + split.json + individual partition parquet files
    print("Saving parquet files...", flush=True)
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
    assert "title_token_set" in pairs_df.columns
    assert "title_exact_match" in pairs_df.columns
    assert "first_author_match" in pairs_df.columns
    assert "year_diff" in pairs_df.columns
    assert "arxiv_hint_match" in pairs_df.columns

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
    parser.add_argument("--bibtex", nargs="+", help="Path(s) to unified bibtex parquet file(s) or dir(s).")
    parser.add_argument("--references", nargs="+", help="Path(s) to references parquet file(s) or dir(s).")
    parser.add_argument("--manual-labels", type=str, default="src/m3/manual_truth.csv", help="Path to manual_truth.csv.")
    parser.add_argument("--auto-labels", type=str, default="src/m3/auto_truth.csv", help="Path to auto_truth.csv.")
    parser.add_argument("--labels", type=str, default=None, help="Optional single unified labels file (overrides manual/auto).")
    parser.add_argument("--out", type=str, default="data/matching", help="Output directory (default: data/matching).")
    parser.add_argument("--k", type=int, default=1, help="Number of manual & auto pubs in valid/test (default: 1).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--max-neg-per-query", type=int, default=None, help="Optional negative sampling for train.")
    parser.add_argument("--selftest", action="store_true", help="Run integration selftest and exit.")

    args = parser.parse_args()

    if args.selftest:
        _run_integration_selftest()
        return

    if not args.bibtex:
        parser.print_help()
        print("\nError: --bibtex is required when not running --selftest.")
        return

    print("=== Milestone 3: Building Unified Dataset (Text + Metadata) ===")
    bib_df = load_many_tables(args.bibtex)

    if args.labels:
        labels_df = load_table(args.labels)
    else:
        labels_df = load_and_merge_labels(args.manual_labels, args.auto_labels)
    print(f"Loaded {len(labels_df):,} valid matched label entries.")

    if args.references:
        ref_df = load_many_tables(args.references)
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
