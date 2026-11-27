"""Select publications and export reproducible Milestone 3 citation labels."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import csv
from itertools import islice
import json
from multiprocessing import get_context
from pathlib import Path
import random
import sys

from inputs import (RANGES, csv_rows, paper_ids, range_of, read_bibtex,
                    read_completion, read_references)
from matching import DEFAULT_TITLE_SCORER, TITLE_SCORERS, arxiv_id, get_title_similarity, match

STATUSES = ("matched", "no_match", "needs_review")  # Keep the zero-count review field in legacy reports.
LABEL_FIELDS = {
    "arxiv_id": "string",
    "citation_key": "string",
    "groundtruth_arxiv_id": "string",
    "label_source": "string",
    "label_status": "string",
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers", nargs="+", required=True, help="M1 papers Parquet or corpus CSVs")
    parser.add_argument("--references", nargs="+", required=True, help="M1 references Parquet files")
    parser.add_argument("--bibtex", nargs="+", required=True, help="Merged M2 Parquet files or bibtex directories")
    parser.add_argument("--refs-done", nargs="+", default=[], help="Scraper refs_done.jsonl files, oldest first")
    parser.add_argument("--empty-reference-ranges", nargs="+", type=int, choices=RANGES, default=[],
                        help="Ranges with a verified, wholly empty M1 references export; still requires --refs-done")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--manual", nargs="+", help="Manual exclusion CSVs with arxiv_id")
    selection.add_argument("--manual-ids", nargs="+", help="Manual publication IDs")
    selection.add_argument("--full-corpus", action="store_true",
                           help="Label all eligible papers, including papers with manual labels")
    parser.add_argument("--ranges", nargs="+", type=int, choices=RANGES, default=list(RANGES))
    parser.add_argument("--quota", type=int, help="Minimum per range (default: 500); below 500 marks a trial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-score", type=float, default=.85)
    parser.add_argument("--min-margin", type=float, default=.06)
    parser.add_argument("--title-scorer", choices=TITLE_SCORERS, default=DEFAULT_TITLE_SCORER,
                        help="Title similarity backend (default: rapidfuzz).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Labeling processes (default: 4). Use 1 for sequential labeling.")
    parser.add_argument("--eval-labels", type=Path,
                        help="Development-only manual CSV: arxiv_id,citation_key,groundtruth_arxiv_id")
    parser.add_argument("--out", type=Path, required=True, help="New directory for labels and summary")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0 <= args.min_score <= 1 or not 0 <= args.min_margin <= 1:
        parser.error("Score and margin must be between 0 and 1")
    if args.full_corpus:
        if args.quota is not None or args.eval_labels is not None:
            parser.error("--full-corpus cannot be used with --quota or --eval-labels")
    elif args.quota is None:
        args.quota = 500
    elif args.quota < 1:
        parser.error("--quota must be positive")
    return args


def select(eligible, external_count, label, args):
    # Per-range 10% floors also satisfy the pooled 10% requirement.
    target = args.quota
    if args.quota >= 500:
        target = max(target, (external_count + 9) // 10)
    rng = random.Random(f"{args.seed}:{label}")
    return sorted(rng.sample(sorted(eligible), min(target, len(eligible)))), target


def evaluate(path, manual, entries, candidates, confirmed, args):
    title_similarity = get_title_similarity(getattr(args, "title_scorer", DEFAULT_TITLE_SCORER))
    counts = Counter()
    seen = set()
    for row in csv_rows(path, ("arxiv_id", "citation_key", "groundtruth_arxiv_id")):
        ident, key = arxiv_id(row["arxiv_id"]), row["citation_key"]
        if ident not in manual:
            raise ValueError(f"ID đối chiếu phải thuộc tập thủ công: {ident}")
        if (ident, key) in seen:
            raise ValueError(f"Trùng khóa đối chiếu: ({ident}, {key})")
        seen.add((ident, key))
        if range_of(ident) not in args.ranges:
            continue
        raw_target = row["groundtruth_arxiv_id"]
        if raw_target is None or not key:
            raise ValueError(f"Dòng đối chiếu thiếu trường: ({ident}, {key})")
        target = arxiv_id(raw_target) if raw_target.strip() else None
        counts["total"] += 1
        entry = entries.get(ident, {}).get(key)
        if entry is None or (ident not in candidates and ident not in confirmed):
            counts["unavailable"] += 1
            continue
        prediction = match(entry, candidates.get(ident, {}), args.min_score, args.min_margin,
                           title_similarity=title_similarity)
        estimated = prediction["groundtruth_arxiv_id"]
        counts["evaluated"] += 1
        counts["matched"] += estimated is not None
        counts["correct_matches"] += estimated is not None and estimated == target
        counts["missed"] += target is not None and estimated != target
        counts["needs_review"] += prediction["label_status"] == "needs_review"
    fields = ("total", "unavailable", "evaluated", "matched", "correct_matches", "missed", "needs_review")
    return {**{name: counts[name] for name in fields},
            "match_precision": counts["correct_matches"] / counts["matched"] if counts["matched"] else None}


def _label_batch(selected, entries, candidates, min_score, min_margin, title_scorer):
    title_similarity = get_title_similarity(title_scorer)
    labels = []
    papers = []
    for ident in selected:
        pool = candidates.get(ident, {})
        counts = Counter()
        for key, row in sorted(entries[ident].items()):
            result = match(row, pool, min_score, min_margin, title_similarity=title_similarity)
            target = result["groundtruth_arxiv_id"]
            if result["label_status"] == "matched" and (target not in pool or target == ident):
                raise RuntimeError(f"Nhãn dương không thuộc ứng viên của {ident}: {target}")
            labels.append(dict(arxiv_id=ident, citation_key=key, label_source="automatic", **result))
            counts[result["label_status"]] += 1
        papers.append(dict(arxiv_id=ident, range=range_of(ident), bibtex_entries=len(entries[ident]),
                           **{status: counts[status] for status in STATUSES}))
    return labels, papers


def labeling_pool(workers):
    if workers == 1:
        return nullcontext()
    return ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn"))


def label_papers(selected, entries, candidates, args, *, executor=None):
    title_scorer = getattr(args, "title_scorer", DEFAULT_TITLE_SCORER)
    if executor is None or len(selected) < 2:
        labels, papers = _label_batch(selected, entries, candidates, args.min_score, args.min_margin, title_scorer)
    else:
        labels, papers = [], []
        batches_per_worker = 4
        target_batches = args.workers * batches_per_worker
        batch_size = min(25, (len(selected) + target_batches - 1) // target_batches)
        batches = (selected[i:i + batch_size] for i in range(0, len(selected), batch_size))

        def submit(batch):
            entries_for_batch = {ident: entries[ident] for ident in batch}
            references_for_batch = {ident: candidates[ident] for ident in batch if ident in candidates}
            future = executor.submit(_label_batch, batch, entries_for_batch, references_for_batch,
                                     args.min_score, args.min_margin, title_scorer)
            return batch, future

        # A bounded queue limits serialized data and preserves input order.
        pending = deque(submit(batch) for batch in islice(batches, 2 * args.workers))
        while pending:
            batch, future = pending.popleft()
            try:
                batch_labels, batch_papers = future.result()
            except Exception as exc:
                for _, remaining in pending:
                    remaining.cancel()
                raise RuntimeError(f"Lỗi gán nhãn lô {batch[0]} đến {batch[-1]}: {exc}") from exc
            if next_batch := next(batches, None):
                pending.append(submit(next_batch))
            labels.extend(batch_labels)
            papers.extend(batch_papers)

    expected = sum(len(entries[ident]) for ident in selected)
    keys = {(row["arxiv_id"], row["citation_key"]) for row in labels}
    if len(labels) != expected or len(keys) != expected:
        raise RuntimeError("Số nhãn hoặc số khóa duy nhất không khớp BibTeX đã chọn")
    return labels, papers


def write_outputs(directory, labels, papers, summary):
    import pyarrow as pa
    import pyarrow.parquet as pq

    file_names = ("auto_labels.parquet", "selected_papers.csv", "summary.json")
    schema = pa.schema(list(LABEL_FIELDS.items()))
    table = pa.Table.from_pylist(labels, schema=schema)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        pq.write_table(table, directory / file_names[0], compression="zstd")
        with (directory / file_names[1]).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("arxiv_id", "range", "bibtex_entries", *STATUSES))
            writer.writeheader()
            writer.writerows(papers)
        (directory / file_names[2]).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError):
        for name in file_names:
            (directory / name).unlink(missing_ok=True)
        directory.rmdir()
        raise


def run(args):
    get_title_similarity(args.title_scorer)
    if args.out.exists():
        raise ValueError(f"Thư mục đầu ra đã tồn tại: {args.out}")
    ranges = sorted(set(args.ranges))
    corpus = {ident for ident in paper_ids(args.papers) if range_of(ident) in ranges}
    manual = {arxiv_id(value) for value in (args.manual_ids or [])}
    for path in args.manual or []:
        manual.update(arxiv_id(row["arxiv_id"]) for row in csv_rows(path, ("arxiv_id",)))
    external = corpus - manual
    entries = read_bibtex(args.bibtex, corpus if args.eval_labels else external)
    candidates, coverage = read_references(args.references, entries)
    completed = read_completion(args.refs_done)
    coverage.update(args.empty_reference_ranges)
    confirmed = {ident for ident in completed if range_of(ident) in coverage}
    if args.full_corpus:
        mode = "full_corpus"
    else:
        mode = "trial" if args.quota < 500 else "official"
    summary = dict(mode=mode,
                   configuration=dict(seed=args.seed, quota=args.quota, min_score=args.min_score,
                                      min_margin=args.min_margin, ranges=ranges, workers=args.workers,
                                      title_scorer=args.title_scorer),
                   inputs={name: getattr(args, name) for name in
                           ("papers", "references", "bibtex", "refs_done", "manual", "manual_ids",
                            "empty_reference_ranges")},
                   ranges={}, corpus_outside_manual=len(external), warnings=[])
    all_labels, all_papers = [], []
    with labeling_pool(args.workers) as executor:
        for label in ranges:
            pool = {ident for ident in external if range_of(ident) == label}
            eligible = pool & entries.keys() & (candidates.keys() | confirmed)
            if args.full_corpus:
                selected, target = sorted(eligible), len(eligible)
            else:
                selected, target = select(eligible, len(pool), label, args)
            if manual.intersection(selected):
                raise RuntimeError("Mẫu tự động trùng ID thủ công")
            labels, papers = label_papers(selected, entries, candidates, args, executor=executor)
            all_labels.extend(labels)
            all_papers.extend(papers)
            statuses = Counter(row["label_status"] for row in labels)
            finalized = sum(row["needs_review"] == 0 for row in papers)
            report = dict(
                corpus=len(pool) + sum(range_of(ident) == label for ident in corpus & manual),
                corpus_outside_manual=len(pool),
                manual_excluded=sum(range_of(ident) == label for ident in corpus & manual),
                m1_references_export_observed=label in coverage,
                without_bibtex=len(pool - entries.keys()),
                bibtex_without_confirmed_m1=len((pool & entries.keys()) - eligible),
                available=len(eligible), target=target, selected=len(papers),
                selection_shortfall=target - len(papers), finalized_papers=finalized,
                finalized_shortfall=target - finalized, bibtex_entries=len(labels),
                by_status={state: statuses[state] for state in STATUSES},
                by_method=dict(Counter(row["match_method"] for row in labels)),
            )
            summary["ranges"][f"R{label}"] = report
            if not report["corpus"]:
                summary["warnings"].append(f"R{label}: thiếu publication trong bảng papers/danh sách corpus")
            if label not in coverage:
                summary["warnings"].append(f"R{label}: không xác minh được đầu ra M1 references")
            if not args.full_corpus and report["manual_excluded"] < 5:
                summary["warnings"].append(f"R{label}: mới có {report['manual_excluded']} ID thủ công trong corpus")
            if report["selection_shortfall"]:
                summary["warnings"].append(
                    f"R{label}: có {len(eligible)} bài khả dụng, thiếu {report['selection_shortfall']} bài để chọn")
    finalized = sum(row["needs_review"] == 0 for row in all_papers)
    group_target = None
    if not args.full_corpus and len(ranges) == len(RANGES):
        group_target = max(2500, (len(external) + 9) // 10)
    quota_met = None
    if not args.full_corpus:
        quota_met = (
            mode == "official"
            and all(r["finalized_shortfall"] == 0 and r["m1_references_export_observed"]
                    for r in summary["ranges"].values())
            and (group_target is None or finalized >= group_target)
        )
    summary.update(
        selected_papers=len(all_papers), bibtex_entries=len(all_labels), finalized_papers=finalized,
        needs_review=sum(row["needs_review"] for row in all_papers),
        group_target=group_target,
        group_finalized_shortfall=max(0, group_target - finalized) if group_target is not None else None,
        quota_met=quota_met,
    )
    if args.eval_labels:
        summary["inputs"]["eval_labels"] = str(args.eval_labels)
        summary["manual_evaluation"] = evaluate(args.eval_labels, manual, entries, candidates, confirmed, args)
    write_outputs(args.out, all_labels, all_papers, summary)
    for warning in summary["warnings"]:
        print(warning, file=sys.stderr)
    print(f"{mode}: {len(all_papers)} bài đã chọn, {len(all_labels)} mục BibTeX. {args.out}")
    return 1 if mode == "official" and not quota_met else 0


def main():
    args = arguments()
    try:
        return run(args)
    except ImportError as exc:
        if exc.name != "pyarrow":
            raise
        print("Thiếu PyArrow. Cài bằng python -m pip install pyarrow.", file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
