"""Compare sequential and parallel labeling on a labeled BibTeX CSV."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

from auto_label import label_papers, labeling_pool, write_outputs
from inputs import csv_rows, range_of, read_completion, read_references
from matching import DEFAULT_TITLE_SCORER, TITLE_SCORERS, arxiv_id, get_title_similarity

INPUT_FIELDS = ("arxiv_id", "citation_key", "title")
LABEL_FIELDS = ("groundtruth_arxiv_id", "label_status")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", required=True, type=Path, help="Labeled CSV with BibTeX metadata")
    parser.add_argument("--references", required=True, nargs="+", help="M1 references Parquet files")
    parser.add_argument("--refs-done", nargs="+", default=[], help="M1 scraper checkpoints")
    parser.add_argument("--out", required=True, type=Path, help="New directory for benchmark results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-score", type=float, default=.85)
    parser.add_argument("--min-margin", type=float, default=.06)
    parser.add_argument("--title-scorer", choices=TITLE_SCORERS, default=DEFAULT_TITLE_SCORER)
    args = parser.parse_args()
    if args.workers < 2 or args.repeats < 1:
        parser.error("--workers must be at least 2 and --repeats must be positive")
    if not 0 <= args.min_score <= 1 or not 0 <= args.min_margin <= 1:
        parser.error("Score and margin must be between 0 and 1")
    return args


def read_manual(path):
    entries = defaultdict(dict)
    expected = {}
    for row in csv_rows(path, INPUT_FIELDS + LABEL_FIELDS):
        if any(row[field] is None for field in (*INPUT_FIELDS, *LABEL_FIELDS)):
            raise ValueError(f"Dòng CSV thiếu giá trị: {row.get('citation_key')}")
        paper, key = arxiv_id(row["arxiv_id"]), row["citation_key"]
        if not key.strip() or (paper, key) in expected:
            raise ValueError(f"citation_key trống hoặc trùng: ({paper}, {key})")
        status = row["label_status"].strip()
        target = arxiv_id(row["groundtruth_arxiv_id"]) if row["groundtruth_arxiv_id"].strip() else None
        if status not in ("matched", "no_match") or (status == "matched") != bool(target):
            raise ValueError(f"Nhãn tay chưa hoàn tất hoặc không nhất quán: ({paper}, {key})")
        features = {field: row[field] for field in INPUT_FIELDS}
        features["arxiv_id"] = paper
        entries[paper][key] = features
        expected[paper, key] = (target, status)
    if not expected:
        raise ValueError(f"CSV không có dòng dữ liệu: {path}")
    return entries, expected


def accuracy(labels, expected):
    counts = Counter()
    for row in labels:
        target, status = expected[row["arxiv_id"], row["citation_key"]]
        prediction, predicted_status = row["groundtruth_arxiv_id"], row["label_status"]
        counts["exact_rows"] += (prediction, predicted_status) == (target, status)
        counts["assigned_ids"] += prediction is not None
        counts["correct_ids"] += prediction is not None and prediction == target
        counts["wrong_ids"] += prediction is not None and prediction != target
        counts["manual_ids"] += target is not None
        counts["missed_ids"] += target is not None and target != prediction
        counts["needs_review"] += predicted_status == "needs_review"
    assigned, manual = counts["assigned_ids"], counts["manual_ids"]
    return dict(counts, exact_row_rate=counts["exact_rows"] / len(expected),
                id_precision=counts["correct_ids"] / assigned if assigned else None,
                id_recall=counts["correct_ids"] / manual if manual else None)


def run(args):
    get_title_similarity(args.title_scorer)
    total_start = perf_counter()
    if args.out.exists():
        raise ValueError(f"Thư mục đầu ra đã tồn tại: {args.out}")
    parallel_workers = args.workers
    entries, expected = read_manual(args.manual)
    candidates, coverage = read_references(args.references, entries)
    confirmed = {ident for ident in read_completion(args.refs_done) if range_of(ident) in coverage}
    unavailable = entries.keys() - candidates.keys() - confirmed
    if unavailable:
        raise ValueError(f"Thiếu dữ liệu hoặc checkpoint references cho {len(unavailable)} bài: "
                         f"{', '.join(sorted(unavailable)[:5])}")
    input_seconds = perf_counter() - total_start

    timings = {"serial": [], "parallel": []}
    results = {}
    repeats_identical = True
    for _ in range(args.repeats):
        for mode, workers in (("serial", 1), ("parallel", parallel_workers)):
            options = argparse.Namespace(workers=workers, min_score=args.min_score,
                                         min_margin=args.min_margin, title_scorer=args.title_scorer)
            started = perf_counter()
            with labeling_pool(workers) as executor:
                current = label_papers(list(entries), entries, candidates, options, executor=executor)
            timings[mode].append(perf_counter() - started)
            first = results.setdefault(mode, current)
            repeats_identical &= current == first

    serial, parallel = results["serial"][0], results["parallel"][0]
    serial_by_key = {(r["arxiv_id"], r["citation_key"]): r for r in serial}
    parallel_by_key = {(r["arxiv_id"], r["citation_key"]): r for r in parallel}
    if serial_by_key.keys() != expected.keys() or parallel_by_key.keys() != expected.keys():
        raise RuntimeError("Khóa nhãn không khớp với CSV đối chiếu")

    summary = {
        "manual_csv": str(args.manual),
        "papers": len(entries),
        "bibtex_entries": len(expected),
        "configuration": {
            "workers": parallel_workers, "repeats": args.repeats, "min_score": args.min_score,
            "min_margin": args.min_margin, "title_scorer": args.title_scorer, "available_cpus": (
                len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
            ),
        },
        "input_seconds": input_seconds,
        "identical_predictions": results["serial"] == results["parallel"],
        "identical_across_repeats": repeats_identical,
        "runs": {},
    }
    for mode in results:
        summary["runs"][mode] = {
            "workers": 1 if mode == "serial" else parallel_workers,
            "labeling_seconds": timings[mode],
            "median_labeling_seconds": median(timings[mode]),
            "accuracy": accuracy(results[mode][0], expected),
        }
    summary["labeling_speedup"] = (
        summary["runs"]["serial"]["median_labeling_seconds"]
        / summary["runs"]["parallel"]["median_labeling_seconds"]
    )
    write_outputs(args.out, parallel, results["parallel"][1], summary)

    with (args.out / "manual_comparison.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arxiv_id", "citation_key", "manual_arxiv_id", "manual_status",
                         "serial_arxiv_id", "serial_status", "parallel_arxiv_id", "parallel_status",
                         "serial_equals_parallel", "parallel_matches_manual", "parallel_reason"])
        for key, (target, status) in expected.items():
            first, second = serial_by_key[key], parallel_by_key[key]
            prediction = (second["groundtruth_arxiv_id"], second["label_status"])
            writer.writerow([*key, target, status, first["groundtruth_arxiv_id"], first["label_status"],
                             *prediction, first == second, prediction == (target, status), second["reason"]])

    summary["total_seconds"] = perf_counter() - total_start
    summary_path = args.out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["identical_predictions"] and repeats_identical else 1


def main():
    try:
        return run(arguments())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
