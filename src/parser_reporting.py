"""Chạy parser theo bài, lưu checkpoint và xuất số liệu cho notebook parser."""
from __future__ import annotations

import ast
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import inspect
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import threading
import time
import uuid

import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from .parser_pipeline import parse_publication
from .parser_storage import (OUTPUT_FILES as _OUTPUT_FILES, atomic_json as _atomic_json, file_signature as _file_signature,
    output_manifest as _output_manifest, paper_name as _paper_name, persist_result,
    table_schemas, validate_receipt, write_tables as _write_tables)
from .utils import walk_files as _walk_files

_PAPER_REPORT_COLUMNS = [
    "arxiv_id", "status", "paper_success", "versions_expected", "versions_ok", "versions_empty",
    "versions_partial", "versions_error", "elements_before", "elements_after", "bibtex_before",
    "bibtex_after", "elapsed_sec", "reused",
]
_VERSION_REPORT_COLUMNS = ["arxiv_id", "version", "status", "message"]

# Checkpoints from the preceding parser release (pinned dependencies, LF/CRLF).
# Only accept this migration while the parsing code matches the reviewed version.
_LEGACY_CODE_SIGNATURES = (
    "54dc4b342ebda0d75900f1174523365513a9c67c0a1bcf7c790a774b644e026a",
    "e8c7468a29aaf18865e88d25f20e4edf215f830a9af6860ddc207ee0f320325c",
)
_LEGACY_COMPATIBLE_PARSER = "968da39ebccc8b694e15763b724b9ece672091cdb058422379790cccb85cb8da"
# Version-2 checkpoints from the preceding, semantically equivalent parser.
_COMPATIBLE_CODE_SIGNATURES = ("a6bf50c658e5d2d8470b94125f00d1c8eb147f84ceaad0a3cd8c80097d5e1c58",)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    if not Path(path).exists():
        return []
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Phiên bị ngắt có thể để lại dòng cuối chưa hoàn chỉnh.
            continue
    return rows


def _code_signature():
    """Fingerprint parsing semantics; reporting/scheduling edits do not invalidate data."""
    digest = hashlib.sha256()
    for name in ("parser_pipeline.py", "gather_clean_tex.py", "hierarchy.py", "bibtex.py", "m2_types.py", "utils.py"):
        tree = ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8-sig"))
        if name == "parser_pipeline.py":
            # Discovery is represented in each job's metadata, versions and errors.
            discovery = {"_arxiv_id", "_version", "_read_json", "_expected", "_state_paths", "discover_publications"}
            tree.body = [node for node in tree.body if getattr(node, "name", None) not in discovery]
        digest.update(ast.dump(tree, include_attributes=False).encode())
    digest.update(ast.dump(ast.parse(inspect.getsource(table_schemas)), include_attributes=False).encode())
    for name in ("bibtexparser", "wtpsplit", "onnxruntime", "pyarrow"):
        try:
            digest.update(f"{name}={importlib.metadata.version(name)}".encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(f"{name}=missing".encode())
    return digest.hexdigest()


def input_signature(job, options):
    """Chữ ký nội dung theo định dạng cũ, chỉ dùng khi chuyển checkpoint cũ."""
    digest = hashlib.sha256(json.dumps({"arxiv_id": job["arxiv_id"], "metadata": job.get("metadata", {}),
        "versions": {str(v): str(p) for v, p in job["version_dirs"].items()},
        "source_status": job.get("source_status_by_version", {}),
        "errors": job.get("discovery_errors", []), "options": options}, sort_keys=True).encode())
    directory = job.get("paper_dir")
    if directory is not None and Path(directory).is_dir():
        directory = Path(directory)
        for path in sorted(p for p in directory.rglob("*") if p.is_file() and
                           (p.suffix.lower() in {".tex", ".bib"} or p.name == "metadata.json")):
            digest.update(path.relative_to(directory).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


def _stat_info(stat):
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _source_manifest(job):
    """Inventory version files, including .bbl and extensionless TeX inputs."""
    files = {}
    if job.get("paper_dir") is not None:
        metadata = Path(job["paper_dir"]) / "metadata.json"
        if metadata.is_file():
            files["paper/metadata.json"] = _stat_info(metadata.stat())
    for version, directory in sorted(job["version_dirs"].items()):
        if directory is None or not Path(directory).is_dir():
            continue
        directory = Path(directory)
        for path, stat in _walk_files(directory):
            files[f"{version}/{path.relative_to(directory).as_posix()}"] = _stat_info(stat)
    return files


def _input_context(job, options):
    return json.dumps({"arxiv_id": job["arxiv_id"], "metadata": job.get("metadata", {}),
        "versions": {str(v): str(p) for v, p in job["version_dirs"].items()},
        "source_status": job.get("source_status_by_version", {}),
        "errors": job.get("discovery_errors", []), "options": options}, sort_keys=True)


def _source_signature(job, context, manifest):
    digest = hashlib.sha256(context.encode())
    for name in sorted(manifest):
        version, relative = name.split("/", 1)
        directory = job["paper_dir"] if version == "paper" else job["version_dirs"][int(version)]
        path = Path(directory) / relative
        digest.update(json.dumps([name, manifest[name]["size"]]).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _parquet_rows(path):
    # Close readers explicitly before a later run replaces a Parquet file on Windows.
    with pq.ParquetFile(path) as source:
        return source.metadata.num_rows


def _prepare_checkpoint(job, partition, stamp, options, *, resume, validation, retry_partial):
    context = _input_context(job, options)
    context_signature = hashlib.sha256(context.encode()).hexdigest()
    manifest = _source_manifest(job)
    record = None
    if resume:
        try:
            candidate = json.loads(stamp.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                record = candidate
        except (OSError, ValueError):
            pass
    original = json.dumps(record, sort_keys=True) if record is not None else None
    migrated = False
    if (validation == "fast" and record is not None
            and record.get("input_context") == context_signature
            and record.get("input_manifest") == manifest
            and isinstance(record.get("signature"), str)):
        signature = record["signature"]
    else:
        signature = _source_signature(job, context, manifest)
        compatible = record is not None and options["code_signature"] == _LEGACY_COMPATIBLE_PARSER
        if compatible and "input_context" in record:
            for prior_code in _COMPATIBLE_CODE_SIGNATURES:
                prior_context = _input_context(job, {**options, "code_signature": prior_code})
                if (hashlib.sha256(prior_context.encode()).hexdigest() == record["input_context"]
                        and _source_signature(job, prior_context, manifest) == record.get("signature")):
                    record["signature"] = signature
                    migrated = True
                    break
        if (compatible and "input_context" not in record
                and all(name == "paper/metadata.json" or Path(name).suffix.lower() in {".tex", ".bib"}
                        for name in manifest)):
            # A one-time full source/output validation upgrades old checkpoints.
            # Never migrate legacy inputs such as .bbl that the old hash ignored.
            for legacy_code in _LEGACY_CODE_SIGNATURES:
                if input_signature(job, {**options, "code_signature": legacy_code}) == record.get("signature"):
                    record["signature"] = signature
                    migrated = True
                    break
    record = _checkpoint(stamp, partition, signature, record=record,
                         validation="full" if migrated else validation,
                         retry_partial=retry_partial) if record is not None else None
    if record is not None:
        record.update(input_context=context_signature, input_manifest=manifest)
        if json.dumps(record, sort_keys=True) != original:
            _atomic_json(stamp, record)
    return signature, context_signature, manifest, record


class _Monitor:
    """Keep peak RSS in memory; write individual samples only in debug mode."""
    def __init__(self, path, run_id, worker, interval):
        self.path = Path(path) if path is not None else None
        self.run_id, self.worker = run_id, worker
        self.interval = max(0.05, interval)
        self.phase = "initializing"
        self.stop = threading.Event()
        self.peak, self.total, self.count = 0, 0, 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self, writer=None, stream=None):
        rss = 0
        process = psutil.Process()
        for item in [process, *process.children(recursive=True)]:
            try:
                rss += item.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.peak = max(self.peak, rss)
        self.total += rss
        self.count += 1
        if writer is not None:
            writer.writerow({"timestamp": utc_now(), "run_id": self.run_id, "worker": self.worker,
                             "phase": self.phase, "rss_bytes": rss})
            stream.flush()

    def _loop(self):
        if self.path is None:
            self._sample()
            while not self.stop.wait(self.interval):
                self._sample()
            return
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["timestamp", "run_id", "worker", "phase", "rss_bytes"])
            writer.writeheader()
            self._sample(writer, stream)
            while not self.stop.wait(self.interval):
                self._sample(writer, stream)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self._sample()


def _disk_snapshot(root, logs, run_id, worker):
    started = time.perf_counter()
    size = 0
    for path, stat in _walk_files(root, ignore_missing=True):
        size += stat.st_size
    row = {"timestamp": utc_now(), "run_id": run_id, "worker": worker,
           "output_bytes": size, "scan_sec": time.perf_counter() - started}
    path = logs / f"disk_{run_id}.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def _checkpoint(state, partition, signature, *, record=None, validation="full", retry_partial=False):
    try:
        if record is None:
            record = json.loads(state.read_text(encoding="utf-8"))
        accepted = {"ok", "empty"} if retry_partial else {"ok", "empty", "partial"}
        if record["signature"] != signature or record["stats"]["status"] not in accepted:
            return None
        manifest = _output_manifest(partition)
        if validation != "fast" or record.get("output_manifest") != manifest:
            for filename in _OUTPUT_FILES:
                if _file_signature(partition / filename) != record["file_signatures"][filename]:
                    return None
            for name in table_schemas():
                if _parquet_rows(partition / f"{name}.parquet") != record["table_counts"][name]:
                    return None
        record["output_manifest"] = manifest
        record["partition"] = str(partition)
        return record
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
        return None


def _combine_tables(records, tables, schemas, *, cache_state=None, validation="full"):
    """Ghép từng batch, không nạp toàn bộ corpus vào RAM."""
    identity = hashlib.sha256(json.dumps([
        [record["stats"]["arxiv_id"], {name: record["file_signatures"][f"{name}.parquet"] for name in schemas}]
        for record in records], sort_keys=True).encode()).hexdigest()
    if cache_state is not None and validation == "fast":
        try:
            cached = json.loads(cache_state.read_text(encoding="utf-8"))
            if cached["signature"] == identity and cached["files"] == {
                    name: _stat_info((tables / f"{name}.parquet").stat()) for name in schemas}:
                return True
        except (OSError, ValueError, KeyError, TypeError):
            pass
    for name, schema in schemas.items():
        temporary = tables / f"{name}.parquet.tmp"
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for record in records:
                with pq.ParquetFile(Path(record["partition"]) / f"{name}.parquet") as source:
                    for batch in source.iter_batches(batch_size=8192):
                        writer.write_table(pa.Table.from_batches([batch]).cast(schema))
        os.replace(temporary, tables / f"{name}.parquet")
    if cache_state is not None:
        _atomic_json(cache_state, {"signature": identity, "files": {
            name: _stat_info((tables / f"{name}.parquet").stat()) for name in schemas}})
    return False


def _export_csv(path, rows, columns, *, include_extra=True):
    frame = pd.DataFrame(rows) if include_extra else pd.DataFrame(rows, columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _core_stats(paper):
    total = lambda name: int(pd.to_numeric(paper[name], errors="coerce").sum())
    times = pd.to_numeric(paper.elapsed_sec, errors="coerce").dropna()
    summary = {
        "target_ids": len(paper), "papers_success": int(paper.paper_success.fillna(False).sum()),
        "success_rate": float(paper.paper_success.fillna(False).mean()) if len(paper) else 0.0,
        "statuses": dict(Counter(paper.status)),
        "versions": {name: total(name) for name in
                     ("versions_expected", "versions_ok", "versions_empty", "versions_partial", "versions_error")},
        "average_parse_sec": float(times.mean()) if len(times) else None,
    }
    for name in ("elements", "bibtex"):
        before, after = total(f"{name}_before"), total(f"{name}_after")
        summary.update({f"{name}_before": before, f"{name}_after": after,
                        f"{name}_reduction_ratio": (before - after) / before if before else 0.0})
    return summary


def _report_run(run, debug_reports):
    if debug_reports:
        return run
    return {key: run[key] for key in (
        "run_id", "worker", "started_at", "ended_at", "duration_sec", "stop_reason",
        "papers_processed", "papers_reused", "remaining", "peak_rss_bytes", "machine", "error",
    ) if key in run}


def _exports(reports, logs, records, pending, ranges, run, config, tables):
    if config.get("debug_reports", False):
        summary = _debug_exports(reports, logs, records, pending, ranges, run, config, tables)
        summary["report_mode"] = "debug"
        times = [record["stats"]["elapsed_sec"] for record in records]
        summary["average_parse_sec"] = sum(times) / len(times) if times else None
        return summary
    rows = [{**record["stats"], "reused": record["reused"], "current_run_sec": record["current_run_sec"]}
            for record in records]
    rows.extend({"arxiv_id": job["arxiv_id"], "status": "pending", "paper_success": False,
                 "versions_expected": len(job["version_dirs"]), "reused": False, "current_run_sec": 0,
                 "message": "Not processed in this run"} for job in pending)
    paper = _export_csv(reports / "per_paper.csv", rows, _PAPER_REPORT_COLUMNS, include_extra=False)
    versions = [row for record in records for row in record["version_status"]]
    versions.extend({"arxiv_id": job["arxiv_id"], "version": version, "status": "pending",
                     "message": "Not processed in this run"}
                    for job in pending for version in job["version_dirs"])
    _export_csv(reports / "per_version.csv", versions, _VERSION_REPORT_COLUMNS, include_extra=False)
    summary = {"report_mode": "minimal", "exported_at": utc_now(),
        "scope": {key: value for key, value in config.items() if key not in {"target_ids", "ranges"}},
        "run": _report_run(run, False), **_core_stats(paper),
        "table_bytes": sum((tables / f"{name}.parquet").stat().st_size for name in table_schemas()),
        "table_rows": {name: sum(record["table_counts"][name] for record in records) for name in table_schemas()},
        "measurement_note": "duration_sec is current-run wall time; average_parse_sec is the mean saved/performed "
                            "per-paper parse time, including cached results. Final benchmark: Colab CPU, resume=False. "
                            "peak_rss_bytes is sampled Python+child RSS. table_bytes counts the three dataset tables."}
    _atomic_json(reports / "summary_stats.json", summary)
    (reports / "README.md").write_text(
        "# Milestone 2 report\n\n"
        "- per_paper.csv: success, version coverage, element/reference counts before/after deduplication, and parse time.\n"
        "- per_version.csv: each version's status, including empty, partial, error, and pending.\n"
        "- summary_stats.json: totals, average parse time, current-run wall time, peak RAM, and final dataset size.\n"
        "- Reused partial results remain incomplete. Saved parse times are distinct from current resume wall time.\n"
        "- table_bytes counts only the three dataset tables; checkpoints and logs are working files.\n"
        "- Detailed diagnostics stay in tables/<worker>/papers/<ID>/warnings.jsonl.\n"
        "- Set DEBUG_REPORTS=True to export warning/range/category summaries and monitor/timing histories.\n"
        "- Optional reports left by older runs are not part of this minimal export.\n",
        encoding="utf-8")
    return summary


def _debug_exports(reports, logs, records, pending, ranges, run, config, tables):
    rows = [{**record["stats"], "range": record["range"], "reused": record["reused"],
             "current_run_sec": record["current_run_sec"]} for record in records]
    for job in pending:
        rows.append({"arxiv_id": job["arxiv_id"], "range": job.get("range", "local"), "status": "pending",
                     "paper_success": False, "versions_expected": len(job["version_dirs"]),
                     "current_run_sec": 0, "reused": False})
    paper = _export_csv(reports / "per_paper.csv", rows,
        ["arxiv_id", "range", "status", "paper_success", "versions_expected", "versions_ok", "versions_empty",
         "versions_partial", "versions_error", "elements_before", "elements_after", "bibtex_before",
         "bibtex_after", "elapsed_sec", "input_bytes", "category", "warnings_count", "blocking_warning_count", "notice_count",
         "review_required", "bibliography_status", "versions_missing_bibliography", "message", "reused", "current_run_sec"])
    versions = [row for record in records for row in record["version_status"]]
    versions.extend({"arxiv_id": job["arxiv_id"], "version": version, "status": "pending",
                     "message": "Not processed in this run", "missing_count": 0, "warning_count": 0}
                    for job in pending for version in job["version_dirs"])
    _export_csv(reports / "per_version.csv", versions,
                ["arxiv_id", "version", "status", "message", "missing_count", "warning_count", "blocking_warning_count", "notice_count", "review_required",
                 "bibliography_status", "bibliography_missing", "required_missing_count"])
    _export_csv(reports / "issues.csv", [row for row in versions if row["status"] not in {"ok", "empty"}],
                ["arxiv_id", "version", "status", "message", "missing_count", "warning_count"])
    diagnostic_columns = ["arxiv_id", "version", "stage", "category", "severity", "affects_status", "requires_review", "message"]
    with (reports / "warnings.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=diagnostic_columns, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            with (Path(record["partition"]) / "warnings.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    writer.writerow(json.loads(line))
    _export_csv(reports / "ranges.csv", [{"range": f"R{key}", "month": value[0], "first_id": value[1],
                "last_id": value[2]} for key, value in ranges.items()], ["range", "month", "first_id", "last_id"])
    range_rows = []
    for label, frame in paper.groupby("range", dropna=False):
        row = {"range": label, "target_ids": len(frame), "papers_success": int(frame.paper_success.fillna(False).sum())}
        row["success_rate"] = row["papers_success"] / row["target_ids"]
        for column in ["versions_expected", "versions_ok", "versions_empty", "versions_partial", "versions_error",
                       "elements_before", "elements_after", "bibtex_before", "bibtex_after", "input_bytes", "current_run_sec"]:
            row[column] = float(pd.to_numeric(frame[column], errors="coerce").sum())
        for column in ("input_bytes", "bibtex_after", "elapsed_sec"):
            row[f"mean_{column}"] = float(pd.to_numeric(frame[column], errors="coerce").mean()) if len(frame) else None
        range_rows.append(row)
    _export_csv(reports / "range_stats.csv", range_rows, ["range", "target_ids", "papers_success", "success_rate"])
    categories = Counter(row.get("category") or "unknown" for row in rows
                         if row["status"] not in {"pending", "missing_input"})
    _export_csv(reports / "category_counts.csv", [{"category": k, "paper_count": v} for k, v in sorted(categories.items())],
                ["category", "paper_count"])
    runs = {}
    for row in _read_jsonl(logs / "runs.jsonl"):
        runs.setdefault(row["run_id"], {}).update(row)
    _export_csv(reports / "runs.csv", list(runs.values()), ["run_id", "worker", "started_at", "ended_at", "duration_sec", "stop_reason"])
    samples = [pd.read_csv(path) for path in sorted(logs.glob("monitor_*.csv")) if path.stat().st_size]
    if samples:
        samples = pd.concat(samples, ignore_index=True)
        samples["minute"] = pd.to_datetime(samples.timestamp, utc=True).dt.floor("min")
        samples.groupby(["worker", "run_id", "minute"], as_index=False).agg(
            mean_rss_bytes=("rss_bytes", "mean"), max_rss_bytes=("rss_bytes", "max"), samples=("rss_bytes", "size")
        ).to_csv(reports / "monitor.csv", index=False)
    disks = [pd.read_csv(path) for path in sorted(logs.glob("disk_*.csv"))]
    if disks:
        pd.concat(disks, ignore_index=True).to_csv(reports / "disk.csv", index=False)
    total = lambda name: int(pd.to_numeric(paper[name], errors="coerce").sum())
    before, after = total("elements_before"), total("elements_after")
    bib_before, bib_after = total("bibtex_before"), total("bibtex_after")
    summary = {"exported_at": utc_now(), "scope": config, "run": run,
        "target_ids": len(paper), "papers_attempted": len(records),
        "papers_success": int(paper.paper_success.fillna(False).sum()),
        "success_rate": float(paper.paper_success.fillna(False).mean()) if len(paper) else 0.0,
        "statuses": dict(Counter(row["status"] for row in rows)),
        "versions": {name: total(name) for name in ["versions_expected", "versions_ok", "versions_empty", "versions_partial", "versions_error"]},
        "version_count_distribution": dict(Counter(str(row.get("versions_expected", 0)) for row in rows
                                                    if row["status"] not in {"missing_input", "pending"})),
        "unknown_or_pending_version_counts": sum(row["status"] in {"missing_input", "pending"} for row in rows),
        "elements_before": before, "elements_after": after,
        "elements_reduction_ratio": (before - after) / before if before else 0.0,
        "bibtex_before": bib_before, "bibtex_after": bib_after,
        "bibtex_reduction_ratio": (bib_before - bib_after) / bib_before if bib_before else 0.0,
        "input_bytes": total("input_bytes"),
        "table_bytes": sum((tables / f"{name}.parquet").stat().st_size for name in table_schemas()),
        "table_rows": {name: _parquet_rows(tables / f"{name}.parquet") for name in table_schemas()},
        "recorded_parse_sec": float(pd.to_numeric(paper.elapsed_sec, errors="coerce").sum()),
        "current_run_processing_sec": float(paper.current_run_sec.sum()),
        "diagnostics": {"total": total("warnings_count"), "blocking": total("blocking_warning_count"),
                        "notices": total("notice_count"), "papers_for_review": int(paper.review_required.fillna(False).sum())},
        "bibliography": {"papers_missing_sources": int(paper.bibliography_status.eq("missing_sources").sum()),
                         "versions_missing_sources": total("versions_missing_bibliography")},
        "current_run_average_sec": run["duration_sec"] / run["papers_processed"] if run["papers_processed"] else None,
        "measurement_note": "RSS samples cover the Python process and its children; disk usage is a snapshot of the output directory. Checkpoint parsing times are reported separately from current-run times."}
    _atomic_json(reports / "summary_stats.json", summary)
    (reports / "README.md").write_text(
        "# Milestone 2 Statistics\n\n"
        "The report covers exactly the configured IDs, including IDs with missing data and papers not yet processed (pending).\n\n"
        "- per_paper.csv: paper status, version counts, element/reference counts before and after deduplication, source size, timings, and warnings.\n"
        "- per_version.csv: status of each version; empty is distinct from missing_dir and parsing errors.\n"
        "- issues.csv: versions with missing data, errors, or pending processing; warnings.csv: all diagnostics, distinguishing notices from warnings that affect result completeness.\n"
        "- ok may include notices/review_required: the pipeline completed, but metadata/schema may need review. Notices do not invalidate checkpoints.\n"
        "- Missing .bib/.bbl files do not set paper_success=False if the main content was processed. bibliography_status=missing_sources identifies skipped sources; references with available sources are still extracted.\n"
        "- summary_stats.json: scope, configuration, summary, and machine information for the current run.\n"
        "- range_stats.csv, ranges.csv: statistics and definitions for ID ranges. category_counts.csv uses the first category.\n"
        "- runs.csv: worker run history; monitor.csv: mean/maximum RSS per minute; disk.csv: output snapshots and scan durations.\n"
        "- The three combined Parquet tables are in tables/<worker>/; per-paper partitions are in papers/. Combined tables and partitions contain overlapping data and should not be counted as separate scientific datasets.\n"
        "- partial does not count as full success; valid partial checkpoints are reused unless retry_partial=True. Check per_version.csv before use.\n"
        "- reused=True: a valid checkpoint was reused; elapsed_sec is the saved parsing time, and current_run_sec is the time spent in the current run.\n"
        "- checkpoint_validation=fast compares size/mtime before falling back to hashes on changes; full always hashes source/output content.\n"
        "- Raw runs/timings/events logs and monitor/disk samples are in run_logs/<worker>/.\n",
        encoding="utf-8")
    return summary


def run_parser(jobs, output_dir, *, worker="local", ranges=None, enrich=False, fuzzy_threshold=0.92,
               resume=True, max_run_hours=None, monitor_interval=1.0, disk_every=None, progress=None,
               num_workers=1, max_workers=None, adaptive_ram=True, reserve_ram_gb=None,
               ram_per_worker_gb=1.5, ram_wait_timeout_sec=60, retry_partial=False,
               checkpoint_validation="fast", checkpoint_progress=None, debug_reports=False,
               sentence_cache_mb=4):
    """Resume cả partial hợp lệ; mặc định chỉ xuất các thống kê yêu cầu.

    fast kiểm tra size/mtime, chỉ đọc nội dung khi tệp đổi hoặc checkpoint cũ.
    full luôn kiểm tra hash, phát hiện cả thay đổi giữ nguyên size/mtime.
    retry_partial=True chạy lại partial dù nguồn không đổi; error vẫn được thử lại.
    debug_reports=True bật báo cáo chi tiết, lịch sử monitor/timing và quét đĩa.
    """
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", worker):
        raise ValueError("Worker name may contain only letters, digits, _, +, and -")
    if (not 0 <= fuzzy_threshold <= 1 or monitor_interval <= 0
            or (disk_every is not None and (type(disk_every) is not int or disk_every < 1))):
        raise ValueError("Invalid threshold or measurement interval configuration")
    if type(retry_partial) is not bool or type(resume) is not bool:
        raise ValueError("resume and retry_partial must be booleans")
    if type(debug_reports) is not bool:
        raise ValueError("debug_reports must be a boolean")
    if (isinstance(sentence_cache_mb, bool) or not isinstance(sentence_cache_mb, (int, float))
            or not math.isfinite(sentence_cache_mb) or sentence_cache_mb < 0):
        raise ValueError("sentence_cache_mb must be a finite nonnegative number")
    sentence_cache_bytes = int(sentence_cache_mb * 1024 ** 2)
    if checkpoint_validation not in {"fast", "full"}:
        raise ValueError("checkpoint_validation must be 'fast' or 'full'")
    if checkpoint_progress is not None and not callable(checkpoint_progress):
        raise ValueError("checkpoint_progress must be callable")
    if max_run_hours is not None and max_run_hours <= 0:
        raise ValueError("Time limit must be positive")
    if num_workers != "auto" and (type(num_workers) is not int or num_workers < 1):
        raise ValueError("num_workers must be 'auto' or a positive integer")
    if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
        raise ValueError("max_workers must be a positive integer or None")
    if type(adaptive_ram) is not bool:
        raise ValueError("adaptive_ram must be a boolean")
    for name, value, allow_zero in (("reserve_ram_gb", reserve_ram_gb, True),
                                    ("ram_per_worker_gb", ram_per_worker_gb, False),
                                    ("ram_wait_timeout_sec", ram_wait_timeout_sec, False)):
        if value is None and allow_zero:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"Invalid {name}")
    jobs = list(jobs)
    if len({job["arxiv_id"] for job in jobs}) != len(jobs):
        raise ValueError("Paper list contains duplicate IDs")
    root = Path(output_dir).resolve()
    tables, reports, logs, state = [root / name / worker for name in ("tables", "report_exports", "run_logs", ".state")]
    for directory in (tables, reports, logs, state):
        directory.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"parser.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO if debug_reports else logging.WARNING)
    logger.propagate = False
    handler = logging.FileHandler(logs / "parser.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    options = {"enrich": enrich, "fuzzy_threshold": fuzzy_threshold, "code_signature": _code_signature()}
    config = {**options, "worker": worker, "output_dir": str(root), "resume": resume,
              "max_run_hours": max_run_hours, "monitor_interval": monitor_interval,
              "disk_every": disk_every, "target_ids": [job["arxiv_id"] for job in jobs],
              "ranges": ranges or {}, "num_workers": num_workers, "max_workers": max_workers,
              "adaptive_ram": adaptive_ram, "reserve_ram_gb": reserve_ram_gb,
              "ram_per_worker_gb": ram_per_worker_gb, "ram_wait_timeout_sec": ram_wait_timeout_sec,
              "retry_partial": retry_partial, "checkpoint_validation": checkpoint_validation,
              "debug_reports": debug_reports, "sentence_cache_mb": sentence_cache_mb}
    if not debug_reports:
        config.pop("target_ids")
        config.pop("ranges")
    run = {"run_id": run_id, "worker": worker, "started_at": utc_now(), "event": "start", "config": config,
           "target_ids": len(jobs), "remaining_at_start": None, "papers_processed": 0, "papers_reused": 0,
           "partials_reused": 0,
           "machine": {"python": platform.python_version(), "platform": platform.platform(),
                       "cpu_count": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total}}
    _append_jsonl(logs / "runs.jsonl", run)
    records = []
    completed_ids = set()
    summary = None
    started = time.perf_counter()
    stop_reason = "completed"
    schemas = table_schemas()
    monitor = _Monitor(logs / f"monitor_{run_id}.csv" if debug_reports else None, run_id, worker, monitor_interval)

    def parallel_event(event):
        if debug_reports:
            _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id, "worker": worker, **event})
            logger.info("Parallel scheduling: %s", json.dumps(event, ensure_ascii=False))
        run["parallel"] = {**run.get("parallel", {}), **event}

    def save_result(prepared_item, result=None, paper_started=None):
        """Chỉ tiến trình chính ghi bảng/checkpoint và gọi callback tiến độ."""
        job, partition, stamp, signature, context, manifest, record = prepared_item
        reused = record is not None
        write_started = time.perf_counter()
        remote_write_sec = 0
        if not reused:
            if result.get("_partition_receipt") is not True:
                result = persist_result(partition, result, debug_reports=debug_reports, consume=True)
            else:
                remote_write_sec = result.get("write_sec", 0)
            validate_receipt(result, partition, job["arxiv_id"])
            record = {"signature": signature, "stats": result["stats"],
                      "version_status": result["version_status"], "partition": str(partition),
                      "input_context": context, "input_manifest": manifest,
                      "output_manifest": result["output_manifest"],
                      "table_counts": result["table_counts"], "file_signatures": result["file_signatures"]}
            _atomic_json(stamp, record)
            for timing in result.get("timings", []) if debug_reports else ():
                _append_jsonl(logs / "timings.jsonl", {"run_id": run_id, "worker": worker,
                    "arxiv_id": job["arxiv_id"], **timing})
            run["papers_processed"] += 1
        else:
            run["papers_reused"] += 1
            run["partials_reused"] += int(record["stats"]["status"] == "partial")
        duration = (time.perf_counter() - paper_started if paper_started is not None else
                    time.perf_counter() - write_started + (0 if reused else record["stats"]["elapsed_sec"] + remote_write_sec))
        # Source/output manifests belong on disk, not in every in-memory report row.
        report_record = {key: record[key] for key in ("stats", "version_status", "partition", "table_counts", "file_signatures")}
        report_record.update(range=job.get("range", "local"), reused=reused, current_run_sec=duration)
        records.append(report_record)
        completed_ids.add(job["arxiv_id"])
        logger.info("%s: %s; reused=%s; %s", job["arxiv_id"], record["stats"]["status"], reused,
                    record["stats"].get("message", ""))
        if debug_reports:
            _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
                "arxiv_id": job["arxiv_id"], "status": record["stats"]["status"], "reused": reused,
                "message": record["stats"].get("message", "")})
        if progress:
            progress(len(records), len(jobs), record["stats"])
        if debug_reports and disk_every is not None and len(records) % disk_every == 0:
            _disk_snapshot(root, logs, run_id, worker)

    def combine_completed():
        order = {job["arxiv_id"]: index for index, job in enumerate(jobs)}
        records.sort(key=lambda record: order[record["stats"]["arxiv_id"]])
        run["combined_tables_reused"] = _combine_tables(records, tables, schemas,
            cache_state=state / "_combined_tables.json", validation=checkpoint_validation)

    def worker_job(item):
        job, _, _, _, _, manifest, _ = item
        return {**job, "_source_manifest": manifest} if manifest is not None else job
    try:
        with monitor:
            if debug_reports and disk_every is not None:
                _disk_snapshot(root, logs, run_id, worker)
            # Kiểm tra cache trước để số việc còn lại lúc bắt đầu có ý nghĩa.
            prepared = []
            monitor.phase = "checking_checkpoints"
            checking_started = time.perf_counter()
            reused_count = 0
            for checked, job in enumerate(jobs, 1):
                if max_run_hours is not None and time.perf_counter() - started >= max_run_hours * 3600:
                    stop_reason = "time_limit"
                    break
                partition = tables / "papers" / _paper_name(job["arxiv_id"])
                stamp = state / f"{_paper_name(job['arxiv_id'])}.json"
                try:
                    signature, context, manifest, record = _prepare_checkpoint(job, partition, stamp, options,
                        resume=resume, validation=checkpoint_validation, retry_partial=retry_partial)
                except OSError as exc:
                    # Vẫn để parser ghi lỗi riêng của bài; không dùng checkpoint.
                    signature = uuid.uuid4().hex
                    context, manifest, record = None, None, None
                    logger.warning("Could not hash sources for %s: %s", job["arxiv_id"], exc)
                prepared.append((job, partition, stamp, signature, context, manifest, record))
                reused_count += int(record is not None)
                if checkpoint_progress:
                    checkpoint_progress(checked, len(jobs), reused_count)
            run["checkpoint_check_sec"] = time.perf_counter() - checking_started
            run["checkpoint_checked"] = len(prepared)
            run["remaining_at_start"] = len(jobs) - reused_count
            if debug_reports:
                _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
                              "status": "checkpoint_checked", "remaining_at_start": run["remaining_at_start"]})
            if num_workers == 1:
                for index, item in enumerate(prepared):
                    if (item[-1] is None and max_run_hours is not None
                            and time.perf_counter() - started >= max_run_hours * 3600):
                        stop_reason = "time_limit"
                        break
                    paper_started = time.perf_counter()
                    job, record = item[0], item[-1]
                    monitor.phase = job["arxiv_id"]
                    result = parse_publication(worker_job(item), enrich=enrich, fuzzy_threshold=fuzzy_threshold,
                        sentence_cache_bytes=sentence_cache_bytes) if record is None else None
                    save_result(item, result, paper_started)
                    prepared[index] = None
                    del result, item
                prepared.clear()
                item = record = None
            else:
                from .parser_parallel import iter_parallel_results
                pending_items = []
                for item in prepared:
                    if item[-1] is not None:
                        save_result(item)
                    else:
                        pending_items.append(item)
                prepared.clear()
                item = None
                deadline = started + max_run_hours * 3600 if max_run_hours is not None else None
                monitor.phase = "parallel_parsing"
                with closing(iter_parallel_results((worker_job(item) for item in pending_items),
                        num_workers=num_workers, max_workers=max_workers, adaptive_ram=adaptive_ram,
                        reserve_ram_gb=reserve_ram_gb, ram_per_worker_gb=ram_per_worker_gb,
                        ram_wait_timeout_sec=ram_wait_timeout_sec, enrich=enrich,
                        fuzzy_threshold=fuzzy_threshold, deadline=deadline, on_event=parallel_event,
                        sentence_cache_bytes=sentence_cache_bytes, result_dir=tables / "papers",
                        debug_reports=debug_reports)) as results:
                    for index, result in results:
                        save_result(pending_items[index], result)
                        pending_items[index] = None
                        del result
                if len(records) < len(jobs) and deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_limit"
                pending_items.clear()
            monitor.phase = "exporting_tables"
            combine_completed()
            if not debug_reports:
                run.update(ended_at=utc_now(), duration_sec=time.perf_counter() - started,
                           stop_reason=stop_reason, peak_rss_bytes=monitor.peak, remaining=len(jobs) - len(records))
                summary = _exports(reports, logs, records,
                    [job for job in jobs if job["arxiv_id"] not in completed_ids], ranges or {}, run, config, tables)
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        combine_completed()
    except Exception as exc:
        stop_reason = "error"
        logger.exception("Run stopped due to an error")
        _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
                      "status": "error", "message": f"{type(exc).__name__}: {exc}"})
        combine_completed()
        run["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        run.update(event="end", ended_at=utc_now(), duration_sec=time.perf_counter() - started,
                   stop_reason=stop_reason, peak_rss_bytes=monitor.peak,
                   mean_rss_bytes=monitor.total / monitor.count if monitor.count else None,
                   remaining=len(jobs) - len(records))
        _append_jsonl(logs / "runs.jsonl", run)
        handler.close()
        logger.removeHandler(handler)
    if summary is None:
        summary = _exports(reports, logs, records, [job for job in jobs if job["arxiv_id"] not in completed_ids],
                           ranges or {}, run, config, tables)
    if debug_reports:
        summary["output_snapshot"] = _disk_snapshot(root, logs, run_id, worker)
        # Full history export is only requested in debug mode.
        pd.concat([pd.read_csv(p) for p in sorted(logs.glob("disk_*.csv"))], ignore_index=True).to_csv(reports / "disk.csv", index=False)
    summary["run"] = _report_run(run, debug_reports)
    _atomic_json(reports / "summary_stats.json", summary)
    return {"summary": summary, "reports_dir": str(reports), "tables_dir": str(tables), "logs_dir": str(logs),
            "tables": {name: str(tables / f"{name}.parquet") for name in schemas}}


def combine_worker_reports(output_dir, *, debug_reports=False):
    """Combine required statistics only; optional histories require debug_reports=True."""
    if type(debug_reports) is not bool:
        raise ValueError("debug_reports must be a boolean")
    base = Path(output_dir) / "report_exports"
    sources = sorted(path for path in base.glob("*/summary_stats.json") if path.parent.name != "_combined")
    if not sources:
        raise FileNotFoundError("No worker reports available to combine")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    if debug_reports:
        if any(summary.get("report_mode") == "minimal" for summary in summaries):
            raise ValueError("Debug combination requires DEBUG_REPORTS=True reports from every worker")
        result = _combine_debug_worker_reports(output_dir)
        result["summary"]["report_mode"] = "debug"
        _atomic_json(Path(result["reports_dir"]) / "summary_stats.json", result["summary"])
        return result
    configurations = {json.dumps({name: summary["scope"][name] for name in
                      ("enrich", "fuzzy_threshold", "code_signature")}, sort_keys=True) for summary in summaries}
    if len(configurations) > 1:
        raise ValueError("Workers use different code or parsing configurations; rerun with matching configurations before combining")
    papers, versions = [], []
    for path, summary in zip(sources, summaries):
        paper = pd.read_csv(path.parent / "per_paper.csv", dtype={"arxiv_id": str})
        paper["source_worker"], paper["exported_at"] = path.parent.name, summary["exported_at"]
        papers.append(paper)
        version = pd.read_csv(path.parent / "per_version.csv", dtype={"arxiv_id": str})
        version["source_worker"] = path.parent.name
        versions.append(version)
    paper = pd.concat(papers, ignore_index=True).sort_values(["exported_at", "source_worker"])
    duplicate_count = int(paper.duplicated("arxiv_id", keep="last").sum())
    paper = paper.drop_duplicates("arxiv_id", keep="last").sort_values("arxiv_id")
    version = pd.concat(versions, ignore_index=True).merge(
        paper[["arxiv_id", "source_worker"]], on=["arxiv_id", "source_worker"], how="inner")
    target = base / "_combined"
    target.mkdir(parents=True, exist_ok=True)
    paper[_PAPER_REPORT_COLUMNS + ["source_worker"]].to_csv(target / "per_paper.csv", index=False, encoding="utf-8-sig")
    version[_VERSION_REPORT_COLUMNS + ["source_worker"]].to_csv(target / "per_version.csv", index=False, encoding="utf-8-sig")
    note = "Duplicate IDs use the latest worker export. Only statistics are combined; worker dataset tables are not merged."
    summary = {"report_mode": "minimal", "exported_at": utc_now(), "workers": [p.parent.name for p in sources],
        "scope": json.loads(next(iter(configurations))), "duplicate_rows_removed": duplicate_count,
        **_core_stats(paper), "worker_performance": [
            {**_report_run(item["run"], False), "table_bytes": item["table_bytes"]} for item in summaries],
        "note": note}
    _atomic_json(target / "summary_stats.json", summary)
    (target / "README.md").write_text(note + "\nPerformance and dataset size are reported separately for each worker.\n", encoding="utf-8")
    return {"summary": summary, "reports_dir": str(target)}


def _combine_debug_worker_reports(output_dir):
    """Ghép số liệu theo ID; không cộng đôi khi các worker có phạm vi chồng nhau."""
    base = Path(output_dir) / "report_exports"
    sources = sorted(path for path in base.glob("*/summary_stats.json") if path.parent.name != "_combined")
    if not sources:
        raise FileNotFoundError("No worker reports available to combine")
    configurations = set()
    papers, versions, runs, monitors, disks, definitions = [], [], [], [], [], []
    for path in sources:
        summary = json.loads(path.read_text(encoding="utf-8"))
        configurations.add(json.dumps({name: summary["scope"][name] for name in
                                       ("enrich", "fuzzy_threshold", "code_signature")}, sort_keys=True))
        frame = pd.read_csv(path.parent / "per_paper.csv", dtype={"arxiv_id": str})
        frame["source_worker"] = path.parent.name
        frame["exported_at"] = summary["exported_at"]
        papers.append(frame)
        frame = pd.read_csv(path.parent / "per_version.csv", dtype={"arxiv_id": str})
        frame["source_worker"] = path.parent.name
        versions.append(frame)
        for filename, collection in [("runs.csv", runs), ("monitor.csv", monitors),
                                      ("disk.csv", disks), ("ranges.csv", definitions)]:
            candidate = path.parent / filename
            if candidate.exists():
                collection.append(pd.read_csv(candidate))
    if len(configurations) > 1:
        raise ValueError("Workers use different code or parsing configurations; rerun with matching configurations before combining")
    paper = pd.concat(papers, ignore_index=True).sort_values(["exported_at", "source_worker"])
    duplicate_count = int(paper.duplicated("arxiv_id", keep="last").sum())
    paper = paper.drop_duplicates("arxiv_id", keep="last").sort_values("arxiv_id")
    version = pd.concat(versions, ignore_index=True).merge(paper[["arxiv_id", "source_worker"]],
                                                          on=["arxiv_id", "source_worker"], how="inner")
    target = base / "_combined"
    target.mkdir(parents=True, exist_ok=True)
    paper.to_csv(target / "per_paper.csv", index=False, encoding="utf-8-sig")
    version.to_csv(target / "per_version.csv", index=False, encoding="utf-8-sig")
    version.loc[~version.status.isin(["ok", "empty"])].to_csv(target / "issues.csv", index=False, encoding="utf-8-sig")
    selected_workers = dict(zip(paper.arxiv_id, paper.source_worker))
    diagnostic_columns = ["arxiv_id", "version", "stage", "category", "severity", "affects_status", "requires_review", "message"]
    with (target / "warnings.csv").open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=diagnostic_columns, extrasaction="ignore")
        writer.writeheader()
        for path in sources:
            diagnostic_path = path.parent / "warnings.csv"
            if diagnostic_path.exists():
                with diagnostic_path.open(encoding="utf-8-sig", newline="") as source:
                    for diagnostic in csv.DictReader(source):
                        if selected_workers.get(diagnostic["arxiv_id"]) == path.parent.name:
                            writer.writerow(diagnostic)
    for name, frames, keys in [("runs.csv", runs, ["run_id"]),
                               ("monitor.csv", monitors, ["run_id", "worker", "minute"]),
                               ("disk.csv", disks, ["run_id", "worker", "timestamp"]),
                               ("ranges.csv", definitions, ["range"])]:
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            if name == "ranges.csv" and merged.drop_duplicates().duplicated("range").any():
                raise ValueError("Range definitions differ between workers")
            merged.drop_duplicates(keys).to_csv(target / name, index=False)
    totals = lambda name: int(pd.to_numeric(paper[name], errors="coerce").sum())
    range_rows = []
    for label, group in paper.groupby("range", dropna=False):
        range_rows.append({"range": label, "target_ids": len(group),
                          "papers_success": int(group.paper_success.sum()),
                          "success_rate": float(group.paper_success.mean()),
                          **{name: int(pd.to_numeric(group[name], errors="coerce").sum()) for name in
                             ("versions_expected", "versions_ok", "versions_empty", "versions_partial", "versions_error",
                              "elements_before", "elements_after", "bibtex_before", "bibtex_after")}})
    _export_csv(target / "range_stats.csv", range_rows, ["range", "target_ids", "papers_success", "success_rate"])
    categories = paper.loc[~paper.status.isin(["pending", "missing_input"]), "category"].astype("string").fillna("unknown").value_counts()
    _export_csv(target / "category_counts.csv", [{"category": k, "paper_count": int(v)} for k, v in categories.items()],
                ["category", "paper_count"])
    summary = {"exported_at": utc_now(), "workers": [path.parent.name for path in sources],
               "scope": json.loads(next(iter(configurations))), "target_ids": len(paper),
               "duplicate_rows_removed": duplicate_count, "papers_success": int(paper.paper_success.sum()),
               "success_rate": float(paper.paper_success.mean()) if len(paper) else 0.0,
               "statuses": {str(k): int(v) for k, v in paper.status.value_counts().items()},
               "diagnostics": {"total": totals("warnings_count"), "blocking": totals("blocking_warning_count"),
                               "notices": totals("notice_count"), "papers_for_review": int(paper.review_required.fillna(False).sum())},
               **{name: totals(name) for name in ("versions_expected", "versions_ok", "versions_empty", "versions_partial",
                    "versions_error", "elements_before", "elements_after", "bibtex_before", "bibtex_after", "input_bytes")},
               "note": "For duplicate IDs, the latest export is selected. Only statistics are combined; worker Parquet tables are not merged."}
    for name in ("elements", "bibtex"):
        before, after = summary[f"{name}_before"], summary[f"{name}_after"]
        summary[f"{name}_reduction_ratio"] = (before - after) / before if before else 0.0
    _atomic_json(target / "summary_stats.json", summary)
    (target / "README.md").write_text(summary["note"] + "\nDo not sum disk.csv across runs: each row is a snapshot.\n", encoding="utf-8")
    return {"summary": summary, "reports_dir": str(target)}
