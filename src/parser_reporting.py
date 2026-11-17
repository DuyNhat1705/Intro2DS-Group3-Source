"""Chạy parser theo bài, lưu checkpoint và xuất số liệu cho notebook parser."""
from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
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
from .utils import list_files


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def table_schemas():
    """Schema tường minh, kể cả khi bảng không có dòng nào."""
    text = pa.string()
    kind = pa.dictionary(pa.int32(), text)
    return {
        "elements": pa.schema([pa.field("element_id", text, False), pa.field("arxiv_id", text, False),
                               pa.field("kind", kind), pa.field("content", text)]),
        "hierarchy": pa.schema([pa.field("arxiv_id", text, False), pa.field("version", pa.int32()),
                                pa.field("element_id", text, False), pa.field("parent_id", text, False)]),
        "bibtex": pa.schema([pa.field("arxiv_id", text, False), pa.field("citation_key", text, False),
                             pa.field("entry_type", kind), pa.field("title", text),
                             pa.field("authors", pa.list_(text)), pa.field("year", pa.int32()),
                             pa.field("doi", text), pa.field("journal", text), pa.field("url", text)]),
    }


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


# Số kết quả tối đa đang chờ luồng ghi nền, để RAM không phình khi ghi chậm.
_MAX_PENDING_WRITES = 2
# Luồng ghi nền và luồng điều phối cùng ghi events.jsonl; khóa giữ từng dòng nguyên vẹn.
_APPEND_LOCK = threading.Lock()


def _append_jsonl(path, value):
    line = json.dumps(value, ensure_ascii=False) + "\n"
    with _APPEND_LOCK, Path(path).open("a", encoding="utf-8") as stream:
        stream.write(line)


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


def _paper_name(arxiv_id):
    return re.sub(r"[^A-Za-z0-9_-]", "-", arxiv_id) + "-" + hashlib.sha256(arxiv_id.encode()).hexdigest()[:8]


def _file_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _code_signature():
    digest = hashlib.sha256()
    for name in ("parser_pipeline.py", "parser_reporting.py", "parser_parallel.py", "gather_clean_tex.py", "hierarchy.py",
                 "bibtex.py", "m2_types.py", "utils.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    for name in ("bibtexparser", "wtpsplit", "onnxruntime", "pyarrow"):
        try:
            digest.update(f"{name}={importlib.metadata.version(name)}".encode())
        except importlib.metadata.PackageNotFoundError:
            digest.update(f"{name}=missing".encode())
    return digest.hexdigest()


def input_signature(job, options):
    """Băm nội dung nguồn; không bỏ sót thay đổi chỉ vì kích thước bằng nhau.

    Thư mục phiên bản được ghi theo tên tương đối với thư mục bài, nên cùng dữ
    liệu đọc từ Drive hay từ bản giải nén trên ổ local cho cùng chữ ký.
    """
    paper_dir = Path(job["paper_dir"]) if job.get("paper_dir") is not None else None

    def location(path):
        if path is None:
            return None
        path = Path(path)
        return path.name if paper_dir is not None and path.parent == paper_dir else str(path)

    digest = hashlib.sha256(json.dumps({"arxiv_id": job["arxiv_id"], "metadata": job.get("metadata", {}),
        "versions": {str(v): location(p) for v, p in job["version_dirs"].items()},
        "source_status": job.get("source_status_by_version", {}),
        "errors": job.get("discovery_errors", []), "options": options}, sort_keys=True).encode())
    directory = job.get("paper_dir")
    if directory is not None and Path(directory).is_dir():
        directory = Path(directory)
        for path in (p for p in list_files(directory) if
                     p.suffix.lower() in {".tex", ".bib"} or p.name == "metadata.json"):
            digest.update(path.relative_to(directory).as_posix().encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
    return digest.hexdigest()


class _Monitor:
    """Ghi mẫu RSS xuống đĩa ngay, không giữ chuỗi mẫu trong RAM."""
    def __init__(self, path, run_id, worker, interval):
        self.path, self.run_id, self.worker = Path(path), run_id, worker
        self.interval = max(0.05, interval)
        self.phase = "initializing"
        self.stop = threading.Event()
        self.peak, self.total, self.count = 0, 0, 0
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self, writer, stream):
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
        writer.writerow({"timestamp": utc_now(), "run_id": self.run_id, "worker": self.worker,
                         "phase": self.phase, "rss_bytes": rss})
        stream.flush()

    def _loop(self):
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


def _disk_snapshot(root, logs, run_id, worker):
    started = time.perf_counter()
    size = 0
    stack = [root]
    while stack:
        # scandir cho biết loại mục từ lúc liệt kê, mỗi tệp chỉ cần một lần stat.
        try:
            with os.scandir(stack.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file():
                            size += entry.stat().st_size
                    except FileNotFoundError:
                        # Worker khác có thể vừa đổi tên tệp tạm khi đo dung lượng chung.
                        continue
        except FileNotFoundError:
            continue
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


def _write_tables(partition, result, schemas):
    partition.mkdir(parents=True, exist_ok=True)
    for name, schema in schemas.items():
        temporary = partition / f"{name}.parquet.tmp"
        pq.write_table(pa.Table.from_pylist(result[name], schema=schema), temporary, compression="zstd")
        os.replace(temporary, partition / f"{name}.parquet")
    _atomic_json(partition / "key_map.json", result.get("key_map", {}))
    for name, rows in (("bibtex_debug", result.get("bibtex_debug", [])), ("warnings", result.get("warnings", []))):
        temporary = partition / f"{name}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, partition / f"{name}.jsonl")


# Tệp của một partition; checkpoint chỉ hợp lệ khi cả sáu tệp còn nguyên.
_OUTPUT_FILES = ("elements.parquet", "hierarchy.parquet", "bibtex.parquet", "key_map.json", "bibtex_debug.jsonl",
                 "warnings.jsonl")
# Bản chụp mọi bản ghi checkpoint của worker, để phiên sau đọc một tệp thay vì một tệp mỗi bài.
_SNAPSHOT = "_manifest.json"
# Trường chỉ có nghĩa trong phiên đang chạy, không lưu vào bản chụp.
_RUN_ONLY_FIELDS = ("partition", "range", "reused", "current_run_sec")


def _checkpoint(state, partition, signature, *, verify="size", record=None):
    """Bản ghi checkpoint dùng lại được, hoặc None.

    ``verify="size"`` so kích thước sáu tệp output với lúc ghi (một lần stat
    mỗi tệp): phát hiện tệp thiếu hoặc ghi dở, nhanh trên Google Drive.
    ``verify="full"`` băm lại SHA-256 và đếm dòng Parquet: phát hiện cả tệp bị
    sửa mà không đổi kích thước, nhưng phải đọc hết từng tệp. Bản ghi cũ chưa
    có kích thước luôn được kiểm tra đầy đủ, rồi được bổ sung kích thước.
    ``record`` là bản ghi đã đọc sẵn (từ bản chụp); None thì đọc ``state``.
    """
    try:
        record = json.loads(state.read_text(encoding="utf-8")) if record is None else dict(record)
        if record["signature"] != signature or record["stats"]["status"] not in {"ok", "empty"}:
            return None
        sizes = record.get("file_sizes")
        if verify == "size" and sizes:
            for filename in _OUTPUT_FILES:
                if (partition / filename).stat().st_size != sizes[filename]:
                    return None
        else:
            for filename in _OUTPUT_FILES:
                if _file_signature(partition / filename) != record["file_signatures"][filename]:
                    return None
            for name in table_schemas():
                metadata = pq.read_metadata(partition / f"{name}.parquet")
                if metadata.num_rows != record["table_counts"][name]:
                    return None
            if not sizes:
                record["file_sizes"] = {filename: (partition / filename).stat().st_size for filename in _OUTPUT_FILES}
        record["partition"] = str(partition)
        return record
    except (OSError, ValueError, KeyError, TypeError, pa.ArrowException):
        return None


def _load_snapshot(path):
    """Bản ghi theo tên partition; tệp thiếu/hỏng thì rỗng (quay về đọc từng tệp state)."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data["records"] if data.get("format") == 1 and isinstance(data.get("records"), dict) else {}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return {}


def _write_snapshot(path, records):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps({"format": 1, "records": records}, ensure_ascii=False, separators=(",", ":")),
                         encoding="utf-8")
    os.replace(temporary, path)


def _combine_tables(records, tables, schemas):
    """Ghép từng batch, không nạp toàn bộ corpus vào RAM."""
    for name, schema in schemas.items():
        temporary = tables / f"{name}.parquet.tmp"
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for record in records:
                with pq.ParquetFile(Path(record["partition"]) / f"{name}.parquet") as source:
                    for batch in source.iter_batches(batch_size=8192):
                        writer.write_table(pa.Table.from_batches([batch]).cast(schema))
        os.replace(temporary, tables / f"{name}.parquet")


def _export_csv(path, rows, columns):
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame:
            frame[column] = pd.Series(dtype="object")
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame


def _exports(reports, logs, records, pending, ranges, run, config, tables):
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
         "review_required", "bibliography_status", "versions_missing_bibliography", "cite_keys", "cite_keys_unresolved",
         "message", "reused", "current_run_sec"])
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
        "table_rows": {name: pq.read_metadata(tables / f"{name}.parquet").num_rows for name in table_schemas()},
        "recorded_parse_sec": float(pd.to_numeric(paper.elapsed_sec, errors="coerce").sum()),
        "current_run_processing_sec": float(paper.current_run_sec.sum()),
        "diagnostics": {"total": total("warnings_count"), "blocking": total("blocking_warning_count"),
                        "notices": total("notice_count"), "papers_for_review": int(paper.review_required.fillna(False).sum())},
        "bibliography": {"papers_missing_sources": int(paper.bibliography_status.eq("missing_sources").sum()),
                         "versions_missing_sources": total("versions_missing_bibliography"),
                         "cite_keys": total("cite_keys"), "cite_keys_unresolved": total("cite_keys_unresolved"),
                         "papers_with_unresolved_cites": int(pd.to_numeric(paper["cite_keys_unresolved"],
                                                                           errors="coerce").fillna(0).gt(0).sum())},
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
        "- partial does not count as full success; tables may contain partially parsed data. Check per_version.csv before use.\n"
        "- reused=True: a valid checkpoint was reused; elapsed_sec is the saved parsing time, and current_run_sec is the time spent in the current run.\n"
        "- Raw runs/timings/events logs and monitor/disk samples are in run_logs/<worker>/.\n",
        encoding="utf-8")
    return summary


def run_parser(jobs, output_dir, *, worker="local", ranges=None, enrich=False, fuzzy_threshold=0.92,
               resume=True, max_run_hours=None, monitor_interval=1.0, disk_every=100, progress=None,
               num_workers=1, max_workers=None, adaptive_ram=True, reserve_ram_gb=None,
               ram_per_worker_gb=1.5, ram_wait_timeout_sec=60, paper_timeout_sec=None,
               verify_checkpoints="size", data_preparation=None):
    """Trả thống kê và đường dẫn; không giữ các bảng element toàn corpus trong RAM.

    Chế độ song song ghi kết quả trong một luồng nền để tiến trình chính tiếp
    tục giao bài cho worker trong lúc ghi Parquet/checkpoint (chậm trên Drive).
    paper_timeout_sec chỉ áp dụng khi chạy song song: bài quá hạn bị dừng và
    ghi lỗi để thử lại ở phiên sau. Chế độ tuần tự chạy trong chính tiến trình
    nên không thể ngắt giữa bài.

    verify_checkpoints: "size" (so kích thước tệp output) hoặc "full" (băm lại
    và đếm dòng), xem ``_checkpoint``. Bản chụp ``.state/<worker>/_manifest.json``
    bị xóa khi phiên bắt đầu và ghi lại khi phiên kết thúc, nên chỉ tồn tại khi
    phiên trước kết thúc bình thường; phiên bị ngắt đột ngột thì phiên sau đọc
    từng tệp state như cũ. data_preparation (dict, ví dụ thời gian giải nén dữ
    liệu) được ghi nguyên vào thông tin phiên để báo cáo thời gian end-to-end.
    """
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", worker):
        raise ValueError("Worker name may contain only letters, digits, _, +, and -")
    if not 0 <= fuzzy_threshold <= 1 or disk_every < 1 or monitor_interval <= 0:
        raise ValueError("Invalid threshold or measurement interval configuration")
    if max_run_hours is not None and max_run_hours <= 0:
        raise ValueError("Time limit must be positive")
    if num_workers != "auto" and (type(num_workers) is not int or num_workers < 1):
        raise ValueError("num_workers must be 'auto' or a positive integer")
    if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
        raise ValueError("max_workers must be a positive integer or None")
    if paper_timeout_sec is not None and (isinstance(paper_timeout_sec, bool) or
            not isinstance(paper_timeout_sec, (int, float)) or not math.isfinite(paper_timeout_sec)
            or paper_timeout_sec <= 0):
        raise ValueError("paper_timeout_sec must be a positive number or None")
    if type(adaptive_ram) is not bool:
        raise ValueError("adaptive_ram must be a boolean")
    if verify_checkpoints not in {"size", "full"}:
        raise ValueError("verify_checkpoints must be 'size' or 'full'")
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
    logger.setLevel(logging.INFO)
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
              "paper_timeout_sec": paper_timeout_sec, "verify_checkpoints": verify_checkpoints}
    run = {"run_id": run_id, "worker": worker, "started_at": utc_now(), "event": "start", "config": config,
           "target_ids": len(jobs), "remaining_at_start": None, "papers_processed": 0, "papers_reused": 0,
           "machine": {"python": platform.python_version(), "platform": platform.platform(),
                       "cpu_count": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total}}
    if data_preparation is not None:
        run["data_preparation"] = dict(data_preparation)
    _append_jsonl(logs / "runs.jsonl", run)
    snapshot_path = state / _SNAPSHOT
    snapshot = _load_snapshot(snapshot_path) if resume else {}
    # Xóa ngay: nếu phiên này bị ngắt đột ngột, phiên sau không dùng một bản chụp cũ hơn các tệp state.
    snapshot_path.unlink(missing_ok=True)
    records = []
    completed_ids = set()
    started = time.perf_counter()
    stop_reason = "completed"
    schemas = table_schemas()
    monitor = _Monitor(logs / f"monitor_{run_id}.csv", run_id, worker, monitor_interval)

    def parallel_event(event):
        _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id, "worker": worker, **event})
        logger.info("Parallel scheduling: %s", json.dumps(event, ensure_ascii=False))
        run["parallel"] = {**run.get("parallel", {}), **event}

    def save_result(prepared_item, result=None, paper_started=None):
        """Chỉ tiến trình chính ghi bảng/checkpoint và gọi callback tiến độ."""
        job, partition, stamp, signature, record = prepared_item
        reused = record is not None
        write_started = time.perf_counter()
        if not reused:
            _write_tables(partition, result, schemas)
            for stage in ("input", "gather", "cleanup", "construct_hierarchy", "extract_bibtex",
                          "unify_publication", "rewrite_citations", "build_element_tables"):
                timings = [row for row in result.get("timings", []) if row["stage"] == stage]
                result["stats"][f"{stage}_sec"] = sum(row["duration_sec"] for row in timings)
                statuses = {row["status"] for row in timings}
                result["stats"][f"{stage}_status"] = next((value for value in
                    ("error", "missing_dir", "not_found", "partial", "ok", "empty") if value in statuses), "not_run")
            result["stats"]["output_table_bytes"] = sum((partition / f"{name}.parquet").stat().st_size for name in schemas)
            record = {"signature": signature, "stats": result["stats"],
                      "version_status": result["version_status"], "partition": str(partition),
                      "table_counts": {name: len(result[name]) for name in schemas},
                      "file_signatures": {filename: _file_signature(partition / filename) for filename in _OUTPUT_FILES},
                      "file_sizes": {filename: (partition / filename).stat().st_size for filename in _OUTPUT_FILES}}
            _atomic_json(stamp, record)
            for timing in result.get("timings", []):
                _append_jsonl(logs / "timings.jsonl", {"run_id": run_id, "worker": worker,
                    "arxiv_id": job["arxiv_id"], **timing})
            run["papers_processed"] += 1
        else:
            run["papers_reused"] += 1
        duration = (time.perf_counter() - paper_started if paper_started is not None else
                    time.perf_counter() - write_started + (0 if reused else record["stats"]["elapsed_sec"]))
        record.update(range=job.get("range", "local"), reused=reused, current_run_sec=duration)
        records.append(record)
        completed_ids.add(job["arxiv_id"])
        logger.info("%s: %s; reused=%s; %s", job["arxiv_id"], record["stats"]["status"], reused,
                    record["stats"].get("message", ""))
        _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
            "arxiv_id": job["arxiv_id"], "status": record["stats"]["status"], "reused": reused,
            "message": record["stats"].get("message", "")})
        if progress:
            progress(len(records), len(jobs), record["stats"])
        if len(records) % disk_every == 0:
            _disk_snapshot(root, logs, run_id, worker)

    def combine_completed():
        order = {job["arxiv_id"]: index for index, job in enumerate(jobs)}
        records.sort(key=lambda record: order[record["stats"]["arxiv_id"]])
        _combine_tables(records, tables, schemas)
    try:
        with monitor:
            _disk_snapshot(root, logs, run_id, worker)
            # Kiểm tra cache trước để số việc còn lại lúc bắt đầu có ý nghĩa.
            prepared = []
            monitor.phase = "checking_checkpoints"
            check_started = time.perf_counter()
            sources = Counter()
            for job in jobs:
                name = _paper_name(job["arxiv_id"])
                partition = tables / "papers" / name
                stamp = state / f"{name}.json"
                try:
                    signature = input_signature(job, options)
                except OSError as exc:
                    # Vẫn để parser ghi lỗi riêng của bài; không dùng checkpoint.
                    signature = uuid.uuid4().hex
                    logger.warning("Could not hash sources for %s: %s", job["arxiv_id"], exc)
                record = None
                if resume:
                    # Bản chụp chỉ còn khi phiên trước kết thúc bình thường, nên trùng với tệp
                    # state; bài có trong bản chụp không cần mở tệp state nữa.
                    source = "snapshot" if name in snapshot else "state_file"
                    record = _checkpoint(stamp, partition, signature, verify=verify_checkpoints,
                                         record=snapshot.get(name))
                    sources[source if record is not None else "not_reusable"] += 1
                prepared.append((job, partition, stamp, signature, record))
            run["remaining_at_start"] = sum(record is None for _, _, _, _, record in prepared)
            run["checkpoint"] = {"sec": round(time.perf_counter() - check_started, 3), "verify": verify_checkpoints,
                                 "reused_from_snapshot": sources["snapshot"],
                                 "reused_from_state_files": sources["state_file"],
                                 "not_reusable": sources["not_reusable"]}
            _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
                          "status": "checkpoint_checked", "remaining_at_start": run["remaining_at_start"],
                          **{f"checkpoint_{key}": value for key, value in run["checkpoint"].items()}})
            if num_workers == 1:
                for item in prepared:
                    if max_run_hours is not None and time.perf_counter() - started >= max_run_hours * 3600:
                        stop_reason = "time_limit"
                        break
                    paper_started = time.perf_counter()
                    job, _, _, _, record = item
                    monitor.phase = job["arxiv_id"]
                    result = parse_publication(job, enrich=enrich, fuzzy_threshold=fuzzy_threshold) if record is None else None
                    save_result(item, result, paper_started)
                    del result
            else:
                from .parser_parallel import iter_parallel_results
                pending_items = []
                for item in prepared:
                    if item[-1] is not None:
                        save_result(item)
                    else:
                        pending_items.append(item)
                deadline = started + max_run_hours * 3600 if max_run_hours is not None else None
                monitor.phase = "parallel_parsing"
                writes = deque()
                # Thoát with theo thứ tự ngược: dừng worker trước, rồi chờ luồng
                # ghi xong mọi kết quả đã nhận, kể cả khi Interrupt hoặc lỗi.
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="parser-writer") as writer, \
                        closing(iter_parallel_results([item[0] for item in pending_items],
                        num_workers=num_workers, max_workers=max_workers, adaptive_ram=adaptive_ram,
                        reserve_ram_gb=reserve_ram_gb, ram_per_worker_gb=ram_per_worker_gb,
                        ram_wait_timeout_sec=ram_wait_timeout_sec, enrich=enrich,
                        fuzzy_threshold=fuzzy_threshold, deadline=deadline, on_event=parallel_event,
                        job_timeout_sec=paper_timeout_sec)) as results:
                    for index, result in results:
                        writes.append(writer.submit(save_result, pending_items[index], result))
                        del result
                        # Báo lỗi ghi sớm; giới hạn số kết quả chờ ghi để RAM không phình.
                        while writes and (writes[0].done() or len(writes) > _MAX_PENDING_WRITES):
                            writes.popleft().result()
                while writes:
                    writes.popleft().result()
                if len(records) < len(jobs) and deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_limit"
            monitor.phase = "exporting_tables"
            combine_completed()
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
        # Bản ghi của phiên này đè lên bản chụp cũ; bài không chạy trong phiên giữ bản ghi cũ.
        merged = dict(snapshot)
        for record in records:
            merged[Path(record["partition"]).name] = {key: value for key, value in record.items()
                                                      if key not in _RUN_ONLY_FIELDS}
        try:
            _write_snapshot(snapshot_path, merged)
        except OSError as exc:
            logger.warning("Could not write checkpoint snapshot: %s", exc)
        run.update(event="end", ended_at=utc_now(), duration_sec=time.perf_counter() - started,
                   stop_reason=stop_reason, peak_rss_bytes=monitor.peak,
                   mean_rss_bytes=monitor.total / monitor.count if monitor.count else None,
                   remaining=len(jobs) - len(records))
        _append_jsonl(logs / "runs.jsonl", run)
        handler.close()
        logger.removeHandler(handler)
    summary = _exports(reports, logs, records, [job for job in jobs if job["arxiv_id"] not in completed_ids],
                       ranges or {}, run, config, tables)
    snapshot = _disk_snapshot(root, logs, run_id, worker)
    summary["output_snapshot"] = snapshot
    _atomic_json(reports / "summary_stats.json", summary)
    # Đồng bộ ảnh chụp cuối vào CSV báo cáo.
    pd.concat([pd.read_csv(p) for p in sorted(logs.glob("disk_*.csv"))], ignore_index=True).to_csv(reports / "disk.csv", index=False)
    return {"summary": summary, "reports_dir": str(reports), "tables_dir": str(tables), "logs_dir": str(logs),
            "tables": {name: str(tables / f"{name}.parquet") for name in schemas}}


def combine_worker_reports(output_dir):
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
    categories = paper.loc[~paper.status.isin(["pending", "missing_input"]), "category"].fillna("unknown").value_counts()
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
