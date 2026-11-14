"""Chạy parser theo bài, lưu checkpoint và xuất số liệu cho notebook parser."""
from __future__ import annotations

from collections import Counter
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
    """Băm nội dung nguồn; không bỏ sót thay đổi chỉ vì kích thước bằng nhau."""
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


class _Monitor:
    """Ghi mẫu RSS xuống đĩa ngay, không giữ chuỗi mẫu trong RAM."""
    def __init__(self, path, run_id, worker, interval):
        self.path, self.run_id, self.worker = Path(path), run_id, worker
        self.interval = max(0.05, interval)
        self.phase = "khoi_tao"
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
    for path in root.rglob("*"):
        try:
            if path.is_file():
                size += path.stat().st_size
        except FileNotFoundError:
            # Worker khác có thể vừa đổi tên tệp tạm khi đo dung lượng chung.
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


def _checkpoint(state, partition, signature):
    try:
        record = json.loads(state.read_text(encoding="utf-8"))
        if record["signature"] != signature or record["stats"]["status"] not in {"ok", "empty"}:
            return None
        for filename in ("elements.parquet", "hierarchy.parquet", "bibtex.parquet", "key_map.json", "bibtex_debug.jsonl", "warnings.jsonl"):
            if _file_signature(partition / filename) != record["file_signatures"][filename]:
                return None
        for name in table_schemas():
            metadata = pq.read_metadata(partition / f"{name}.parquet")
            if metadata.num_rows != record["table_counts"][name]:
                return None
        if not (partition / "key_map.json").exists() or not (partition / "bibtex_debug.jsonl").exists():
            return None
        record["partition"] = str(partition)
        return record
    except (OSError, ValueError, KeyError, pa.ArrowException):
        return None


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
         "review_required", "bibliography_status", "versions_missing_bibliography", "message", "reused", "current_run_sec"])
    versions = [row for record in records for row in record["version_status"]]
    versions.extend({"arxiv_id": job["arxiv_id"], "version": version, "status": "pending",
                     "message": "Chưa xử lý trong phiên này", "missing_count": 0, "warning_count": 0}
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
    categories = Counter(row.get("category") or "khong_ro" for row in rows
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
                         "versions_missing_sources": total("versions_missing_bibliography")},
        "current_run_average_sec": run["duration_sec"] / run["papers_processed"] if run["papers_processed"] else None,
        "measurement_note": "RSS là mẫu tiến trình Python và tiến trình con; đĩa là ảnh chụp thư mục output. Thời gian parse lưu từ checkpoint tách khỏi thời gian phiên hiện tại."}
    _atomic_json(reports / "summary_stats.json", summary)
    (reports / "README.md").write_text(
        "# Số liệu Milestone 2\n\n"
        "Phạm vi của báo cáo là đúng danh sách ID cấu hình, kể cả ID thiếu dữ liệu và bài chưa chạy (pending).\n\n"
        "- per_paper.csv: trạng thái bài, số phiên bản, số element/reference trước và sau khử trùng, dung lượng nguồn, thời gian, cảnh báo.\n"
        "- per_version.csv: trạng thái từng phiên bản; empty khác missing_dir và lỗi parse.\n"
        "- issues.csv: phiên bản còn thiếu dữ liệu/lỗi hoặc chưa xử lý; warnings.csv: chẩn đoán đầy đủ, phân biệt notice với cảnh báo làm thiếu kết quả.\n"
        "- ok có thể có notice/review_required: pipeline đã hoàn tất nhưng metadata/schema cần xem xét. Notice không làm mất quyền dùng checkpoint.\n"
        "- Thiếu .bib/.bbl không làm paper_success=False nếu nội dung chính đã xử lý được. bibliography_status=missing_sources ghi rõ phần bỏ qua; vẫn trích các reference có nguồn.\n"
        "- summary_stats.json: phạm vi, cấu hình, tóm tắt và thông tin máy của phiên hiện tại.\n"
        "- range_stats.csv, ranges.csv: số liệu và định nghĩa các range. category_counts.csv dùng danh mục đầu tiên.\n"
        "- runs.csv: lịch sử phiên của worker; monitor.csv: RSS trung bình/tối đa mỗi phút; disk.csv: ảnh chụp output và thời gian quét.\n"
        "- Ba bảng Parquet tổng nằm trong tables/<worker>/; phân mảnh theo bài nằm trong papers/. Không cộng dung lượng bảng tổng và phân mảnh như dữ liệu khoa học khác nhau.\n"
        "- partial không tính thành công đầy đủ; bảng có thể chứa phần dữ liệu parse được, cần đọc per_version.csv trước khi sử dụng.\n"
        "- reused=True: dùng checkpoint hợp lệ, elapsed_sec là lần parse đã lưu, current_run_sec là chi phí phiên hiện tại.\n"
        "- Log runs/timings/events và mẫu monitor/disk gốc nằm trong run_logs/<worker>/.\n",
        encoding="utf-8")
    return summary


def run_parser(jobs, output_dir, *, worker="local", ranges=None, enrich=False, fuzzy_threshold=0.92,
               resume=True, max_run_hours=None, monitor_interval=1.0, disk_every=100, progress=None,
               num_workers=1, max_workers=None, adaptive_ram=True, reserve_ram_gb=None,
               ram_per_worker_gb=1.5, ram_wait_timeout_sec=60):
    """Trả thống kê và đường dẫn; không giữ các bảng element toàn corpus trong RAM."""
    if not re.fullmatch(r"[A-Za-z0-9_+-]+", worker):
        raise ValueError("Tên worker chỉ gồm chữ, số, _, + và -")
    if not 0 <= fuzzy_threshold <= 1 or disk_every < 1 or monitor_interval <= 0:
        raise ValueError("Cấu hình ngưỡng hoặc chu kỳ đo không hợp lệ")
    if max_run_hours is not None and max_run_hours <= 0:
        raise ValueError("Giới hạn thời gian phải dương")
    if num_workers != "auto" and (type(num_workers) is not int or num_workers < 1):
        raise ValueError("num_workers phải là 'auto' hoặc số nguyên dương")
    if max_workers is not None and (type(max_workers) is not int or max_workers < 1):
        raise ValueError("max_workers phải là số nguyên dương hoặc None")
    if type(adaptive_ram) is not bool:
        raise ValueError("adaptive_ram phải là boolean")
    for name, value, allow_zero in (("reserve_ram_gb", reserve_ram_gb, True),
                                    ("ram_per_worker_gb", ram_per_worker_gb, False),
                                    ("ram_wait_timeout_sec", ram_wait_timeout_sec, False)):
        if value is None and allow_zero:
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"{name} không hợp lệ")
    jobs = list(jobs)
    if len({job["arxiv_id"] for job in jobs}) != len(jobs):
        raise ValueError("Danh sách bài có ID trùng")
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
              "ram_per_worker_gb": ram_per_worker_gb, "ram_wait_timeout_sec": ram_wait_timeout_sec}
    run = {"run_id": run_id, "worker": worker, "started_at": utc_now(), "event": "start", "config": config,
           "target_ids": len(jobs), "remaining_at_start": None, "papers_processed": 0, "papers_reused": 0,
           "machine": {"python": platform.python_version(), "platform": platform.platform(),
                       "cpu_count": os.cpu_count(), "ram_total_bytes": psutil.virtual_memory().total}}
    _append_jsonl(logs / "runs.jsonl", run)
    records = []
    completed_ids = set()
    started = time.perf_counter()
    stop_reason = "completed"
    schemas = table_schemas()
    monitor = _Monitor(logs / f"monitor_{run_id}.csv", run_id, worker, monitor_interval)

    def parallel_event(event):
        _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id, "worker": worker, **event})
        logger.info("Điều phối song song: %s", json.dumps(event, ensure_ascii=False))
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
                      "file_signatures": {filename: _file_signature(partition / filename) for filename in
                          ("elements.parquet", "hierarchy.parquet", "bibtex.parquet", "key_map.json", "bibtex_debug.jsonl", "warnings.jsonl")}}
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
        logger.info("%s: %s; tái sử dụng=%s; %s", job["arxiv_id"], record["stats"]["status"], reused,
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
            monitor.phase = "kiem_tra_checkpoint"
            for job in jobs:
                partition = tables / "papers" / _paper_name(job["arxiv_id"])
                stamp = state / f"{_paper_name(job['arxiv_id'])}.json"
                try:
                    signature = input_signature(job, options)
                except OSError as exc:
                    # Vẫn để parser ghi lỗi riêng của bài; không dùng checkpoint.
                    signature = uuid.uuid4().hex
                    logger.warning("Không băm được nguồn %s: %s", job["arxiv_id"], exc)
                record = _checkpoint(stamp, partition, signature) if resume else None
                prepared.append((job, partition, stamp, signature, record))
            run["remaining_at_start"] = sum(record is None for _, _, _, _, record in prepared)
            _append_jsonl(logs / "events.jsonl", {"timestamp": utc_now(), "run_id": run_id,
                          "status": "checkpoint_checked", "remaining_at_start": run["remaining_at_start"]})
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
                monitor.phase = "parse_song_song"
                with closing(iter_parallel_results([item[0] for item in pending_items],
                        num_workers=num_workers, max_workers=max_workers, adaptive_ram=adaptive_ram,
                        reserve_ram_gb=reserve_ram_gb, ram_per_worker_gb=ram_per_worker_gb,
                        ram_wait_timeout_sec=ram_wait_timeout_sec, enrich=enrich,
                        fuzzy_threshold=fuzzy_threshold, deadline=deadline, on_event=parallel_event)) as results:
                    for index, result in results:
                        save_result(pending_items[index], result)
                        del result
                if len(records) < len(jobs) and deadline is not None and time.perf_counter() >= deadline:
                    stop_reason = "time_limit"
            monitor.phase = "xuat_bang"
            combine_completed()
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        combine_completed()
    except Exception as exc:
        stop_reason = "error"
        logger.exception("Phiên dừng do lỗi")
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
        raise FileNotFoundError("Chưa có báo cáo worker để ghép")
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
        raise ValueError("Các worker dùng mã hoặc cấu hình parse khác nhau; chạy lại cùng cấu hình trước khi ghép")
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
                raise ValueError("Định nghĩa range giữa các worker không thống nhất")
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
    categories = paper.loc[~paper.status.isin(["pending", "missing_input"]), "category"].fillna("khong_ro").value_counts()
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
               "note": "ID trùng lấy bản xuất mới nhất. Chỉ ghép thống kê; không ghép bảng Parquet của các worker."}
    for name in ("elements", "bibtex"):
        before, after = summary[f"{name}_before"], summary[f"{name}_after"]
        summary[f"{name}_reduction_ratio"] = (before - after) / before if before else 0.0
    _atomic_json(target / "summary_stats.json", summary)
    (target / "README.md").write_text(summary["note"] + "\nKhông cộng trực tiếp disk.csv giữa các phiên: mỗi dòng là một ảnh chụp.\n", encoding="utf-8")
    return {"summary": summary, "reports_dir": str(target)}
