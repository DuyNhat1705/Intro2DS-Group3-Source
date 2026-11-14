"""Điều phối spawn theo RAM; mỗi tiến trình giữ cache SaT của parser gốc.

Kết quả đến theo thứ tự hoàn thành, chỉ số luôn thuộc đầu vào. Không ghi tệp,
không sửa thuật toán parser. Iterable chỉ được đọc trước tối đa một job.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import multiprocessing as mp
from multiprocessing.connection import wait
from numbers import Real
import os
from pathlib import Path
import signal
import sys
import time

import psutil

__all__ = ["iter_parallel_results"]
_GB = 1024 ** 3
_POLL_SEC = 0.1
_HEADROOM = 1.2


def _read_number(path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _cgroup_paths():
    """Đọc nhóm hiện tại và các nhóm cha, hỗ trợ cgroup v1/v2 phổ biến."""
    root = Path("/sys/fs/cgroup")
    paths = {root, root / "memory", root / "cpu", root / "cpu,cpuacct"}
    try:
        lines = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return paths
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, relative = parts
        bases = [root] if not controllers else [
            root / name for name in (controllers, *controllers.split(","))]
        for base in bases:
            path = base / relative.lstrip("/")
            # Một số namespace trả đường dẫn có '..'; không đi ra ngoài mount.
            if ".." in path.parts:
                continue
            while True:
                paths.add(path)
                if path == base:
                    break
                path = path.parent
    return paths


def _cgroup_limits():
    limits, available, cpus = [], [], []
    if not sys.platform.startswith("linux"):
        return None, None, None
    for path in _cgroup_paths():
        for limit_name, used_name in (("memory.max", "memory.current"),
                                      ("memory.limit_in_bytes", "memory.usage_in_bytes")):
            limit = _read_number(path / limit_name)
            used = _read_number(path / used_name)
            if limit is not None and 0 < limit < 2 ** 60:
                limits.append(limit)
                if used is not None:
                    available.append(max(0, limit - used))
        try:
            quota, period = (path / "cpu.max").read_text().split()
            quota, period = int(quota), int(period)
        except (OSError, ValueError):
            quota = _read_number(path / "cpu.cfs_quota_us")
            period = _read_number(path / "cpu.cfs_period_us")
        if quota is not None and period and quota > 0 and period > 0:
            cpus.append(max(1, quota // period))
    return (min(limits) if limits else None, min(available) if available else None,
            min(cpus) if cpus else None)


def _resource_snapshot(pids=()):
    """Ảnh tài nguyên độc lập để test có thể thay bằng dữ liệu cố định."""
    memory = psutil.virtual_memory()
    total, available = memory.total, memory.available
    cpu = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    try:
        cpu = min(cpu, len(psutil.Process().cpu_affinity()) or 1)
    except (AttributeError, psutil.Error, OSError):
        pass
    cg_total, cg_available, cg_cpu = _cgroup_limits()
    if cg_total is not None:
        total = min(total, cg_total)
    if cg_available is not None:
        available = min(available, cg_available)
    if cg_cpu is not None:
        cpu = min(cpu, cg_cpu)
    rss = {}
    for pid in pids:
        try:
            rss[pid] = psutil.Process(pid).memory_info().rss
        except (psutil.Error, OSError):
            rss[pid] = 0
    return dict(cpu_count=max(1, cpu), total_bytes=total,
                available_bytes=max(0, min(total, available)), rss_by_pid=rss)


def _worker_main(connection, enrich, fuzzy_threshold):
    """Đích spawn phải import được; chỉ cha nhận Ctrl-C."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        while True:
            job = connection.recv()
            if job is None:
                break
            try:
                # Nhập trễ để đặt giới hạn BLAS trước khi parser/model được nhập.
                if __package__:
                    from .parser_pipeline import parse_publication
                else:
                    from parser_pipeline import parse_publication
                result = parse_publication(job, enrich=enrich, fuzzy_threshold=fuzzy_threshold)
                kind = "result"
            except Exception as exc:
                kind, result = "error", f"{type(exc).__name__}: {exc}"
            connection.send((kind, result, psutil.Process().memory_info().rss))
            del result, job
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def _error_result(job, message, elapsed):
    """Giữ đủ hợp đồng bảng/thống kê khi tiến trình hoặc IPC thất bại."""
    identity = job.get("arxiv_id", "")
    sources = job.get("source_status_by_version", {})
    numbers = sorted(set(job.get("version_dirs", {})) | set(sources)) or [1]
    metadata = job.get("metadata", {})
    categories = metadata.get("categories", [])
    category = categories[0] if isinstance(categories, list) and categories else metadata.get("category", "")
    ended = datetime.now(timezone.utc)
    began = ended - timedelta(seconds=elapsed)
    return dict(elements=[], hierarchy=[], bibtex=[], bibtex_debug=[], key_map={}, warnings=[],
                version_status=[dict(arxiv_id=identity, version=v, status="error", message=message,
                    missing_count=0, warning_count=0, source_status=sources.get(v, ""),
                    missing=[], errors=[message]) for v in numbers],
                timings=[dict(arxiv_id=identity, stage="parallel_worker", started_at=began.isoformat(),
                              ended_at=ended.isoformat(), duration_sec=elapsed, status="error", message=message)],
                stats=dict(arxiv_id=identity, status="error", paper_success=False,
                    versions_expected=len(numbers), versions_ok=0, versions_empty=0,
                    versions_partial=0, versions_error=len(numbers), versions_not_found=0,
                    versions_missing_dir=0, elements_before=0, elements_after=0,
                    bibtex_before=0, bibtex_after=0, elapsed_sec=elapsed, input_bytes=0,
                    category=category, warnings_count=0, message=message))


@dataclass
class _Worker:
    process: object
    connection: object
    index: int | None = None
    job: object = None
    started: float = 0
    rss: int = 0


def _dispose(workers, *, gentle=False):
    """Đóng cả nhóm với thời gian chờ chung, không để con mồ côi."""
    started_workers = []
    for worker in workers:
        if worker.process.pid is None:
            worker.connection.close()
            worker.process.close()
            continue
        started_workers.append(worker)
        if gentle:
            try:
                worker.connection.send(None)
            except (OSError, EOFError):
                pass
        elif worker.process.is_alive():
            worker.process.terminate()
        worker.connection.close()
    until = time.perf_counter() + 0.5
    for worker in started_workers:
        worker.process.join(max(0, until - time.perf_counter()))
    for worker in started_workers:
        if worker.process.is_alive():
            worker.process.kill()
    for worker in started_workers:
        worker.process.join()
        worker.process.close()


def _positive_int(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} phải là số nguyên dương")
    return value


def _number(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} phải là số hữu hạn >= {minimum}")
    return float(value)


def iter_parallel_results(jobs, *, num_workers="auto", max_workers=None,
                          reserve_ram_gb=None, ram_per_worker_gb=1.5, adaptive_ram=True,
                          enrich=False, fuzzy_threshold=.92, deadline=None, on_event=None,
                          ram_wait_timeout_sec=60):
    """Yield ``(index, parse_publication(job))`` với tối đa một job mỗi worker.

    GB dùng GiB (2**30 byte); mặc định chừa max(0.5 GiB, 10% tổng RAM
    theo giới hạn hệ thống). Ước lượng worker là max(sàn cấu hình, RSS đỉnh
    * 1.2). adaptive_ram=False bỏ giới hạn RAM, vẫn hiệu chuẩn một job đầu.
    deadline là perf_counter tuyệt đối; close()/Ctrl-C dừng ngay các con.

    on_event nhận dict JSON với event, elapsed_sec, workers, active, submitted,
    completed. parallel_summary có pending (None nếu iterable không có len),
    peak_workers, peak_in_flight, worker_errors, ram_wait_sec, stopped_reason.
    Summary cũng giữ memory_estimate_gb, reserve_ram_gb, cpu_cap, worker_cap.
    completed bao gồm lỗi worker; pending chỉ đếm job chưa giao.
    """
    if num_workers != "auto":
        _positive_int("num_workers", num_workers)
    if max_workers is not None:
        _positive_int("max_workers", max_workers)
    floor = _number("ram_per_worker_gb", ram_per_worker_gb)
    if floor == 0:
        raise ValueError("ram_per_worker_gb phải > 0")
    reserve = None if reserve_ram_gb is None else _number("reserve_ram_gb", reserve_ram_gb)
    timeout = _number("ram_wait_timeout_sec", ram_wait_timeout_sec)
    threshold = _number("fuzzy_threshold", fuzzy_threshold)
    if threshold > 1:
        raise ValueError("fuzzy_threshold phải <= 1")
    if not isinstance(adaptive_ram, bool) or not isinstance(enrich, bool):
        raise ValueError("adaptive_ram và enrich phải là bool")
    if on_event is not None and not callable(on_event):
        raise ValueError("on_event phải gọi được")
    if deadline is not None:
        deadline = _number("deadline", deadline, minimum=-float("inf"))

    snapshot = _resource_snapshot()
    cpu_cap = snapshot["cpu_count"]
    cap = min(cpu_cap, cpu_cap if num_workers == "auto" else num_workers,
              max_workers if max_workers is not None else cpu_cap)
    reserve = max(.5, snapshot["total_bytes"] / _GB * .1) if reserve is None else reserve
    estimate = floor * _GB
    try:
        total = len(jobs)
    except TypeError:
        total = None
    iterator = iter(enumerate(jobs))
    context = mp.get_context("spawn")
    workers = []
    submitted = completed = peak_workers = peak_in_flight = worker_errors = 0
    started = time.perf_counter()
    wait_started, waited, last_wait_event = None, 0., -float("inf")
    calibrated, exhausted, pending = False, False, None
    reason = "exhausted"

    def emit(event, **fields):
        if on_event is not None:
            on_event(dict(event=event, elapsed_sec=time.perf_counter() - started,
                          workers=len(workers), active=sum(w.index is not None for w in workers),
                          submitted=submitted, completed=completed, **fields))

    def resize(previous, why):
        emit("parallel_resize", previous_workers=previous, reason=why,
             available_ram_gb=snapshot["available_bytes"] / _GB,
             memory_estimate_gb=estimate / _GB)

    def failure(worker, index, job, message, elapsed):
        nonlocal worker_errors
        worker_errors += 1
        message = "Lỗi worker: " + message
        emit("worker_error", index=index, arxiv_id=job.get("arxiv_id") if job else None,
             pid=worker.process.pid if worker else None, message=message)
        return _error_result(job, message, elapsed) if job is not None else None

    try:
        emit("parallel_plan", total_jobs=total, cpu_cap=cpu_cap, worker_cap=cap,
             adaptive_ram=adaptive_ram, reserve_ram_gb=reserve, ram_per_worker_gb=floor)
        while True:
            now = time.perf_counter()
            expired = deadline is not None and now >= deadline
            if expired:
                reason = "deadline"
            if pending is None and not exhausted and not expired:
                pending = next(iterator, None)
                exhausted = pending is None
            active = sum(w.index is not None for w in workers)
            if (expired or exhausted) and not active:
                break

            snapshot = _resource_snapshot([w.process.pid for w in workers])
            for worker in workers:
                worker.rss = snapshot["rss_by_pid"].get(worker.process.pid, 0)
                estimate = max(estimate, worker.rss * _HEADROOM)
            target = cap if calibrated else 1
            if adaptive_ram:
                budget = snapshot["available_bytes"] - reserve * _GB + sum(w.rss for w in workers)
                target = min(target, max(0, int(budget // estimate)))
            if expired or exhausted:
                target = 0
            idle = [w for w in workers if w.index is None]
            retiring = idle[:max(0, len(workers) - target)]
            if retiring:
                previous = len(workers)
                _dispose(retiring, gentle=True)
                for worker in retiring:
                    workers.remove(worker)
                resize(previous, "deadline" if expired else ("exhausted" if exhausted else "ram_pressure"))
                # Lấy ảnh mới sau khi hệ điều hành thu hồi RSS.
                continue

            if pending is not None and not expired and active < target:
                if wait_started is not None:
                    waited += now - wait_started
                    wait_started = None
                worker = next((w for w in workers if w.index is None), None)
                index, job = pending
                if worker is None:
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker_main, args=(child, enrich, threshold))
                    worker = _Worker(process, parent)
                    workers.append(worker)
                    try:
                        process.start()
                    except Exception as exc:
                        _dispose([worker])
                        workers.remove(worker)
                        submitted += 1
                        pending = None
                        result = failure(None, index, job, str(exc), 0.)
                        completed += 1
                        calibrated = True
                        yield index, result
                        del result, job
                        continue
                    finally:
                        child.close()
                    peak_workers = max(peak_workers, len(workers))
                    resize(len(workers) - 1, "calibration" if not calibrated else "capacity")
                if deadline is not None and time.perf_counter() >= deadline:
                    del job
                    continue
                worker.index, worker.job, worker.started = index, job, time.perf_counter()
                submitted += 1
                pending = None
                peak_in_flight = max(peak_in_flight, active + 1)
                try:
                    worker.connection.send(job)
                except Exception as exc:
                    result = failure(worker, index, job, str(exc), time.perf_counter() - worker.started)
                    previous = len(workers)
                    _dispose([worker])
                    workers.remove(worker)
                    resize(previous, "worker_error")
                    completed += 1
                    calibrated = True
                    yield index, result
                    del result
                del job
                continue

            busy = [w for w in workers if w.index is not None]
            if busy:
                ready = wait([w.connection for w in busy], timeout=_POLL_SEC)
                for worker in busy:
                    if worker.connection not in ready and worker.process.is_alive():
                        continue
                    index, job = worker.index, worker.job
                    try:
                        if worker.connection not in ready:
                            raise ChildProcessError(f"Tiến trình kết thúc, exitcode={worker.process.exitcode}")
                        kind, result, rss = worker.connection.recv()
                        estimate = max(estimate, rss * _HEADROOM)
                        if kind != "result":
                            raise RuntimeError(result)
                    except Exception as exc:
                        result = failure(worker, index, job, str(exc), time.perf_counter() - worker.started)
                        previous = len(workers)
                        _dispose([worker])
                        workers.remove(worker)
                        resize(previous, "worker_error")
                    worker.index, worker.job = None, None
                    completed += 1
                    calibrated = True
                    yield index, result
                    del result, job
                continue

            if wait_started is None:
                wait_started = now
                last_wait_event = -float("inf")
            duration = now - wait_started
            if now - last_wait_event >= 1:
                emit("ram_wait", wait_sec=duration, timeout_sec=timeout,
                     available_ram_gb=snapshot["available_bytes"] / _GB,
                     memory_estimate_gb=estimate / _GB)
                last_wait_event = now
            if duration >= timeout:
                reason = "ram_timeout"
                raise MemoryError("Không đủ RAM để chạy worker an toàn trong thời gian chờ")
            time.sleep(min(_POLL_SEC, timeout - duration))
    except GeneratorExit:
        reason = "closed"
        raise
    except KeyboardInterrupt:
        reason = "interrupted"
        raise
    except BaseException:
        if reason != "ram_timeout":
            reason = "error"
        raise
    finally:
        unwinding = sys.exc_info()[0] is not None
        if wait_started is not None:
            waited += time.perf_counter() - wait_started
        _dispose(workers)
        workers.clear()
        try:
            emit("parallel_summary", total_jobs=total, pending=None if total is None else total - submitted,
                 peak_workers=peak_workers, peak_in_flight=peak_in_flight,
                 worker_errors=worker_errors, ram_wait_sec=waited, stopped_reason=reason,
                 memory_estimate_gb=estimate / _GB, reserve_ram_gb=reserve,
                 cpu_cap=cpu_cap, worker_cap=cap)
        except Exception:
            if not unwinding:
                raise
