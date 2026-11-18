"""RAM-aware spawn scheduling; each process keeps the original parser's SaT cache.

Results arrive in completion order, and indices always refer to the input. With
result_dir, workers write their own partitions and return only metadata. The
iterable is read at most one job ahead.
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
    """Read the current cgroup and its parents; supports common cgroup v1/v2 layouts."""
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
            # Some namespaces return paths with '..'; never leave the mount.
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
    """Independent resource snapshot, so tests can substitute fixed data."""
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


def _worker_main(connection, enrich, fuzzy_threshold, sentence_cache_bytes, result_dir, debug_reports):
    """Spawn target must be importable; only the parent handles Ctrl-C."""
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
                # Import late so the BLAS limits are set before the parser/model is imported.
                if __package__:
                    from .parser_pipeline import parse_publication
                else:
                    from parser_pipeline import parse_publication
                result = parse_publication(job, enrich=enrich, fuzzy_threshold=fuzzy_threshold,
                                           sentence_cache_bytes=sentence_cache_bytes)
                rss = psutil.Process().memory_info().rss
                if result_dir is not None:
                    if __package__:
                        from .parser_storage import paper_name, persist_result
                    else:
                        from parser_storage import paper_name, persist_result
                    result = persist_result(Path(result_dir) / paper_name(job["arxiv_id"]), result,
                                            debug_reports=debug_reports, consume=True)
                kind = "result"
            except Exception as exc:
                kind, result = "error", f"{type(exc).__name__}: {exc}"
                rss = 0
            connection.send((kind, result, max(rss, psutil.Process().memory_info().rss)))
            del result, job
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def _error_result(job, message, elapsed):
    """Keep the full table/statistics contract when a process or IPC fails."""
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
    """Close the whole group with a shared timeout, leaving no orphaned children."""
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
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum}")
    return float(value)


def iter_parallel_results(jobs, *, num_workers="auto", max_workers=None,
                          reserve_ram_gb=None, ram_per_worker_gb=1.5, adaptive_ram=True,
                          enrich=False, fuzzy_threshold=.92, deadline=None, on_event=None,
                          ram_wait_timeout_sec=60, sentence_cache_bytes=4 * 1024 ** 2,
                          result_dir=None, debug_reports=False, job_timeout_sec=None,
                          _worker_target=None):
    """Yield ``(index, parse_publication(job))`` with at most one job per worker.

    GB means GiB (2**30 bytes); by default max(0.5 GiB, 10% of total RAM under
    the system limit) is reserved. The worker estimate is max(configured floor,
    peak RSS * 1.2). adaptive_ram=False removes the RAM limit but still
    calibrates on one first job. deadline is an absolute perf_counter value;
    close()/Ctrl-C stops the children immediately.

    on_event receives a JSON dict with event, elapsed_sec, workers, active,
    submitted, completed. parallel_summary has pending (None if the iterable
    has no len), peak_workers, peak_in_flight, worker_errors, ram_wait_sec,
    stopped_reason. The summary also keeps memory_estimate_gb, reserve_ram_gb,
    cpu_cap, worker_cap. completed includes worker errors; pending counts only
    jobs not yet handed out. result_dir=None keeps the API that returns full
    results. With a directory, each result is the metadata receipt of the
    written partition, and no Python tables pass through the pipe.

    job_timeout_sec limits the time of one paper: when exceeded, the worker is
    killed, the paper gets an error result (retried in a later session) and a
    new worker is created when needed. The summary counts timeouts in
    job_timeouts. The limit starts when the paper is handed out, so for a new
    worker it includes process start-up and the first SaT model load/download;
    do not set it too low.
    _worker_target is for tests only: it replaces the function run in the child.
    """
    if num_workers != "auto":
        _positive_int("num_workers", num_workers)
    if max_workers is not None:
        _positive_int("max_workers", max_workers)
    floor = _number("ram_per_worker_gb", ram_per_worker_gb)
    if floor == 0:
        raise ValueError("ram_per_worker_gb must be > 0")
    reserve = None if reserve_ram_gb is None else _number("reserve_ram_gb", reserve_ram_gb)
    timeout = _number("ram_wait_timeout_sec", ram_wait_timeout_sec)
    job_timeout = None if job_timeout_sec is None else _number("job_timeout_sec", job_timeout_sec)
    if job_timeout == 0:
        raise ValueError("job_timeout_sec must be > 0")
    threshold = _number("fuzzy_threshold", fuzzy_threshold)
    if threshold > 1:
        raise ValueError("fuzzy_threshold must be <= 1")
    if not isinstance(adaptive_ram, bool) or not isinstance(enrich, bool):
        raise ValueError("adaptive_ram and enrich must be booleans")
    if type(sentence_cache_bytes) is not int or sentence_cache_bytes < 0:
        raise ValueError("sentence_cache_bytes must be a nonnegative integer")
    if type(debug_reports) is not bool:
        raise ValueError("debug_reports must be a boolean")
    result_dir = None if result_dir is None else str(Path(result_dir).resolve())
    if on_event is not None and not callable(on_event):
        raise ValueError("on_event must be callable")
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
    submitted = completed = peak_workers = peak_in_flight = worker_errors = job_timeouts = 0
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
        message = "Worker error: " + message
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
                # Take a fresh snapshot after the OS reclaims RSS.
                continue

            if pending is not None and not expired and active < target:
                if wait_started is not None:
                    waited += now - wait_started
                    wait_started = None
                worker = next((w for w in workers if w.index is None), None)
                index, job = pending
                if worker is None:
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker_target or _worker_main,
                        args=(child, enrich, threshold, sentence_cache_bytes, result_dir, debug_reports))
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
                    timed_out = (job_timeout is not None and worker.connection not in ready
                                 and time.perf_counter() - worker.started >= job_timeout)
                    if worker.connection not in ready and worker.process.is_alive() and not timed_out:
                        continue
                    index, job = worker.index, worker.job
                    try:
                        if timed_out:
                            job_timeouts += 1
                            raise TimeoutError(f"paper exceeded the {job_timeout:g}s time limit; worker stopped")
                        if worker.connection not in ready:
                            raise ChildProcessError(f"Process exited, exitcode={worker.process.exitcode}")
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
                raise MemoryError("Insufficient RAM to start a worker safely within the wait timeout")
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
                 worker_errors=worker_errors, job_timeouts=job_timeouts, ram_wait_sec=waited, stopped_reason=reason,
                 memory_estimate_gb=estimate / _GB, reserve_ram_gb=reserve,
                 cpu_cap=cpu_cap, worker_cap=cap)
        except Exception:
            if not unwinding:
                raise
