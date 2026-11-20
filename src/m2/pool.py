r"""Custom worker pool: watchdog with per-paper deadline, respawn, safe-mode retry, LPT scheduling,
RSS-based recycling.

* one process per worker (``spawn`` start method on every platform, worker code lives in this
  module so it works on Windows and from notebooks); the parent sends one paper at a time;
* the parent watchdog kills a worker whose paper exceeds the deadline (``signal.alarm`` cannot
  interrupt a regex running in C), respawns a worker and retries the paper once in safe mode;
  a second failure records the paper as ``failed`` -- crashes take the same path;
* papers are dispatched largest first (LPT on total ``.tex``/``.bib`` bytes);
* a worker above the RSS limit flushes, exits and is replaced; the parent samples the RSS of
  itself and all workers to report the peak;
* workers write their own Parquet shards (:mod:`storage`); only small metadata travels back.
  Papers that were finished but not yet flushed by a killed worker are re-queued.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from . import config as C
from .discover import PaperJob


# ---------------------------------------------------------------------------------------
# Worker side
# ---------------------------------------------------------------------------------------
def _summary(res, sig: str, elapsed_ms: float) -> dict:
    st = {}
    for r in res.status:
        st[r["status"]] = st.get(r["status"], 0) + 1
    return {"status": st, "elements": len(res.elements), "bibtex": len(res.bibtex),
            "elapsed_ms": elapsed_ms, "signature": sig, "error": res.error,
            "version_ms": [r["elapsed_ms"] for r in res.status if r["status"] not in ("missing", "empty")],
            "cache_hits": res.stats.get("cache_hits", 0), "cache_misses": res.stats.get("cache_misses", 0),
            "reused": res.stats.get("reused_versions", 0)}


def worker_main(wid: int, run_id: str, out_dir: str, work_dir: str, inq, outq,
                rss_limit_mb: float, flush_papers: int) -> None:
    # imports inside the worker keep the parent light
    from .pipeline import load_paper_files, process_paper
    from .storage import ShardWriter

    writer = ShardWriter(Path(out_dir), Path(work_dir), run_id, wid, flush_papers=flush_papers)
    me = psutil.Process()
    while True:
        try:
            msg = inq.get()
        except (EOFError, OSError):
            return
        if msg is None:
            outq.put(("flushed", wid, writer.flush()))
            outq.put(("bye", wid, me.memory_info().rss))
            return
        job, prev_sig, safe = msg
        t0 = time.perf_counter()
        loaded, sig = load_paper_files(job)
        if prev_sig and prev_sig == sig:
            outq.put(("skip", wid, job.paper_key, sig))
            continue
        res = process_paper(job, safe_mode=safe, loaded=loaded)
        del loaded
        if safe:
            for r in res.status:
                if r["status"] in ("ok", "root_fallback"):
                    r["status"] = "partial"
                    r["retry_mode"] = "safe"
        elapsed = 1000 * (time.perf_counter() - t0)
        writer.add(res, sig, len(job.versions), job.tex_bytes, elapsed)
        outq.put(("done", wid, job.paper_key, _summary(res, sig, elapsed)))
        if writer.should_flush():
            outq.put(("flushed", wid, writer.flush()))
        rss = me.memory_info().rss
        if rss > rss_limit_mb * 1024 * 1024:
            outq.put(("flushed", wid, writer.flush()))
            outq.put(("recycle", wid, rss))
            return


# ---------------------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------------------
def cgroup_memory_limit() -> int | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < 1 << 50:
                return int(v)
        except OSError:
            pass
    return None


def auto_workers(est_mb: float = C.EST_WORKER_RAM_MB) -> int:
    try:
        cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu = os.cpu_count() or 1
    avail = psutil.virtual_memory().available
    lim = cgroup_memory_limit()
    if lim:
        avail = min(avail, lim - psutil.Process().memory_info().rss)
    by_ram = int(max(avail, 0) / (est_mb * 1024 * 1024))
    return max(1, min(cpu, by_ram))


@dataclass
class _W:
    wid: int
    proc: mp.Process
    inq: object
    busy: tuple | None = None          # (job, start_time, safe)
    unflushed: set = field(default_factory=set)
    closing: bool = False


@dataclass
class PoolStats:
    done: int = 0
    skipped: int = 0
    failed: int = 0
    timeouts: int = 0
    crashes: int = 0
    safe_retries: int = 0
    recycled: int = 0
    respawns: int = 0
    peak_rss_bytes: int = 0
    peak_workers_rss_bytes: int = 0
    paper_ms: list = field(default_factory=list)
    version_ms: list = field(default_factory=list)
    status_counts: dict = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    reused_versions: int = 0
    failed_papers: list = field(default_factory=list)


class Pool:
    def __init__(self, n_workers: int, run_id: str, out_dir: Path, work_dir: Path,
                 timeout_s: float = C.PAPER_TIMEOUT_S, rss_limit_mb: float = C.WORKER_RSS_LIMIT_MB,
                 flush_papers: int = C.SHARD_FLUSH_PAPERS, progress=None, log=print):
        self.n = max(1, n_workers)
        self.run_id = run_id
        self.out_dir, self.work_dir = Path(out_dir), Path(work_dir)
        self.timeout_s = timeout_s
        self.rss_limit_mb = rss_limit_mb
        self.flush_papers = flush_papers
        self.ctx = mp.get_context("spawn")
        self.outq = self.ctx.Queue()
        self.workers: dict[int, _W] = {}
        self.next_wid = 0
        self.stats = PoolStats()
        self.progress = progress
        self.log = log
        self._failed_writer = None
        self._cache: dict = {}
        self._cache_done = [0, 0]

    def _spawn(self) -> _W:
        wid = self.next_wid
        self.next_wid += 1
        inq = self.ctx.Queue()
        p = self.ctx.Process(target=worker_main, args=(wid, self.run_id, str(self.out_dir),
                                                       str(self.work_dir), inq, self.outq,
                                                       self.rss_limit_mb, self.flush_papers),
                             daemon=True)
        p.start()
        w = _W(wid, p, inq)
        self.workers[wid] = w
        return w

    def _record_failed(self, job: PaperJob, reason: str) -> None:
        from .pipeline import PaperResult, _status_row
        from .storage import ShardWriter
        if self._failed_writer is None:
            self._failed_writer = ShardWriter(self.out_dir, self.work_dir, self.run_id, 99,
                                              flush_papers=10 ** 9)
        res = PaperResult(arxiv_id=job.arxiv_id, paper_key=job.paper_key, error=reason)
        res.status = [_status_row(job, vi.version, None,
                                  "failed" if vi.on_disk else "missing", reason) for vi in job.versions]
        self._failed_writer.add(res, "", len(job.versions), job.tex_bytes, 0.0)
        self.stats.failed += 1
        self.stats.failed_papers.append((job.paper_key, reason))
        for r in res.status:
            self.stats.status_counts[r["status"]] = self.stats.status_counts.get(r["status"], 0) + 1
        if self.progress:
            self.progress.update(1)

    def _kill(self, w: _W, todo: deque, why: str) -> None:
        """Terminate worker ``w``; re-queue its unflushed papers and retry/fail its paper."""
        try:
            w.proc.kill()
            w.proc.join(5)
        except (OSError, AttributeError, ValueError):
            pass
        del self.workers[w.wid]
        self._retire_cache(w.wid)
        for key in list(w.unflushed):
            job = self._jobs_by_key.get(key)
            if job is not None:
                todo.appendleft((job, False))
                self.stats.done -= 1
                if self.progress:
                    self.progress.update(-1)
        if w.busy is not None:
            job, _t, safe = w.busy
            if not safe:
                self.stats.safe_retries += 1
                todo.appendleft((job, True))
            else:
                self._record_failed(job, why)
        self.stats.respawns += 1
        self._spawn()

    def _sample_rss(self) -> None:
        total = 0
        wtotal = 0
        try:
            total = psutil.Process().memory_info().rss
            for w in self.workers.values():
                if w.proc.is_alive():
                    r = psutil.Process(w.proc.pid).memory_info().rss
                    wtotal += r
        except (psutil.Error, ValueError):
            pass
        total += wtotal
        self.stats.peak_rss_bytes = max(self.stats.peak_rss_bytes, total)
        self.stats.peak_workers_rss_bytes = max(self.stats.peak_workers_rss_bytes, wtotal)

    def run(self, jobs: list[PaperJob], prev_sigs: dict[str, str] | None = None) -> PoolStats:
        prev_sigs = prev_sigs or {}
        self._jobs_by_key = {j.paper_key: j for j in jobs}
        todo: deque = deque((j, False) for j in sorted(jobs, key=lambda j: -j.tex_bytes))
        for _ in range(min(self.n, max(1, len(jobs)))):
            self._spawn()
        last_sample = 0.0
        while todo or any(w.busy for w in self.workers.values()):
            for w in list(self.workers.values()):
                if w.busy is None and not w.closing and todo and w.proc.is_alive():
                    job, safe = todo.popleft()
                    w.busy = (job, time.time(), safe)
                    w.inq.put((job, prev_sigs.get(job.paper_key, ""), safe))
            try:
                msg = self.outq.get(timeout=0.2)
            except queue.Empty:
                msg = None
            if msg is not None:
                self._handle(msg, todo)
            now = time.time()
            if now - last_sample > 1.0:
                self._sample_rss()
                last_sample = now
            for w in list(self.workers.values()):
                if w.busy is not None and now - w.busy[1] > self.timeout_s * (2 if w.busy[2] else 1):
                    self.stats.timeouts += 1
                    self.log(f"[pool] timeout on {w.busy[0].paper_key} (worker {w.wid})")
                    self._kill(w, todo, f"timeout>{self.timeout_s:.0f}s")
                elif not w.proc.is_alive() and not w.closing:
                    # crash (or recycle message still in the queue)
                    self._drain(todo)
                    if w.wid in self.workers and not w.closing:
                        self.stats.crashes += 1
                        self.log(f"[pool] worker {w.wid} died (exit {w.proc.exitcode})")
                        self._kill(w, todo, f"crash(exit={w.proc.exitcode})")
        # shutdown
        for w in self.workers.values():
            w.closing = True
            try:
                w.inq.put(None)
            except (OSError, ValueError):
                pass
        deadline = time.time() + 120
        while any(w.proc.is_alive() for w in self.workers.values()) and time.time() < deadline:
            try:
                msg = self.outq.get(timeout=0.5)
                self._handle(msg, todo)
            except queue.Empty:
                pass
            self._sample_rss()
        self._drain(todo)
        for w in self.workers.values():
            w.proc.join(5)
        if self._failed_writer is not None:
            self._failed_writer.flush()
        return self.stats

    def _drain(self, todo: deque) -> None:
        while True:
            try:
                msg = self.outq.get_nowait()
            except queue.Empty:
                return
            self._handle(msg, todo)

    def _handle(self, msg: tuple, todo: deque) -> None:
        kind, wid = msg[0], msg[1]
        w = self.workers.get(wid)
        if kind == "done":
            key, meta = msg[2], msg[3]
            self.stats.done += 1
            self.stats.paper_ms.append(meta["elapsed_ms"])
            self.stats.version_ms.extend(meta["version_ms"])
            for s, c in meta["status"].items():
                self.stats.status_counts[s] = self.stats.status_counts.get(s, 0) + c
            self.stats.reused_versions += meta["reused"]
            self._cache[wid] = (meta["cache_hits"], meta["cache_misses"])
            if w is not None:
                w.busy = None
                w.unflushed.add(key)
            if self.progress:
                self.progress.update(1)
        elif kind == "skip":
            self.stats.skipped += 1
            if w is not None:
                w.busy = None
            if self.progress:
                self.progress.update(1)
        elif kind == "flushed":
            if w is not None:
                w.unflushed.difference_update(msg[2])
        elif kind in ("recycle", "bye"):
            if w is not None:
                if kind == "recycle":
                    self.stats.recycled += 1
                    w.closing = True
                    w.proc.join(10)
                    del self.workers[wid]
                    self._retire_cache(wid)
                    self._spawn()
                else:
                    w.closing = True
        self.stats.cache_hits = sum(h for h, _m in self._cache.values()) + self._cache_done[0]
        self.stats.cache_misses = sum(m for _h, m in self._cache.values()) + self._cache_done[1]

    def _retire_cache(self, wid: int) -> None:
        h, m = self._cache.pop(wid, (0, 0))
        self._cache_done[0] += h
        self._cache_done[1] += m


def run_inprocess(jobs: list[PaperJob], prev_sigs: dict[str, str], run_id: str, out_dir: Path,
                  work_dir: Path, progress=None) -> PoolStats:
    """Sequential execution without watchdog (debugging, notebooks)."""
    from .pipeline import load_paper_files, process_paper, worker_caches
    from .storage import ShardWriter

    st = PoolStats()
    writer = ShardWriter(out_dir, work_dir, run_id, 0)
    me = psutil.Process()
    for job in sorted(jobs, key=lambda j: -j.tex_bytes):
        t0 = time.perf_counter()
        loaded, sig = load_paper_files(job)
        if prev_sigs.get(job.paper_key) == sig:
            st.skipped += 1
        else:
            res = process_paper(job, loaded=loaded)
            elapsed = 1000 * (time.perf_counter() - t0)
            writer.add(res, sig, len(job.versions), job.tex_bytes, elapsed)
            meta = _summary(res, sig, elapsed)
            st.done += 1
            st.paper_ms.append(elapsed)
            st.version_ms.extend(meta["version_ms"])
            st.reused_versions += meta["reused"]
            for s, c in meta["status"].items():
                st.status_counts[s] = st.status_counts.get(s, 0) + c
            if writer.should_flush():
                writer.flush()
        st.peak_rss_bytes = max(st.peak_rss_bytes, me.memory_info().rss)
        if progress:
            progress.update(1)
    writer.flush()
    sp, pc_ = worker_caches()
    st.cache_hits, st.cache_misses = pc_.hits, pc_.misses
    return st
