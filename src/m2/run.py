r"""Command-line entry point: ``python -m src.m2.run``.

Examples::

    python -m src.m2.run --data-dir data --out data/tables --work-dir data/m2 --workers auto
    python -m src.m2.run --limit 20 --workers 2
    python -m src.m2.run --papers 2608.04009 2608-04010 --workers 0     # in-process (debug)
    python -m src.m2.run --resume                 # skip papers whose signature is unchanged
    python -m src.m2.run --compact-only           # only merge existing shards
    python -m src.m2.run --refine-bibitem         # optional AnyStyle refinement (needs Ruby)

A run summary (wall time, per-paper / per-version time, peak RAM, output size, cache and reuse
rates, status counts) is written to ``<work-dir>/runs/<run_id>.json`` and printed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

from . import config as C
from .discover import discover
from .pool import Pool, auto_workers, run_inprocess
from .storage import compact, dir_size, load_manifest


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[k]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.m2.run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="M1 corpus root (read-only)")
    ap.add_argument("--out", default="data/tables", help="published tables root")
    ap.add_argument("--work-dir", default="data/m2", help="internal working data root")
    ap.add_argument("--workers", default="auto", help="'auto', N >= 1, or 0 for in-process")
    ap.add_argument("--limit", type=int, default=None, help="first K papers (sorted by key)")
    ap.add_argument("--papers", nargs="*", default=None, help="arXiv IDs or paper keys")
    ap.add_argument("--ranges", nargs="*", default=None,
                    help="restrict to arXiv ID ranges, e.g. 2608:4007-9006 (the M1 scraper ranges)")
    ap.add_argument("--log-dir", default=None, help="where run summaries go (default <work-dir>/runs)")
    ap.add_argument("--sample", choices=["dev"], default=None,
                    help="restrict to the 200-paper stratified dev sample (tests/samples.py)")
    ap.add_argument("--resume", action="store_true", help="skip papers with unchanged signature")
    ap.add_argument("--compact", action="store_true", help="merge shards after the run")
    ap.add_argument("--compact-only", action="store_true", help="only merge existing shards")
    ap.add_argument("--n-shards", type=int, default=8, help="number of shards after --compact")
    ap.add_argument("--timeout", type=float, default=C.PAPER_TIMEOUT_S, help="per-paper deadline (s)")
    ap.add_argument("--refine-bibitem", action="store_true",
                    help="refine low-confidence bibitems with the AnyStyle CLI (optional)")
    ap.add_argument("--no-progress", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    data, out, work = Path(a.data_dir), Path(a.out), Path(a.work_dir)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    if a.compact_only:
        s = compact(out, work, a.n_shards)
        print(json.dumps(s, indent=1))
        return 0
    if a.refine_bibitem:
        import os
        os.environ["M2_REFINE_BIBITEM"] = "1"
    run_id = time.strftime("%Y%m%d-%H%M%S")
    t0 = time.perf_counter()
    papers = a.papers
    if a.sample == "dev":
        from .tests.samples import DEV_SAMPLE
        papers = list(DEV_SAMPLE) + list(papers or [])
    if a.ranges:
        wanted = []
        for spec in a.ranges:
            month, span = spec.split(":")
            lo, hi = (int(x) for x in span.split("-"))
            wanted.append((month, lo, hi))

        def in_ranges(key: str) -> bool:
            m, _, num = key.partition("-")
            return num.isdigit() and any(m == mo and lo <= int(num) <= hi for mo, lo, hi in wanted)
        from .discover import list_paper_dirs
        keys = [d.name for d in list_paper_dirs(data) if in_ranges(d.name)]
        papers = (papers or []) + keys
        if not papers:
            print("[m2] no paper folder in the requested ranges")
            return 1
    jobs = discover(data, papers=papers, limit=a.limit)
    t_disc = time.perf_counter() - t0
    prev = {}
    if a.resume:
        prev = {k: r["signature"] for k, r in load_manifest(work).items()
                if r.get("parser_version") == C.PARSER_VERSION}
    nw = auto_workers() if a.workers == "auto" else int(a.workers)
    print(f"[m2] run {run_id}: {len(jobs)} papers, workers={nw}, resume={a.resume} "
          f"(discovery {t_disc:.1f}s)", flush=True)
    progress = None
    if not a.no_progress:
        try:
            from tqdm import tqdm
            progress = tqdm(total=len(jobs), unit="paper", file=sys.stdout, mininterval=2)
        except ImportError:
            progress = None
    t1 = time.perf_counter()
    if nw == 0:
        st = run_inprocess(jobs, prev, run_id, out, work, progress)
    else:
        pool = Pool(nw, run_id, out, work, timeout_s=a.timeout, progress=progress,
                    log=lambda m: print(m, flush=True))
        st = pool.run(jobs, prev)
    wall = time.perf_counter() - t1
    if progress:
        progress.close()
    comp = None
    t_comp = 0.0
    if a.compact:
        tc = time.perf_counter()
        comp = compact(out, work, a.n_shards)
        t_comp = time.perf_counter() - tc
    sizes = {t: dir_size(out / t) for t in ("elements", "hierarchy", "bibtex")}
    summary = {
        "run_id": run_id, "parser_version": C.PARSER_VERSION, "papers": len(jobs),
        "workers": nw, "done": st.done, "skipped": st.skipped, "failed": st.failed,
        "timeouts": st.timeouts, "crashes": st.crashes, "safe_retries": st.safe_retries,
        "recycled": st.recycled, "discovery_s": round(t_disc, 2), "wall_s": round(wall, 2),
        "compact_s": round(t_comp, 2),
        "paper_ms_mean": round(statistics.fmean(st.paper_ms), 2) if st.paper_ms else 0,
        "paper_ms_median": round(statistics.median(st.paper_ms), 2) if st.paper_ms else 0,
        "paper_ms_p95": round(_pct(st.paper_ms, 0.95), 2),
        "version_ms_mean": round(statistics.fmean(st.version_ms), 2) if st.version_ms else 0,
        "version_ms_median": round(statistics.median(st.version_ms), 2) if st.version_ms else 0,
        "version_ms_p95": round(_pct(st.version_ms, 0.95), 2),
        "peak_rss_mb": round(st.peak_rss_bytes / 2 ** 20, 1),
        "peak_workers_rss_mb": round(st.peak_workers_rss_bytes / 2 ** 20, 1),
        "status_counts": st.status_counts,
        "paragraph_cache_hit_rate": round(st.cache_hits / max(st.cache_hits + st.cache_misses, 1), 4),
        "reused_versions": st.reused_versions,
        "output_bytes": sizes, "output_bytes_total": sum(sizes.values()),
        "work_bytes_total": dir_size(work),
        "failed_papers": st.failed_papers[:50], "compaction": comp,
    }
    rd = Path(a.log_dir) if a.log_dir else work / "runs"
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd / f"{run_id}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "failed_papers"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
