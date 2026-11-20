r"""Profile the parser on a sample (prompt §5.10): sequential, in-process, nothing written.

Usage::

    python -m src.m2.profile_run [--sample dev] [--limit 200] [--top 30] [--pyinstrument]

Prints per-version timing percentiles, the per-stage totals and the top functions by cumulative
time (``cProfile``; ``--pyinstrument`` prints a pyinstrument call tree instead).
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import statistics
import time
from pathlib import Path

from .discover import discover
from .pipeline import load_paper_files, process_paper


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sample", choices=["dev"], default="dev")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--pyinstrument", action="store_true")
    a = ap.parse_args(argv)
    from .tests.samples import DEV_SAMPLE
    jobs = discover(Path(a.data_dir), papers=list(DEV_SAMPLE), limit=a.limit)
    loaded = [(j, load_paper_files(j)[0]) for j in jobs]  # I/O excluded from the profile

    def work():
        vms, stages = [], {}
        for j, l in loaded:
            r = process_paper(j, loaded=l)
            for s in r.status:
                if s["status"] in ("ok", "root_fallback", "partial"):
                    vms.append(s["elapsed_ms"])
                    for k in ("t_gather_ms", "t_clean_ms", "t_hierarchy_ms", "t_bib_ms", "t_dedup_ms"):
                        stages[k] = stages.get(k, 0.0) + s[k]
        return vms, stages

    t0 = time.perf_counter()
    if a.pyinstrument:
        from pyinstrument import Profiler
        prof = Profiler()
        prof.start()
        vms, stages = work()
        prof.stop()
        print(prof.output_text(unicode=False, color=False)[:20000])
    else:
        pr = cProfile.Profile()
        pr.enable()
        vms, stages = work()
        pr.disable()
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(a.top)
        print(s.getvalue()[:12000])
        s = io.StringIO()
        pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(a.top)
        print(s.getvalue()[:8000])
    wall = time.perf_counter() - t0
    vms.sort()
    print(f"papers={len(jobs)} versions={len(vms)} wall={wall:.1f}s (profiler overhead included)")
    if vms:
        print(f"version ms: mean={statistics.fmean(vms):.1f} median={statistics.median(vms):.1f} "
              f"p95={vms[int(0.95 * (len(vms) - 1))]:.1f}")
    print({k: round(v / 1000, 2) for k, v in stages.items()}, "(seconds per stage)")


if __name__ == "__main__":
    main()
