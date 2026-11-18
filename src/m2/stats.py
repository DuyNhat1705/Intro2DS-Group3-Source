r"""Statistics and small report exports for the Jupyter Book chapter.

Usage::

    python -m src.m2.stats --out data/tables --work-dir data/m2

Reads the internal working data (never the corpus) and writes small CSV/JSON files to
``<work>/report_exports/`` that the book notebooks turn into interactive figures:

* ``status_counts.csv`` / ``failure_reasons.csv`` -- parsing success by status and reason;
* ``version_coverage.csv`` -- every publication with its versions' statuses;
* ``dedup_counts.json`` -- elements and BibTeX entries before / after deduplication;
* ``kind_counts.csv`` -- element kinds (from ``validation.json`` when present);
* ``files.csv`` -- compiled vs. redundant files and missing includes per version;
* ``bib_sources.csv`` / ``bib_methods.csv`` / ``entry_types.csv`` -- ``.bib`` vs ``\bibitem``;
* ``timing.csv`` -- per-version time and per-stage times; ``runs.json`` -- run summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _read(d: Path, columns=None) -> pa.Table | None:
    files = sorted(d.glob("*.parquet")) if d.is_dir() else []
    if not files:
        return None
    return pa.concat_tables([pq.read_table(f, columns=columns) for f in files]).unify_dictionaries()


def _write_csv(path: Path, header: list[str], rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def export(out_dir: Path, work_dir: Path, log=print) -> dict:
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    ed = work_dir / "report_exports"
    ed.mkdir(parents=True, exist_ok=True)
    res: dict = {}
    st = _read(work_dir / "parse_status")
    if st is not None:
        d = st.to_pydict()
        status = [str(s) for s in d["status"]]
        cnt = Counter(status)
        _write_csv(ed / "status_counts.csv", ["status", "versions"], sorted(cnt.items()))
        reasons = Counter((s, (r or "").split(":")[0][:80]) for s, r in zip(status, d["failure_reason"])
                          if s in ("failed", "missing", "empty"))
        _write_csv(ed / "failure_reasons.csv", ["status", "reason", "versions"],
                   [(s, r, c) for (s, r), c in sorted(reasons.items())])
        cov = defaultdict(Counter)
        nver = Counter()
        for a, s in zip(d["arxiv_id"], status):
            cov[a][s] += 1
            nver[a] += 1
        _write_csv(ed / "version_coverage.csv",
                   ["arxiv_id", "n_versions", "ok", "root_fallback", "partial", "empty", "missing", "failed"],
                   [(a, nver[a], c["ok"], c["root_fallback"], c["partial"], c["empty"], c["missing"], c["failed"])
                    for a, c in sorted(cov.items())])
        rows = []
        for i in range(st.num_rows):
            if status[i] in ("ok", "root_fallback", "partial"):
                rows.append((d["arxiv_id"][i], d["version"][i], d["n_compile_files"][i],
                             d["n_redundant_files"][i], d["n_missing_includes"][i], d["n_bib_raw"][i],
                             d["n_bibitem"][i], d["n_cited"][i], str(d["bib_source"][i]),
                             d["n_sentences"][i], d["n_formulas"][i], d["n_figures"][i], d["n_tables"][i],
                             d["max_depth"][i], d["n_section_cmds"][i], d["n_section_nodes"][i]))
        _write_csv(ed / "files.csv", ["arxiv_id", "version", "n_compile_files", "n_redundant_files",
                                      "n_missing_includes", "n_bib_raw", "n_bibitem", "n_cited",
                                      "bib_source", "n_sentences", "n_formulas", "n_figures", "n_tables",
                                      "max_depth", "n_section_cmds", "n_section_nodes"], rows)
        trows = []
        for i in range(st.num_rows):
            if status[i] in ("ok", "root_fallback", "partial"):
                trows.append((d["arxiv_id"][i], d["version"][i], status[i], d["elapsed_ms"][i],
                              d["t_gather_ms"][i], d["t_clean_ms"][i], d["t_hierarchy_ms"][i],
                              d["t_bib_ms"][i], d["t_dedup_ms"][i], bool(d["reused_from_prev"][i])))
        _write_csv(ed / "timing.csv", ["arxiv_id", "version", "status", "elapsed_ms", "gather_ms",
                                       "preprocess_ms", "hierarchy_ms", "bib_ms", "dedup_ms", "reused"], trows)
        non_empty = sum(c for s, c in cnt.items() if s not in ("empty", "missing"))
        succ = sum(cnt[s] for s in ("ok", "root_fallback", "partial"))
        succ1 = sum(1 for i in range(st.num_rows) if status[i] in ("ok", "root_fallback", "partial")
                    and (status[i] != "partial" or d["n_sentences"][i] >= 1))
        res["status"] = dict(cnt)
        res["success_rate_non_empty"] = succ / max(non_empty, 1)
        res["success_rate_with_sentence_rule"] = succ1 / max(non_empty, 1)
        res["failed_rate_non_empty"] = cnt["failed"] / max(non_empty, 1)
        res["versions_without_sentences"] = sum(1 for i in range(st.num_rows)
                                                if status[i] in ("ok", "root_fallback", "partial")
                                                and d["n_sentences"][i] == 0)
        res["redundant_files_total"] = int(sum(d["n_redundant_files"]))
        res["compile_files_total"] = int(sum(d["n_compile_files"]))
    ps = _read(work_dir / "paper_stats")
    if ps is not None:
        d = ps.to_pydict()
        res["dedup"] = {
            "papers": ps.num_rows,
            "nodes_before": int(sum(d["nodes_before"])), "elements_after": int(sum(d["elements_after"])),
            "bib_raw": int(sum(d["bib_raw"])), "bib_cited_before_dedup": int(sum(d["bib_before"])),
            "bib_after": int(sum(d["bib_after"])),
            "leaf_repeats_within_version": int(sum(d["leaf_repeats_within_version"])),
            "reused_versions": int(sum(d["reused_versions"])),
        }
        with open(ed / "dedup_counts.json", "w", encoding="utf-8") as fh:
            json.dump(res["dedup"], fh, indent=1)
        _write_csv(ed / "paper_dedup.csv", ["arxiv_id", "n_versions", "nodes_before", "elements_after",
                                            "bib_before", "bib_after", "elapsed_ms", "tex_bytes"],
                   zip(d["arxiv_id"], d["n_versions"], d["nodes_before"], d["elements_after"],
                       d["bib_before"], d["bib_after"], d["elapsed_ms"], d["tex_bytes"]))
    bf = _read(work_dir / "bibtex_full", columns=["source", "parse_method", "entry_type", "confidence",
                                                   "title", "year"])
    if bf is not None:
        d = bf.to_pydict()
        _write_csv(ed / "bib_sources.csv", ["source", "entries"], sorted(Counter(map(str, d["source"])).items()))
        _write_csv(ed / "bib_methods.csv", ["parse_method", "entries"],
                   sorted(Counter(map(str, d["parse_method"])).items()))
        _write_csv(ed / "entry_types.csv", ["entry_type", "entries"],
                   sorted(Counter(map(str, d["entry_type"])).items()))
        res["bib_missing_title_share"] = sum(1 for t in d["title"] if not t) / max(bf.num_rows, 1)
        res["bib_missing_year_share"] = sum(1 for y in d["year"] if y is None) / max(bf.num_rows, 1)
    vj = ed / "validation.json"
    if vj.exists():
        v = json.load(open(vj, encoding="utf-8"))
        _write_csv(ed / "kind_counts.csv", ["kind", "elements"], sorted(v.get("kinds", {}).items()))
    runs = []
    for f in sorted((work_dir / "runs").glob("*.json")) if (work_dir / "runs").is_dir() else []:
        try:
            r = json.load(open(f, encoding="utf-8"))
            r.pop("failed_papers", None)
            runs.append(r)
        except ValueError:
            pass
    with open(ed / "runs.json", "w", encoding="utf-8") as fh:
        json.dump(runs, fh, indent=1)
    with open(ed / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    log(json.dumps(res, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tables")
    ap.add_argument("--work-dir", default="data/m2")
    a = ap.parse_args(argv)
    export(Path(a.out), Path(a.work_dir))


if __name__ == "__main__":
    main()
