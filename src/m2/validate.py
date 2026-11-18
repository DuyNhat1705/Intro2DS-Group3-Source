r"""Corpus-wide structural checks of the output tables (prompt §5.11).

Usage::

    python -m src.m2.validate --out data/tables --work-dir data/m2 --data-dir data

Checks (per shard, so memory stays bounded):

* every ``element_id`` / ``parent_id`` of ``hierarchy`` exists in ``elements``;
* every version is a tree rooted at ``<key>#root<v>``: one parent per node, no cycle, every node
  reaches the root;
* no empty leaf content; formulas start with ``\begin{equation}`` and have balanced braces; no
  ``\(``, ``\[`` or ``$$`` in any content;
* residual LaTeX rate (sentences with a control word outside ``$..$``/``\cite``/``\ref``);
* cite coverage (``\cite`` keys in contents that exist in ``bibtex`` for the paper);
* section sanity (raw ``\section`` commands vs section nodes, from ``parse_status``);
* accuracy proxies from the paper's own M1 metadata: abstract fidelity
  (``rapidfuzz.token_set_ratio`` of the parsed abstract vs ``metadata.json``) and title fidelity.

Writes ``<work>/report_exports/validation.json``, ``fidelity.csv`` and ``depth_hist.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
from rapidfuzz import fuzz

from .clean import has_residual_latex
from .discover import arxiv_id_to_key
from .mathnorm import braces_balanced

_LEAVES = {"sentence", "formula", "figure", "table"}
_CITE_RE = re.compile(r"\\cite\{([^{}]*)\}")


def _norm_text(t: str) -> str:
    t = re.sub(r"\$[^$]*\$", " ", t or "")
    t = re.sub(r"\\[A-Za-z]+\*?(\{[^{}]*\})?", " ", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def _files(d: Path) -> list[Path]:
    return sorted(d.glob("*.parquet")) if d.is_dir() else []


def validate(out_dir: Path, work_dir: Path, data_dir: Path | None = None, log=print) -> dict:
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    R = Counter()
    depth_hist = Counter()
    fidelity_rows = []
    bib_keys: dict[str, set[str]] = defaultdict(set)
    for f in _files(out_dir / "bibtex"):
        t = pq.read_table(f, columns=["arxiv_id", "citation_key"])
        for a, k in zip(t["arxiv_id"].to_pylist(), t["citation_key"].to_pylist()):
            bib_keys[a].add(k)
    examples = defaultdict(list)

    def ex(kind, val):
        R[kind] += 1
        if len(examples[kind]) < 5:
            examples[kind].append(val)

    hfiles = {f.name: f for f in _files(out_dir / "hierarchy")}
    efiles = {f.name: f for f in _files(out_dir / "elements")}
    for name in sorted(set(hfiles) | set(efiles)):
        E = pq.read_table(efiles[name]) if name in efiles else None
        H = pq.read_table(hfiles[name]) if name in hfiles else None
        kinds: dict[str, str] = {}
        content: dict[str, str] = {}
        if E is not None:
            ids = E["element_id"].to_pylist()
            ks = E["kind"].cast("string").to_pylist() if hasattr(E["kind"], "cast") else E["kind"].to_pylist()
            cs = E["content"].to_pylist()
            aids = E["arxiv_id"].to_pylist()
            R["elements"] += len(ids)
            dup = len(ids) - len(set(ids))
            if dup:
                ex("duplicate_element_ids", name)
            for eid, k, c, aid in zip(ids, ks, cs, aids):
                kinds[eid] = k
                content[eid] = c or ""
                R[f"kind:{k}"] += 1
                if k in _LEAVES and not (c or "").strip():
                    ex("empty_leaf", eid)
                if c and ("\\(" in c or "\\[" in c or "$$" in c):
                    if k != "formula" or "$$" in c or "\\[" in c:
                        ex("raw_math_delimiter", eid)
                if k == "formula":
                    if not c.startswith("\\begin{equation}"):
                        ex("formula_not_equation", eid)
                    if not braces_balanced(c):
                        ex("formula_unbalanced_braces", eid)
                elif k == "sentence":
                    R["sentences"] += 1
                    if has_residual_latex(c):
                        R["sentences_residual_latex"] += 1
                        if len(examples["residual_examples"]) < 30:
                            examples["residual_examples"].append(c[:200])
                if c and "\\cite{" in c and k in ("sentence", "figure", "table"):
                    for m in _CITE_RE.finditer(c):
                        for key in m.group(1).split(","):
                            if key:
                                R["cite_keys"] += 1
                                if key in bib_keys.get(aid, ()):
                                    R["cite_keys_covered"] += 1
        if H is None:
            continue
        aids = H["arxiv_id"].to_pylist()
        vers = H["version"].to_pylist()
        eids = H["element_id"].to_pylist()
        pids = H["parent_id"].to_pylist()
        R["hierarchy_rows"] += len(eids)
        groups: dict[tuple, list[tuple[str, str]]] = defaultdict(list)
        for a, v, e, p in zip(aids, vers, eids, pids):
            groups[(a, v)].append((e, p))
            if e not in kinds:
                ex("missing_element", e)
            if p not in kinds:
                ex("missing_parent", p)
        for (a, v), edges in groups.items():
            R["versions_with_tree"] += 1
            root = f"{arxiv_id_to_key(a)}#root{v}"
            parent = {}
            for e, p in edges:
                if e in parent:
                    ex("two_parents", f"{a} v{v} {e}")
                parent[e] = p
            if root in parent:
                ex("root_has_parent", root)
            depth = {root: 0}
            for e in parent:
                chain = []
                x = e
                seen = set()
                while x not in depth:
                    if x in seen or x not in parent:
                        ex("cycle_or_orphan", f"{a} v{v} {e}")
                        break
                    seen.add(x)
                    chain.append(x)
                    x = parent[x]
                else:
                    d = depth[x]
                    for y in reversed(chain):
                        d += 1
                        depth[y] = d
            for e, d in depth.items():
                if e != root:
                    depth_hist[(kinds.get(e, "?"), d)] += 1
            # fidelity for the latest version is computed below
        del E, H

    # ---- accuracy proxies
    if data_dir is not None:
        latest: dict[str, int] = {}
        for f in _files(work_dir / "parse_status"):
            t = pq.read_table(f, columns=["arxiv_id", "version", "status"])
            for a, v, s in zip(t["arxiv_id"].to_pylist(), t["version"].to_pylist(),
                               t["status"].cast("string").to_pylist()):
                if s in ("ok", "root_fallback", "partial"):
                    latest[a] = max(latest.get(a, 0), v)
        abs_text: dict[str, str] = {}
        titles: dict[str, str] = {}
        for name in sorted(set(hfiles) & set(efiles)):
            E = pq.read_table(efiles[name], columns=["element_id", "kind", "content"])
            H = pq.read_table(hfiles[name], columns=["arxiv_id", "version", "element_id"])
            kind = dict(zip(E["element_id"].to_pylist(), E["kind"].cast("string").to_pylist()))
            cont = dict(zip(E["element_id"].to_pylist(), E["content"].to_pylist()))
            for a, v, e in zip(H["arxiv_id"].to_pylist(), H["version"].to_pylist(), H["element_id"].to_pylist()):
                if v == latest.get(a) and kind.get(e) == "abstract" and a not in abs_text:
                    abs_text[a] = cont.get(e) or ""
            for e, k in kind.items():
                if k == "document":
                    key, _, r = e.partition("#root")
                    titles[(key, int(r))] = cont.get(e) or ""
        for a, v in sorted(latest.items()):
            key = arxiv_id_to_key(a)
            try:
                meta = json.load(open(Path(data_dir) / key / "metadata.json", encoding="utf-8"))
            except (OSError, ValueError):
                continue
            ma, mt = meta.get("abstract") or "", meta.get("title") or ""
            pa_ = abs_text.get(a, "")
            sa = fuzz.token_set_ratio(_norm_text(pa_), _norm_text(ma)) if pa_ and ma else None
            pt = titles.get((key, v), "")
            st = fuzz.token_set_ratio(_norm_text(pt), _norm_text(mt)) if pt and mt else None
            fidelity_rows.append({"arxiv_id": a, "version": v, "has_abstract": bool(pa_),
                                  "abstract_fidelity": sa, "title_fidelity": st})
    absf = [r["abstract_fidelity"] for r in fidelity_rows if r["abstract_fidelity"] is not None]
    titf = [r["title_fidelity"] for r in fidelity_rows if r["title_fidelity"] is not None]

    # ---- section sanity
    sec_cmds = sec_nodes = 0
    for f in _files(work_dir / "parse_status"):
        t = pq.read_table(f, columns=["n_section_cmds", "n_section_nodes"])
        sec_cmds += int(pc.sum(t["n_section_cmds"]).as_py() or 0)
        sec_nodes += int(pc.sum(t["n_section_nodes"]).as_py() or 0)

    structural = ["missing_element", "missing_parent", "two_parents", "root_has_parent",
                  "cycle_or_orphan", "empty_leaf", "raw_math_delimiter", "formula_not_equation",
                  "formula_unbalanced_braces", "duplicate_element_ids"]
    rep = {
        "elements": R["elements"], "hierarchy_rows": R["hierarchy_rows"],
        "versions_with_tree": R["versions_with_tree"],
        "violations": {k: R[k] for k in structural},
        "violations_total": sum(R[k] for k in structural),
        "sentences": R["sentences"],
        "residual_latex_rate": R["sentences_residual_latex"] / max(R["sentences"], 1),
        "cite_keys": R["cite_keys"],
        "cite_coverage": R["cite_keys_covered"] / max(R["cite_keys"], 1),
        "section_cmds": sec_cmds, "section_nodes": sec_nodes,
        "abstract_fidelity_median": statistics.median(absf) if absf else None,
        "abstract_fidelity_mean": statistics.fmean(absf) if absf else None,
        "abstract_found_share": (sum(r["has_abstract"] for r in fidelity_rows) / len(fidelity_rows))
        if fidelity_rows else None,
        "title_fidelity_median": statistics.median(titf) if titf else None,
        "kinds": {k[5:]: v for k, v in R.items() if k.startswith("kind:")},
        "examples": dict(examples),
    }
    ed = work_dir / "report_exports"
    ed.mkdir(parents=True, exist_ok=True)
    with open(ed / "validation.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1, ensure_ascii=False)
    with open(ed / "fidelity.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["arxiv_id", "version", "has_abstract", "abstract_fidelity", "title_fidelity"])
        w.writeheader()
        w.writerows(fidelity_rows)
    with open(ed / "depth_hist.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "depth", "count"])
        for (k, d), c in sorted(depth_hist.items()):
            w.writerow([k, d, c])
    log(json.dumps({k: v for k, v in rep.items() if k not in ("examples",)}, indent=1))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/tables")
    ap.add_argument("--work-dir", default="data/m2")
    ap.add_argument("--data-dir", default="data")
    a = ap.parse_args(argv)
    rep = validate(Path(a.out), Path(a.work_dir), Path(a.data_dir))
    return 1 if rep["violations_total"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
