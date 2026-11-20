r"""Corpus profiling (Milestone 2, Phase 1). Read-only pass over ``data/`` that measures the
properties the parser design depends on. No parsing logic: only regex counts.

Usage::

    python -m src.m2.profile_corpus --data-dir data --out data/m2/profile [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .discover import discover_paper, list_paper_dirs, load_state
from .io_utils import content_hash, decode_bytes, scan_version_dirs, split_dup_suffix
from .preprocess import Literals, clean_file_text

_DOCCLASS_RE = re.compile(r"\\document(class|style)\s*(?:\[[^\]]*\])?\s*\{\s*([^}\s]+)\s*\}")
_BEGIN_DOC_RE = re.compile(r"\\begin\s*\{document\}")
_INCLUDE_RE = re.compile(
    r"\\(input|include|subfile|import|subimport|includefrom|subincludefrom|InputIfFileExists|"
    r"inputfrom|subinputfrom|includeonly|loadglsentries)(?![A-Za-z])\s*(\{)?")
_BIBCMD_RE = re.compile(r"\\(bibliography|addbibresource)\s*(?:\[[^\]]*\])?\s*\{")
_COND_RE = re.compile(r"\\(iffalse|iftrue|newif|ifdefined|ifcsname|ifxetex|ifluatex|ifpdf|ifx|ifnum)(?![A-Za-z])")


def profile_paper(paper_dir: str, state: dict) -> dict:
    job = discover_paper(Path(paper_dir), state, with_sizes=False)
    rec: dict = {"paper": job.paper_key, "n_versions": len(job.versions), "versions": []}
    prev_sig = None
    for vi in job.versions:
        v: dict = {"v": vi.version}
        if not vi.on_disk:
            v["status"] = "missing"
            rec["versions"].append(v)
            prev_sig = None
            continue
        vf = scan_version_dirs([d for d in vi.dirs if d.is_dir()])
        v["dup_dirs"] = len([d for d in vi.dirs if d.is_dir()]) - 1
        tex = vf.tex
        if not tex:
            v["status"] = "empty"
            v["n_bib_only"] = len(vf.bib)
            rec["versions"].append(v)
            prev_sig = None
            continue
        v["status"] = "present"
        v["n_tex"] = len(tex)
        v["n_bib"] = len(vf.bib)
        v["dup_files"] = sum(1 for r in vf.files if split_dup_suffix(r.rsplit("/", 1)[-1])[1] > 0)
        v["big_files"] = sum(1 for r in tex if vf.sizes[r] > 5 * 1024 * 1024)
        enc = Counter()
        roots, classes = [], []
        flags = Counter()
        inc = Counter()
        hashes = []
        dup_equal = 0
        stripped_by_rel = {}
        for r in tex:
            if vf.sizes[r] > 5 * 1024 * 1024:
                continue
            with open(vf.files[r], "rb") as fh:
                txt, e = decode_bytes(fh.read())
            enc[e] += 1
            s = clean_file_text(txt, Literals())
            stripped_by_rel[r] = s
            hashes.append((r, content_hash(s)))
            m = _DOCCLASS_RE.search(s)
            if m:
                roots.append(r)
                classes.append(("style:" if m.group(1) == "style" else "") + m.group(2).lower())
            if _BEGIN_DOC_RE.search(s):
                flags["begin_document"] += 1
            if "\\bibitem" in s:
                flags["bibitem"] += 1
            if "\\bibinfo" in s:
                flags["bibinfo"] += 1
            if "\\newblock" in s:
                flags["newblock"] += 1
            if _BIBCMD_RE.search(s):
                flags["bib_cmd"] += 1
            if "\\printbibliography" in s:
                flags["biblatex"] += 1
            for mm in _INCLUDE_RE.finditer(s):
                inc[mm.group(1) + ("{}" if mm.group(2) else "_nobrace")] += 1
            if _INCLUDE_RE.search(s):
                for mm in _COND_RE.finditer(s):
                    flags["cond_" + mm.group(1)] += 1
        # "(n)" copies equal to their canonical file after comment stripping
        for r, s in stripped_by_rel.items():
            d, base = (r.rsplit("/", 1) if "/" in r else ("", r))
            c, n = split_dup_suffix(base)
            if n:
                cr = (d + "/" + c) if d else c
                if cr in stripped_by_rel:
                    dup_equal += stripped_by_rel[cr] == s
        sig = content_hash("|".join(f"{r}:{h}" for r, h in sorted(hashes)))
        v.update(roots=roots, classes=classes, enc=dict(enc), flags=dict(flags), inc=dict(inc),
                 dup_equal=dup_equal, same_as_prev=(sig == prev_sig))
        prev_sig = sig
        rec["versions"].append(v)
    return rec


def _work(args):
    try:
        return profile_paper(*args)
    except Exception as exc:  # profiling must not stop on one paper
        return {"paper": Path(args[0]).name, "error": repr(exc), "versions": []}


def summarize(recs: list[dict]) -> dict:
    S: dict = {}
    vers = [v for r in recs for v in r["versions"]]
    st = Counter(v["status"] for v in vers)
    present = [v for v in vers if v["status"] == "present"]
    P = len(present) or 1
    S["papers"] = len(recs)
    S["errors"] = sum(1 for r in recs if "error" in r)
    S["versions"] = len(vers)
    S["status"] = dict(st)
    S["versions_per_paper"] = dict(Counter(r["n_versions"] for r in recs))
    rc = Counter(min(len(v["roots"]), 4) for v in present)
    S["root_candidates"] = {("4+" if k == 4 else str(k)): n for k, n in sorted(rc.items())}
    S["documentclass_top"] = Counter(c for v in present for c in set(v["classes"])).most_common(25)
    S["n_tex_dist"] = {k: n for k, n in sorted(Counter(min(v["n_tex"], 20) for v in present).items())}
    S["multi_file_share"] = sum(v["n_tex"] > 1 for v in present) / P

    def share(flag):
        return sum(1 for v in present if v["flags"].get(flag)) / P
    S["share_bib_file"] = sum(1 for v in present if v["n_bib"]) / P
    for f in ("bib_cmd", "bibitem", "bibinfo", "newblock", "biblatex", "begin_document"):
        S["share_" + f] = share(f)
    S["share_bibitem_no_bib_file"] = sum(1 for v in present if v["flags"].get("bibitem")
                                         and not v["n_bib"]) / P
    inc = Counter()
    for v in present:
        inc.update(v["inc"])
    S["include_commands"] = dict(inc.most_common())
    S["versions_with_include_variant"] = {k: sum(1 for v in present if v["inc"].get(k))
                                          for k in inc}
    cond = Counter()
    for v in present:
        cond.update({k: 1 for k in v["flags"] if k.startswith("cond_")})
    S["versions_with_conditionals_in_including_files"] = dict(cond)
    enc = Counter()
    for v in present:
        enc.update(v["enc"])
    S["file_encodings"] = dict(enc)
    S["versions_non_utf8"] = sum(1 for v in present if set(v["enc"]) - {"utf-8"})
    S["versions_with_dup_files"] = sum(1 for v in present if v["dup_files"])
    S["dup_files_total"] = sum(v["dup_files"] for v in present)
    S["dup_files_equal_after_comment_strip"] = sum(v["dup_equal"] for v in present)
    S["versions_with_dup_dirs"] = sum(1 for v in vers if v.get("dup_dirs"))
    S["big_tex_files"] = sum(v["big_files"] for v in present)
    multi = [v for r in recs for i, v in enumerate(r["versions"]) if i > 0 and v["status"] == "present"]
    S["later_versions_present"] = len(multi)
    S["later_versions_identical_to_prev"] = sum(1 for v in multi if v["same_as_prev"])
    return S


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="data/m2/profile")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    t0 = time.perf_counter()
    state = load_state(data)
    dirs = [str(d) for d in list_paper_dirs(data)][: a.limit]
    with ProcessPoolExecutor(a.workers) as ex:
        recs = list(ex.map(_work, [(d, state) for d in dirs], chunksize=16))
    S = summarize(recs)
    S["elapsed_s"] = round(time.perf_counter() - t0, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "profile_records.jsonl", "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    with open(out / "profile_summary.json", "w", encoding="utf-8") as fh:
        json.dump(S, fh, indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
