r"""Gold-set evaluation of the M2 parser: prints every accuracy metric of prompt §7.

Usage (from the repository root)::

    python -m src.m2.tests.eval_gold [--data-dir data] [--sat] [--refine] [--json out.json]

* structure: root / compilation path exact match, heading-outline accuracy (LCS of
  ``(kind, normalized title)`` divided by ``max(len(gold), len(pred))``, micro-averaged);
* sentences: precision / recall / F1 of exact sentence matches (after normalization), with the
  gold pieces cleaned by the same cleanup so that only boundaries are measured; ``--sat`` also
  runs the SaT ``sat-3l-sm`` A/B (needs ``wtpsplit``; the model is downloaded once);
* bibitems: title / year / first-author accuracy; ``--refine`` also runs the optional AnyStyle
  refinement when the CLI is installed;
* reference pairs: precision / recall of :func:`same_reference`, plus a fuzzy-threshold sweep.

The structure gold needs the local corpus (``data/``); the other three sets are self-contained.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

from rapidfuzz import fuzz

from .gold_bibitems import GOLD_BIBITEMS, SKIP
from .gold_pairs import GOLD_PAIRS
from .gold_sentences import GOLD_SENTENCES, MARK
from .gold_structure import GOLD_STRUCTURE

HEADING_KINDS = {"part", "chapter", "section", "subsection", "subsubsection", "paragraph",
                 "subparagraph", "abstract"}


def norm(t: str | None) -> str:
    if not t:
        return ""
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c)).lower()
    return re.sub(r"[^0-9a-z]+", " ", t).strip()


def lcs(a: list, b: list) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------------------
def eval_structure(data_dir: Path, verbose: bool = False) -> dict:
    from ..discover import discover_paper
    from ..io_utils import scan_version_dirs
    from ..pipeline import parse_version_files, read_version_files

    n = root_ok = comp_ok = 0
    h_match = h_total = 0
    per = []
    for paper, v, g_root, g_comp, g_outline in GOLD_STRUCTURE:
        pdir = data_dir / paper
        if not pdir.is_dir():
            continue
        job = discover_paper(pdir, with_sizes=False)
        vi = next(x for x in job.versions if x.version == v)
        vf = scan_version_dirs([d for d in vi.dirs if d.is_dir()])
        files, sizes = read_version_files(vf)
        vp = parse_version_files(files, sizes=sizes, meta_title=job.title)
        n += 1
        r_ok = vp.gather.root_file == g_root
        c_ok = sorted(vp.gather.compile_files) == sorted(g_comp)
        root_ok += r_ok
        comp_ok += c_ok
        pred = []

        def walk(node):
            if node.kind in HEADING_KINDS:
                pred.append((node.kind, "" if node.kind == "abstract" else norm(node.content)))
            for c in node.children:
                walk(c)
        walk(vp.tree)
        gold = [(k, "" if k == "abstract" else norm(t)) for k, t in g_outline]
        m = lcs(gold, pred)
        h_match += m
        h_total += max(len(gold), len(pred))
        per.append((paper, v, r_ok, c_ok, m, len(gold), len(pred)))
        if verbose and (not r_ok or not c_ok or m < max(len(gold), len(pred))):
            print(f"  [struct] {paper} v{v}: root {vp.gather.root_file!r} (gold {g_root!r}) "
                  f"compile_ok={c_ok} headings {m}/{len(gold)} pred={len(pred)}")
            if not c_ok:
                print("     pred compile:", vp.gather.compile_files, "\n     gold compile:", g_comp)
            gs, ps = set(gold), set(pred)
            for x in [g for g in gold if g not in ps][:6]:
                print("     missing:", x)
            for x in [p for p in pred if p not in gs][:6]:
                print("     extra:  ", x)
    return {"versions": n, "root_acc": root_ok / max(n, 1), "compile_acc": comp_ok / max(n, 1),
            "heading_acc": h_match / max(h_total, 1), "per_version": per}


# ---------------------------------------------------------------------------------------
def _gold_and_pred(splitter):
    from ..pipeline import clean_paragraph_text

    golds, preds = [], []
    for _paper, marked in GOLD_SENTENCES:
        pieces = [p for p in marked.split(MARK)]
        gold = [norm(clean_paragraph_text(p)) for p in pieces]
        gold = [g for g in gold if g]
        raw = marked.replace(MARK, " ")
        pred = [norm(s) for s in splitter(clean_paragraph_text(raw))]
        pred = [p for p in pred if p]
        golds.append(gold)
        preds.append(pred)
    return golds, preds


def _prf(golds, preds, verbose=False, label=""):
    tp = fp = fn = 0
    for g, p in zip(golds, preds):
        cg, cp = Counter(g), Counter(p)
        t = sum((cg & cp).values())
        tp += t
        fp += sum(cp.values()) - t
        fn += sum(cg.values()) - t
        if verbose and (cg != cp):
            print(f"  [{label}] mismatch:")
            for x in (cg - cp):
                print("     gold:", x[:150])
            for x in (cp - cg):
                print("     pred:", x[:150])
    pr = tp / max(tp + fp, 1)
    rc = tp / max(tp + fn, 1)
    return {"precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9),
            "n_gold": tp + fn}


def eval_sentences(verbose=False, sat=False) -> dict:
    from ..sentences import split_sentences

    t0 = time.perf_counter()
    golds, preds = _gold_and_pred(split_sentences)
    out = {"rules": _prf(golds, preds, verbose, "rules")}
    out["rules"]["seconds"] = time.perf_counter() - t0
    if sat:
        try:
            from wtpsplit import SaT  # optional, A/B only
            model = SaT("sat-3l-sm", ort_providers=["CPUExecutionProvider"])
            t1 = time.perf_counter()
            g2, p2 = _gold_and_pred(lambda text: [s.strip() for s in model.split(text) if s.strip()])
            out["sat"] = _prf(g2, p2, verbose, "sat")
            out["sat"]["seconds"] = time.perf_counter() - t1
        except Exception as exc:  # SaT is optional
            out["sat"] = {"error": repr(exc)}
    return out


# ---------------------------------------------------------------------------------------
def _surname(name: str) -> str:
    toks = norm(name).split()
    return toks[-1] if toks else ""


def eval_bibitems(verbose=False, refine=False) -> dict:
    from ..bibitem import parse_bibitems

    def score(get_entry):
        t_ok = t_n = y_ok = y_n = a_ok = a_n = 0
        methods = Counter()
        for paper, style, raw, title, first, n_auth, year in GOLD_BIBITEMS:
            e = get_entry(raw)
            methods[getattr(e, "parse_method", "none")] += 1
            if title != SKIP:
                t_n += 1
                pt = norm(e.title) if e else ""
                if title is None:
                    ok = not pt
                else:
                    ok = bool(pt) and (pt == norm(title) or fuzz.ratio(pt, norm(title)) >= 95)
                t_ok += ok
                if verbose and not ok:
                    print(f"  [bib title] {paper}: gold={title!r}\n               pred={(e.title if e else None)!r}")
            if year != SKIP:
                y_n += 1
                ok = (e.year if e else None) == year
                y_ok += ok
                if verbose and not ok:
                    print(f"  [bib year] {paper}: gold={year} pred={(e.year if e else None)} :: {raw[:100]!r}")
            if first != SKIP:
                a_n += 1
                pa = e.authors if e else []
                ok = bool(pa) and _surname(pa[0]) == _surname(first) and (n_auth == SKIP or len(pa) == n_auth)
                a_ok += ok
                if verbose and not ok:
                    print(f"  [bib auth] {paper}: gold={first}/{n_auth} pred={pa[:3]}..{len(pa)}")
        return {"title_acc": t_ok / max(t_n, 1), "year_acc": y_ok / max(y_n, 1),
                "author_acc": a_ok / max(a_n, 1), "n": len(GOLD_BIBITEMS), "methods": dict(methods)}

    def first_entry(raw):
        ents = parse_bibitems(raw)
        return ents[0] if ents else None

    out = {"plain": score(first_entry)}
    if refine:
        from ..bib_refine import anystyle_available, refine_entries
        if anystyle_available():
            ents = [first_entry(r[2]) for r in GOLD_BIBITEMS]
            refined = refine_entries([e for e in ents if e is not None])
            it = iter(refined)
            table = {id(e): next(it) for e in ents if e is not None}
            out["refined"] = score(lambda raw, _c={r[2]: e for r, e in zip(GOLD_BIBITEMS, ents)}:
                                   table.get(id(_c[raw]), _c[raw]) if _c[raw] is not None else None)
        else:
            out["refined"] = {"error": "AnyStyle CLI not installed"}
    return out


# ---------------------------------------------------------------------------------------
def eval_pairs(verbose=False) -> dict:
    from .. import bib_dedup
    from ..bib_parse import parse_bib

    def run(threshold=None):
        tp = fp = fn = tn = 0
        for paper, va, a, vb, b, same in GOLD_PAIRS:
            ea, eb = parse_bib(a)[0], parse_bib(b)[0]
            if not ea or not eb:
                pred = False
            else:
                kw = {} if threshold is None else {"threshold": threshold}
                pred = bib_dedup.same_reference(ea[0], eb[0], **kw)
            if pred and same:
                tp += 1
            elif pred and not same:
                fp += 1
                if verbose and threshold is None:
                    print(f"  [pair FP] {paper}: {ea[0].title!r} | {eb[0].title!r}")
            elif not pred and same:
                fn += 1
                if verbose and threshold is None:
                    print(f"  [pair FN] {paper}: {ea[0].title!r} ({ea[0].year},{ea[0].doi}) | "
                          f"{eb[0].title!r} ({eb[0].year},{eb[0].doi})")
            else:
                tn += 1
        return {"precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    out = {"default": run()}
    out["sweep"] = {t: run(t) for t in (90, 92, 93, 94, 95, 97)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--sat", action="store_true")
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--only", choices=["structure", "sentences", "bibitems", "pairs"])
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {}
    if a.only in (None, "structure"):
        res["structure"] = eval_structure(Path(a.data_dir), a.verbose)
        s = res["structure"]
        print(f"STRUCTURE  versions={s['versions']}  root={s['root_acc']:.3f}  "
              f"compile_path={s['compile_acc']:.3f}  headings={s['heading_acc']:.3f}  "
              f"(targets 1.000 / 1.000 / >=0.950)")
    if a.only in (None, "sentences"):
        res["sentences"] = eval_sentences(a.verbose, a.sat)
        for k, v in res["sentences"].items():
            if "f1" in v:
                print(f"SENTENCES[{k}]  P={v['precision']:.4f}  R={v['recall']:.4f}  F1={v['f1']:.4f}  "
                      f"n={v['n_gold']}  time={v['seconds']:.2f}s  (target F1>=0.97)")
            else:
                print(f"SENTENCES[{k}]  {v}")
    if a.only in (None, "bibitems"):
        res["bibitems"] = eval_bibitems(a.verbose, a.refine)
        for k, v in res["bibitems"].items():
            if "title_acc" in v:
                print(f"BIBITEM[{k}]  title={v['title_acc']:.3f}  year={v['year_acc']:.3f}  "
                      f"first_author={v['author_acc']:.3f}  n={v['n']}  methods={v['methods']}  "
                      f"(targets title>=0.90, year>=0.95)")
            else:
                print(f"BIBITEM[{k}]  {v}")
    if a.only in (None, "pairs"):
        res["pairs"] = eval_pairs(a.verbose)
        d = res["pairs"]["default"]
        print(f"REF-DEDUP  P={d['precision']:.3f}  R={d['recall']:.3f}  tp={d['tp']} fp={d['fp']} "
              f"fn={d['fn']} tn={d['tn']}  (targets P>=0.98, R>=0.95)")
        for t, v in res["pairs"]["sweep"].items():
            print(f"   fuzzy threshold {t}: P={v['precision']:.3f} R={v['recall']:.3f}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
