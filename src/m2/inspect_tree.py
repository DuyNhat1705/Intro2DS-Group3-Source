r"""Debug helper: print one paper/version tree with its file:line source map.

Usage::

    python -m src.m2.inspect_tree 2608.04009 --version 1 [--data-dir data] [--depth 3] [--width 110]

The paper is parsed afresh (nothing is written). The header shows the gathering result (root,
compilation path, redundant files, missing includes, notes); headings are annotated with the
``file:line`` of their sectioning command in the compilation path.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from .discover import arxiv_id_to_key, discover_paper
from .io_utils import scan_version_dirs
from .pipeline import parse_version_files, read_version_files

_SEC_RE = re.compile(r"\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\*?\s*[\[{]")


def inspect(paper: str, version: int, data_dir: Path, depth: int = 99, width: int = 110,
            out=sys.stdout) -> None:
    pdir = Path(data_dir) / arxiv_id_to_key(paper)
    job = discover_paper(pdir, with_sizes=False)
    vi = next((x for x in job.versions if x.version == version), None)
    if vi is None or not vi.on_disk:
        print(f"{paper} v{version}: missing", file=out)
        return
    vf = scan_version_dirs([d for d in vi.dirs if d.is_dir()])
    files, sizes = read_version_files(vf)
    vp = parse_version_files(files, sizes, meta_title=job.title, version=version)
    g = vp.gather
    print(f"== {job.arxiv_id} v{version}  status={vp.status}", file=out)
    if g is None or vp.tree is None:
        return
    print(f"root: {g.root_file}\ncompile: {g.compile_files}\nredundant: {g.redundant_files}\n"
          f"missing includes: {g.missing_includes}\nbib: {g.bib_files} (fallback={g.bib_fallback})\n"
          f"notes: {g.notes[:15]}\nstats: {vp.stats}", file=out)
    locs = [g.locate(m.start()) for m in _SEC_RE.finditer(g.text) if m.start() >= g.body_offset]
    li = iter(locs)

    def walk(n, d):
        if d > depth:
            return
        loc = ""
        if n.kind in ("part", "chapter", "section", "subsection", "subsubsection", "paragraph",
                      "subparagraph"):
            nxt = next(li, None)
            loc = f"  <{nxt[0]}:{nxt[1]}>" if nxt else ""
        c = (n.content or "").replace("\n", " / ")
        if len(c) > width:
            c = c[:width - 3] + "..."
        print(f"{'  ' * d}[{n.kind}] {c}{loc}", file=out)
        for ch in n.children:
            walk(ch, d + 1)
    walk(vp.tree, 0)
    print(f"bibtex entries (cited): {len(vp.bib_entries)}  raw .bib: {vp.n_bib_raw}  bibitem: {vp.n_bibitem}",
          file=out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paper")
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--depth", type=int, default=99)
    ap.add_argument("--width", type=int, default=110)
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    inspect(a.paper, a.version, Path(a.data_dir), a.depth, a.width)


if __name__ == "__main__":
    main()
