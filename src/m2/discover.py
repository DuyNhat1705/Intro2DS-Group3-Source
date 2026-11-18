r"""Discovery of papers and versions from ``data/``, ``metadata.json`` and the M1 download state,
including versions that are ``empty`` (folder without TeX) or ``missing`` (listed but not on disk).

Only M1 *metadata* is read here (``metadata.json`` and ``data/.state/*/source_state*.json``).
``references.json`` is never opened (Milestone 3 must stay leakage free).
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .io_utils import parse_version_dir

# Folders of data/ that are not papers.
_NON_PAPER_DIRS = {"tables", "run_logs", "report_exports", "m2", ".state"}
# New-style keys "2608-04009" and old-style keys "hep-th-9901001".
_NEW_KEY_RE = re.compile(r"^\d{4}-\d{4,5}$")
_OLD_KEY_RE = re.compile(r"^(?P<arch>[a-z\-]+(?:-[A-Z]{2})?)-(?P<num>\d{7})$")


def key_to_arxiv_id(key: str) -> str:
    """Best-effort inverse of the M1 key normalization (used when metadata is missing)."""
    if _NEW_KEY_RE.match(key):
        a, b = key.split("-")
        return f"{a}.{b}"
    m = _OLD_KEY_RE.match(key)
    if m:
        return f"{m.group('arch')}/{m.group('num')}"
    return key


def arxiv_id_to_key(arxiv_id: str) -> str:
    return arxiv_id.replace(".", "-").replace("/", "-")


@dataclass
class VersionInfo:
    version: int
    dirs: list[Path] = field(default_factory=list)  # primary first, then "(k)" copies
    in_metadata: bool = False
    state_status: str | None = None                 # M1 state status, e.g. done / no_tex

    @property
    def on_disk(self) -> bool:
        return any(d.is_dir() for d in self.dirs)


@dataclass
class PaperJob:
    paper_key: str
    arxiv_id: str
    paper_dir: Path
    title: str = ""
    abstract: str = ""
    num_versions: int = 0
    versions: list[VersionInfo] = field(default_factory=list)
    tex_bytes: int = 0                               # for LPT scheduling
    notes: list[str] = field(default_factory=list)


def load_state(data_dir: Path) -> dict[str, dict[int, str]]:
    """Read every ``source_state*.json`` of the M1 workers: arxiv_id -> {version: status}."""
    out: dict[str, dict[int, str]] = {}
    state_root = data_dir / ".state"
    if not state_root.is_dir():
        return out
    for worker in sorted(state_root.iterdir()):
        if not worker.is_dir():
            continue
        for f in sorted(worker.glob("source_state*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
            except (OSError, ValueError):
                continue
            for aid, rec in (d.get("papers") or {}).items():
                vs = out.setdefault(aid, {})
                for v, vrec in (rec.get("versions") or {}).items():
                    try:
                        vi = int(v)
                    except ValueError:
                        continue
                    st = (vrec or {}).get("status")
                    # "done" wins over anything else seen in a duplicate state file.
                    if vs.get(vi) != "done":
                        vs[vi] = st
    return out


def _load_metadata(paper_dir: Path) -> dict:
    p = paper_dir / "metadata.json"
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _tex_bytes(dirs: list[Path]) -> int:
    total = 0
    for d in dirs:
        for dirpath, _dn, fns in os.walk(d):
            for fn in fns:
                if fn.lower().endswith((".tex", ".bib")):
                    try:
                        total += os.path.getsize(os.path.join(dirpath, fn))
                    except OSError:
                        pass
    return total


def list_paper_dirs(data_dir: Path) -> list[Path]:
    out = []
    with os.scandir(data_dir) as it:
        for e in it:
            if not e.is_dir() or e.name.startswith(".") or e.name in _NON_PAPER_DIRS:
                continue
            out.append(Path(e.path))
    out.sort(key=lambda p: p.name)
    return out


def discover_paper(paper_dir: Path, state: dict[str, dict[int, str]] | None = None,
                   with_sizes: bool = True) -> PaperJob:
    meta = _load_metadata(paper_dir)
    key = paper_dir.name
    aid = meta.get("arxiv_id") or key_to_arxiv_id(key)
    job = PaperJob(paper_key=key, arxiv_id=aid, paper_dir=paper_dir,
                   title=meta.get("title") or "", abstract=meta.get("abstract") or "")
    try:
        nv = int(meta.get("num_versions") or 0)
    except (TypeError, ValueError):
        nv = 0
    job.num_versions = nv
    if not meta:
        job.notes.append("no_metadata")

    vers: dict[int, VersionInfo] = {}
    for v in range(1, nv + 1):
        vers[v] = VersionInfo(version=v, in_metadata=True)
    st = (state or {}).get(aid, {})
    for v, s in st.items():
        vers.setdefault(v, VersionInfo(version=v)).state_status = s

    # Version folders on disk: vN, vN(k).
    found: dict[int, list[tuple[int, Path]]] = {}
    try:
        with os.scandir(paper_dir) as it:
            for e in it:
                if e.is_dir():
                    pv = parse_version_dir(e.name)
                    if pv:
                        found.setdefault(pv[0], []).append((pv[1], Path(e.path)))
    except OSError:
        pass
    for v, lst in found.items():
        lst.sort()
        vi = vers.setdefault(v, VersionInfo(version=v))
        vi.dirs = [p for _k, p in lst]
        if any(k > 0 for k, _p in lst):
            job.notes.append(f"dup_version_dir:v{v}:{len(lst)}")
    for v, vi in vers.items():
        if not vi.dirs:
            vi.dirs = [paper_dir / f"v{v}"]
    job.versions = [vers[v] for v in sorted(vers)]
    job.num_versions = max(nv, max(vers) if vers else 0)
    if with_sizes:
        job.tex_bytes = sum(_tex_bytes([d for d in vi.dirs if d.is_dir()]) for vi in job.versions)
    return job


def discover(data_dir: Path, papers: list[str] | None = None, limit: int | None = None,
             with_sizes: bool = True) -> list[PaperJob]:
    """Discover every paper of ``data_dir`` (optionally restricted to arXiv IDs / keys)."""
    data_dir = Path(data_dir)
    state = load_state(data_dir)
    dirs = list_paper_dirs(data_dir)
    if papers:
        wanted = {arxiv_id_to_key(p) for p in papers} | set(papers)
        dirs = [d for d in dirs if d.name in wanted]
    if limit is not None:
        dirs = dirs[:limit]
    return [discover_paper(d, state, with_sizes=with_sizes) for d in dirs]
