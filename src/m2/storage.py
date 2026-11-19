r"""Per-worker Parquet shards, manifest, checkpoint signatures and compaction.

Layout::

    <out>/elements/  <out>/hierarchy/  <out>/bibtex/                published tables (exact schema)
    <work>/parse_status/ <work>/bibtex_full/ <work>/cite_key_map/  internal working data
    <work>/paper_stats/                                            per-paper counters
    <work>/manifest/<run>-w<id>.jsonl                              one line per finished paper

Each worker buffers the rows of finished papers and flushes every ~200 papers (or when the
buffer exceeds a size limit) into ``part-<run>-w<id>-<seq>.parquet`` of every table: the file
is written under a temporary name and renamed (atomic on one file system), then the manifest
line of each paper of the shard is appended. A paper's rows therefore always live in exactly one
shard per table, and all tables of one flush share the shard name.

Checkpoint signature = hash(relative file paths + sizes + content hashes) + ``PARSER_VERSION``;
``--resume`` skips papers whose signature is unchanged. ``compact`` keeps only the latest shard
of every paper (re-runs leave older copies behind), sorts rows by ``arxiv_id`` (stable, so the
pre-order of hierarchy rows is preserved) and writes a fixed number of deterministic shards.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from . import config as C

_DICT = pa.dictionary(pa.int32(), pa.string())

SCHEMAS: dict[str, pa.Schema] = {
    "elements": pa.schema([
        pa.field("element_id", pa.string(), nullable=False),
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("kind", _DICT),
        pa.field("content", pa.string()),
    ]),
    "hierarchy": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("version", pa.int32()),
        pa.field("element_id", pa.string(), nullable=False),
        pa.field("parent_id", pa.string(), nullable=False),
    ]),
    "bibtex": pa.schema([
        pa.field("arxiv_id", pa.string(), nullable=False),
        pa.field("citation_key", pa.string(), nullable=False),
        pa.field("entry_type", _DICT),
        pa.field("title", pa.string()),
        pa.field("authors", pa.list_(pa.string())),
        pa.field("year", pa.int32(), nullable=True),
        pa.field("doi", pa.string()),
        pa.field("journal", pa.string()),
        pa.field("url", pa.string()),
    ]),
}
SCHEMAS["bibtex_full"] = pa.schema(list(SCHEMAS["bibtex"]) + [
    pa.field("eprint", pa.string()), pa.field("booktitle", pa.string()),
    pa.field("volume", pa.string()), pa.field("pages", pa.string()),
    pa.field("publisher", pa.string()), pa.field("source", _DICT),
    pa.field("parse_method", _DICT), pa.field("confidence", pa.float32()),
    pa.field("versions", pa.list_(pa.int32())), pa.field("merged_keys", pa.list_(pa.string())),
])
SCHEMAS["cite_key_map"] = pa.schema([
    pa.field("arxiv_id", pa.string(), nullable=False), pa.field("version", pa.int32()),
    pa.field("original_key", pa.string(), nullable=False),
    pa.field("canonical_key", pa.string(), nullable=False),
])
_STATUS_INT = ("n_compile_files", "n_redundant_files", "n_missing_includes", "n_nodes",
               "n_sentences", "n_formulas", "n_figures", "n_tables", "max_depth", "n_lists",
               "n_blocks", "n_verbatim", "n_section_cmds", "n_section_nodes",
               "n_unbalanced_formulas", "n_bib_raw", "n_bibitem", "n_cited")
_STATUS_FLOAT = ("t_gather_ms", "t_clean_ms", "t_hierarchy_ms", "t_bib_ms", "t_dedup_ms", "elapsed_ms")
SCHEMAS["parse_status"] = pa.schema(
    [pa.field("arxiv_id", pa.string(), nullable=False), pa.field("version", pa.int32()),
     pa.field("status", _DICT), pa.field("failure_reason", pa.string()),
     pa.field("root_file", pa.string())]
    + [pa.field(c, pa.int32()) for c in _STATUS_INT]
    + [pa.field("bib_source", _DICT), pa.field("bib_fallback", pa.bool_()),
       pa.field("reused_from_prev", pa.bool_())]
    + [pa.field(c, pa.float32()) for c in _STATUS_FLOAT]
    + [pa.field("retry_mode", pa.string()), pa.field("coverage", pa.float32()),
       pa.field("notes", pa.string())])
SCHEMAS["paper_stats"] = pa.schema([
    pa.field("arxiv_id", pa.string(), nullable=False), pa.field("paper_key", pa.string()),
    pa.field("n_versions", pa.int32()), pa.field("nodes_before", pa.int64()),
    pa.field("elements_after", pa.int64()), pa.field("bib_raw", pa.int64()),
    pa.field("bib_before", pa.int64()), pa.field("bib_after", pa.int64()),
    pa.field("reused_versions", pa.int32()), pa.field("leaf_repeats_within_version", pa.int64()),
    pa.field("sentences_distinct", pa.int64()), pa.field("sentences_residual_latex", pa.int64()),
    pa.field("elapsed_ms", pa.float32()), pa.field("tex_bytes", pa.int64()),
    pa.field("error", pa.string()),
])

PUBLISHED = ("elements", "hierarchy", "bibtex")
INTERNAL = ("bibtex_full", "cite_key_map", "parse_status", "paper_stats")
ALL_TABLES = PUBLISHED + INTERNAL


def table_dir(name: str, out_dir: Path, work_dir: Path) -> Path:
    return (out_dir if name in PUBLISHED else work_dir) / name


def to_table(name: str, rows: list[dict]) -> pa.Table:
    schema = SCHEMAS[name]
    cols = {}
    for f in schema:
        vals = [r.get(f.name) for r in rows]
        if pa.types.is_dictionary(f.type):
            cols[f.name] = pa.array(vals, pa.string()).dictionary_encode()
            if cols[f.name].type != f.type:
                cols[f.name] = cols[f.name].cast(f.type)
        else:
            cols[f.name] = pa.array(vals, f.type)
    return pa.Table.from_pydict(cols, schema=schema)


def _conform(tab: pa.Table, schema: pa.Schema) -> pa.Table:
    """Cast a shard to ``schema``; columns added since the shard was written become nulls."""
    if tab.schema.names != schema.names:
        tab = pa.table({f.name: tab[f.name] if f.name in tab.schema.names
                        else pa.nulls(tab.num_rows, f.type) for f in schema})
    return tab.cast(schema)


def _write_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
    os.replace(tmp, path)


class ShardWriter:
    """Buffers the rows of finished papers of one worker and writes shards."""

    def __init__(self, out_dir: Path, work_dir: Path, run_id: str, worker_id: int,
                 flush_papers: int = C.SHARD_FLUSH_PAPERS, flush_bytes: int = C.SHARD_FLUSH_BYTES):
        self.out_dir, self.work_dir = Path(out_dir), Path(work_dir)
        self.run_id, self.worker_id = run_id, worker_id
        self.flush_papers, self.flush_bytes = flush_papers, flush_bytes
        self.seq = 0
        self.rows: dict[str, list[dict]] = {t: [] for t in ALL_TABLES}
        self.pending: list[dict] = []   # manifest records of buffered papers
        self.bytes = 0
        self.bytes_written = 0

    def add(self, res, signature: str, n_versions: int, tex_bytes: int, elapsed_ms: float) -> None:
        self.rows["elements"].extend(res.elements)
        self.rows["hierarchy"].extend(res.hierarchy)
        self.rows["bibtex"].extend(res.bibtex)
        self.rows["bibtex_full"].extend(res.bibtex_full)
        self.rows["cite_key_map"].extend(res.key_map)
        self.rows["parse_status"].extend(res.status)
        st = dict(res.stats)
        st.pop("traceback", None)
        self.rows["paper_stats"].append({
            "arxiv_id": res.arxiv_id, "paper_key": res.paper_key, "n_versions": n_versions,
            "elapsed_ms": elapsed_ms, "tex_bytes": tex_bytes, "error": res.error or None,
            **{k: st.get(k, 0) for k in ("nodes_before", "elements_after", "bib_raw", "bib_before",
                                         "bib_after", "reused_versions", "leaf_repeats_within_version",
                                         "sentences_distinct", "sentences_residual_latex")}})
        self.pending.append({"paper_key": res.paper_key, "arxiv_id": res.arxiv_id,
                             "signature": signature, "parser_version": C.PARSER_VERSION})
        self.bytes += sum(len(e.get("content") or "") + 80 for e in res.elements) + 60 * len(res.hierarchy)

    def should_flush(self) -> bool:
        return len(self.pending) >= self.flush_papers or self.bytes >= self.flush_bytes

    def flush(self) -> list[str]:
        if not self.pending:
            return []
        name = f"part-{self.run_id}-w{self.worker_id:02d}-{self.seq:05d}.parquet"
        for t in ALL_TABLES:
            tab = to_table(t, self.rows[t])
            p = table_dir(t, self.out_dir, self.work_dir) / name
            _write_atomic(tab, p)
            self.bytes_written += p.stat().st_size
        mdir = self.work_dir / "manifest"
        mdir.mkdir(parents=True, exist_ok=True)
        with open(mdir / f"{self.run_id}-w{self.worker_id:02d}.jsonl", "a", encoding="utf-8") as fh:
            for rec in self.pending:
                rec = dict(rec, shard=name, t=round(time.time(), 3))
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done = [r["paper_key"] for r in self.pending]
        self.seq += 1
        self.rows = {t: [] for t in ALL_TABLES}
        self.pending = []
        self.bytes = 0
        return done


def load_manifest(work_dir: Path) -> dict[str, dict]:
    """paper_key -> latest manifest record (by time)."""
    out: dict[str, dict] = {}
    mdir = Path(work_dir) / "manifest"
    if not mdir.is_dir():
        return out
    for f in sorted(mdir.glob("*.jsonl")):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # torn last line of a killed worker
                old = out.get(r["paper_key"])
                if old is None or r.get("t", 0) >= old.get("t", 0):
                    out[r["paper_key"]] = r
    return out


def _shard_files(d: Path) -> list[Path]:
    return sorted(p for p in d.glob("*.parquet") if not p.name.endswith(".tmp"))


def compact(out_dir: Path, work_dir: Path, n_shards: int = 8, log=print) -> dict:
    """Merge shards into ``n_shards`` deterministic files sorted by ``arxiv_id``."""
    out_dir, work_dir = Path(out_dir), Path(work_dir)
    man = load_manifest(work_dir)
    latest_shard = {r["arxiv_id"]: r["shard"] for r in man.values()}
    ids = sorted(latest_shard)
    n_shards = max(1, min(n_shards, len(ids) or 1))
    buckets = [ids[i * len(ids) // n_shards:(i + 1) * len(ids) // n_shards] for i in range(n_shards)]
    summary = {}
    for t in ALL_TABLES:
        d = table_dir(t, out_dir, work_dir)
        files = [f for f in _shard_files(d) if f.name.startswith("part-")]
        if not files:
            continue
        keep_by_file: dict[str, list[str]] = {}
        for aid, shard in latest_shard.items():
            keep_by_file.setdefault(shard, []).append(aid)
        new_files = []
        total = 0
        for bi, bucket in enumerate(buckets):
            bset = pa.array(bucket, pa.string())
            parts = []
            for f in files:
                keep = keep_by_file.get("compacted" if f.name.startswith("part-c") else f.name)
                if not keep:
                    continue
                tab = pq.read_table(f)
                mask = pc.and_(pc.is_in(tab["arxiv_id"], pa.array(keep, pa.string())),
                               pc.is_in(tab["arxiv_id"], bset))
                tab = tab.filter(mask)
                if tab.num_rows:
                    parts.append(tab)
            if not parts:
                continue
            tab = pa.concat_tables([_conform(p, SCHEMAS[t]) for p in parts]).unify_dictionaries()
            tab = tab.append_column("__row", pa.array(range(tab.num_rows), pa.int64()))
            tab = tab.sort_by([("arxiv_id", "ascending"), ("__row", "ascending")]).drop_columns(["__row"])
            p = d / f"new-c{bi:05d}.parquet"
            _write_atomic(tab.combine_chunks(), p)
            new_files.append(p)
            total += tab.num_rows
        for f in files:
            f.unlink()
        final = []
        for p in new_files:
            q = p.with_name(p.name.replace("new-c", "part-c"))
            os.replace(p, q)
            final.append(q)
        summary[t] = {"rows": total, "files": len(final), "bytes": sum(q.stat().st_size for q in final)}
        log(f"compact {t}: {total} rows -> {len(final)} files")
    mdir = work_dir / "manifest"
    if mdir.is_dir():
        recs = [dict(r, shard="compacted") for r in sorted(man.values(), key=lambda r: r["arxiv_id"])]
        tmp = mdir / "compacted.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        for f in mdir.glob("*.jsonl"):
            f.unlink()
        os.replace(tmp, mdir / "compacted.jsonl")
    return summary


def dir_size(p: Path) -> int:
    total = 0
    for root, _d, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total
