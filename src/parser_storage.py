"""Atomic per-paper output shared by inline parsing and spawned workers."""
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

OUTPUT_FILES = ("elements.parquet", "hierarchy.parquet", "bibtex.parquet",
                "key_map.json", "bibtex_debug.jsonl", "warnings.jsonl")


def table_schemas():
    """Explicit schema, even when a table has no rows."""
    text = pa.string()
    kind = pa.dictionary(pa.int32(), text)
    return {
        "elements": pa.schema([pa.field("element_id", text, False), pa.field("arxiv_id", text, False),
                               pa.field("kind", kind), pa.field("content", text)]),
        "hierarchy": pa.schema([pa.field("arxiv_id", text, False), pa.field("version", pa.int32()),
                                pa.field("element_id", text, False), pa.field("parent_id", text, False)]),
        "bibtex": pa.schema([pa.field("arxiv_id", text, False), pa.field("citation_key", text, False),
                             pa.field("entry_type", kind), pa.field("title", text),
                             pa.field("authors", pa.list_(text)), pa.field("year", pa.int32()),
                             pa.field("doi", text), pa.field("journal", text), pa.field("url", text)]),
    }


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def paper_name(arxiv_id):
    return re.sub(r"[^A-Za-z0-9_-]", "-", arxiv_id) + "-" + hashlib.sha256(arxiv_id.encode()).hexdigest()[:8]


def file_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_manifest(partition):
    result = {}
    for name in OUTPUT_FILES:
        info = (partition / name).stat()
        result[name] = {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
    return result


def write_tables(partition, result, schemas=None, *, consume=False, batch_size=8192):
    """Limit Arrow conversion to one batch; optionally release Python rows as written."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    partition = Path(partition)
    partition.mkdir(parents=True, exist_ok=True)
    for name, schema in (table_schemas() if schemas is None else schemas).items():
        rows = result.pop(name) if consume else result[name]
        temporary = partition / f"{name}.parquet.tmp"
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer:
            for start in range(0, len(rows), batch_size):
                writer.write_table(pa.Table.from_pylist(rows[start:start + batch_size], schema=schema))
        os.replace(temporary, partition / f"{name}.parquet")
        del rows
    atomic_json(partition / "key_map.json", result.pop("key_map", {}) if consume else result.get("key_map", {}))
    for name in ("bibtex_debug", "warnings"):
        rows = result.pop(name, []) if consume else result.get(name, [])
        temporary = partition / f"{name}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, partition / f"{name}.jsonl")
        del rows


def persist_result(partition, result, *, debug_reports=False, consume=False):
    """Return only checkpoint metadata after all six partition files are complete."""
    started = time.perf_counter()
    partition = Path(partition)
    schemas = table_schemas()
    counts = {name: len(result[name]) for name in schemas}
    stats = result["stats"]
    if debug_reports:
        for stage in ("input", "gather", "cleanup", "construct_hierarchy", "extract_bibtex",
                      "unify_publication", "rewrite_citations", "build_element_tables"):
            timings = [row for row in result.get("timings", []) if row["stage"] == stage]
            stats[f"{stage}_sec"] = sum(row["duration_sec"] for row in timings)
            statuses = {row["status"] for row in timings}
            stats[f"{stage}_status"] = next((value for value in
                ("error", "missing_dir", "not_found", "partial", "ok", "empty") if value in statuses), "not_run")
    write_tables(partition, result, schemas, consume=consume)
    manifest = output_manifest(partition)
    stats["output_table_bytes"] = sum(manifest[f"{name}.parquet"]["size"] for name in schemas)
    receipt = {"_partition_receipt": True, "partition": str(partition), "stats": stats,
            "version_status": result["version_status"], "table_counts": counts,
            "output_manifest": manifest,
            "file_signatures": {name: file_signature(partition / name) for name in OUTPUT_FILES},
            "timings": result.get("timings", []) if debug_reports else []}
    receipt["write_sec"] = time.perf_counter() - started
    return receipt


def validate_receipt(receipt, partition, arxiv_id):
    """The parent alone publishes a checkpoint, after checking the worker's files."""
    if (receipt.get("_partition_receipt") is not True
            or Path(receipt["partition"]).resolve() != Path(partition).resolve()
            or receipt["stats"]["arxiv_id"] != arxiv_id
            or set(receipt["table_counts"]) != set(table_schemas())
            or set(receipt["file_signatures"]) != set(OUTPUT_FILES)
            or output_manifest(Path(partition)) != receipt["output_manifest"]):
        raise ValueError("Worker returned an incomplete or mismatched partition receipt")
