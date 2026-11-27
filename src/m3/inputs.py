"""Read the published M1/M2 tables and scraper checkpoints."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from matching import arxiv_id, normalize

# Inclusive ranges from scraper.ipynb and parser.ipynb.
RANGES = {
    1: ("2609", 27834, 32833),
    2: ("2609", 17834, 22833),
    3: ("2608", 14007, 19006),
    4: ("2608", 4007, 9006),
    5: ("2511", 6709, 11708),
}
BIB_FIELDS = ("title",)
REF_FIELDS = ("ref_title",)


def range_of(ident):
    prefix, separator, number = ident.partition(".")
    if not separator or not number.isdecimal():
        return None
    for label, (month, start, end) in RANGES.items():
        if prefix == month and start <= int(number) <= end:
            return label
    return None


def parquet_rows(paths, table, required, optional=()):
    import pyarrow.parquet as pq

    files = set()
    for value in paths:
        path = Path(value)
        if path.is_dir():
            if table == "bibtex":
                found = sorted(path.glob("part-c*.parquet"))
                if not found:
                    found = sorted(path.glob("part-*-of-*.parquet"))
            else:
                found = [p for p in [path / f"{table}.parquet"] if p.is_file()]
        else:
            found = [path] if path.is_file() else []
        if not found:
            raise ValueError(f"Thiếu bảng {table}: {path}")
        files.update(p.resolve() for p in found)
    for path in sorted(files):
        with pq.ParquetFile(path) as parquet:
            columns = set(parquet.schema_arrow.names)
            missing = set(required) - columns
            if missing:
                raise ValueError(f"{table} ({path}) thiếu cột: {', '.join(sorted(missing))}")
            projection = list(required) + [name for name in optional if name in columns]
            for batch in parquet.iter_batches(columns=projection):
                for row in batch.to_pylist():
                    yield row, path


def csv_rows(path, fields):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} thiếu cột: {', '.join(sorted(missing))}")
        yield from reader


def paper_ids(paths):
    corpus = set()
    for path in paths:
        if Path(path).suffix.lower() == ".csv":
            corpus.update(arxiv_id(row["arxiv_id"]) for row in csv_rows(path, ("arxiv_id",)))
        else:
            corpus.update(arxiv_id(row["arxiv_id"]) for row, _ in
                          parquet_rows([path], "papers", ("arxiv_id",)))
    return corpus


def read_references(paths, scope):
    candidates = defaultdict(lambda: defaultdict(list))
    coverage = set()
    for row, _ in parquet_rows(paths, "references", ("arxiv_id", "ref_arxiv_id"), REF_FIELDS):
        citing = arxiv_id(row["arxiv_id"])
        coverage.add(range_of(citing))
        if citing not in scope or not row["ref_arxiv_id"]:
            continue
        cited = arxiv_id(row["ref_arxiv_id"])
        pool = candidates[citing]
        if citing != cited:
            pool[cited].append(normalize(row.get("ref_title")))
    return candidates, coverage


def read_completion(paths):
    latest = {}
    for path in paths:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    latest[arxiv_id(row["id"])] = row["status"]
    return {ident for ident, status in latest.items() if status in {"ok", "no_refs"}}


def read_bibtex(paths, scope):
    entries = defaultdict(dict)
    for row, path in parquet_rows(paths, "bibtex", ("arxiv_id", "citation_key"), BIB_FIELDS):
        ident = arxiv_id(row["arxiv_id"])
        if ident not in scope:
            continue
        key = row["citation_key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"citation_key trống: {ident} ({path})")
        item = {name: row.get(name) for name in BIB_FIELDS}
        item.update(arxiv_id=ident, citation_key=key)
        previous = entries[ident].setdefault(key, item)
        if previous != item:
            raise ValueError(f"Xung đột BibTeX ({ident}, {key}) tại {path}")
    return entries
