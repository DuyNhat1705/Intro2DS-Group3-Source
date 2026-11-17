r"""Hậu xử lý bảng ``bibtex`` sau ``run_parser``: tách lại ``\bibitem`` đáng ngờ bằng AnyStyle.

AnyStyle (bộ tách trích dẫn dùng mô hình CRF) chỉ chạy cho các ``\bibitem`` có
dấu hiệu tách sai: thiếu tiêu đề/tác giả, tiêu đề trông như tên nơi đăng, tên
tác giả dài bất thường. Mọi mục được gửi trong một lần gọi CLI theo khối, không
gọi riêng từng mục. Đầu vào duy nhất là văn bản ``\bibitem`` của chính bài báo:
không dùng dữ liệu tham khảo đã thu thập (``references.json``) hay dịch vụ ngoài,
vì Milestone 3 ghép chính các entry này với dữ liệu đó.

Kết quả AnyStyle được cache trong ``anystyle.jsonl``: lần chạy sau chỉ gọi CLI
cho phần chưa có, nên kết quả lặp lại được. Partition và checkpoint của parser
không bị sửa. ``tables/<worker>/bibtex.parquet`` được thay bằng bảng đã tinh
chỉnh (đánh dấu trong metadata Parquet), bảng gốc giữ ở ``bibtex_parsed.parquet``,
mọi thay đổi ghi vào ``bibtex_refinement.jsonl``.

Giới hạn: bước này không chạy lại khử trùng; nếu hai entry trở nên trùng nhau
sau khi được sửa thì vẫn là hai dòng.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

if __package__:
    from .bibtex import _clean_author, _folded, _standardize_text
    from .parser_reporting import _atomic_json, _paper_name, table_schemas
else:
    from bibtex import _clean_author, _folded, _standardize_text
    from parser_reporting import _atomic_json, _paper_name, table_schemas

__all__ = ["refine_bibliography", "run_anystyle", "anystyle_available", "suspicion"]

_ANYSTYLE_CHUNK = 2000
_REFINED_KEY = b"bibtex_refined"

# Tên nơi đăng thường mở đầu như vầy; một "tiêu đề" như thế nhiều khả năng là tên tạp chí/hội nghị.
_VENUE_LIKE = re.compile(
    r"^(?:proc(?:eedings|\.)?|journal|transactions|advances in|annals|ann\.|bulletin|letters|"
    r"international (?:conference|journal|symposium|workshop|colloquium)|conference|symposium|workshop|"
    r"ieee|acm|lecture notes|arxiv|siam j|j\.|phys\. rev|physical review|communications in|"
    r"mathematische|revista|neural information processing)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Cache JSONL: mỗi dòng {"key": ..., "value": ...}; dòng hỏng cuối tệp bị bỏ qua.
# ---------------------------------------------------------------------------


def _load_cache(path: Path) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                cache[row["key"]] = row["value"]
            except (ValueError, KeyError, TypeError):
                continue
    return cache


def _append_cache(path: Path, items: Dict[str, Any]) -> None:
    if not items:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        for key, value in items.items():
            stream.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# AnyStyle
# ---------------------------------------------------------------------------


def anystyle_available(command: str = "anystyle") -> bool:
    return shutil.which(command) is not None


def run_anystyle(texts: Sequence[str], *, command: str = "anystyle") -> List[dict]:
    """Chạy ``anystyle -f json --stdout parse`` trên từng khối 2.000 dòng.

    Mỗi dòng đầu vào là một reference; kết quả là danh sách dict cùng độ dài.
    Khởi động Ruby và nạp mô hình tốn vài giây, nên gọi theo khối chứ không theo mục.
    """
    results: List[dict] = []
    for start in range(0, len(texts), _ANYSTYLE_CHUNK):
        chunk = [" ".join(text.split()) for text in texts[start : start + _ANYSTYLE_CHUNK]]
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "references.txt"
            source.write_text("\n".join(chunk) + "\n", encoding="utf-8")
            completed = subprocess.run(
                [command, "-f", "json", "--stdout", "parse", str(source)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                check=False,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"anystyle failed ({completed.returncode}): {completed.stderr.strip()[:500]}")
        parsed = json.loads(completed.stdout)
        if not isinstance(parsed, list) or len(parsed) != len(chunk):
            raise RuntimeError(
                f"anystyle returned {len(parsed) if isinstance(parsed, list) else 'invalid'} "
                f"items for {len(chunk)} references"
            )
        results.extend(item if isinstance(item, dict) else {} for item in parsed)
    return results


def anystyle_version(command: str = "anystyle") -> Optional[str]:
    try:
        completed = subprocess.run([command, "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return completed.stdout.strip() or None


def _plain_reference(raw: str) -> str:
    """Văn bản đưa vào AnyStyle: Unicode, khối \\newblock thành dấu chấm câu."""
    text = re.sub(r"\\newblock\b\s*", ". ", raw)
    return " ".join((_standardize_text(text) or "").split())


def _first(values: Any) -> Optional[str]:
    if isinstance(values, list) and values and isinstance(values[0], str) and values[0].strip():
        return values[0].strip()
    return None


def _anystyle_fields(item: dict) -> Dict[str, Any]:
    authors = []
    for person in item.get("author") or []:
        if isinstance(person, dict):
            name = person.get("literal") or " ".join(
                part for part in (person.get("given"), person.get("family")) if isinstance(part, str) and part
            )
            if name and name.strip():
                authors.append(_clean_author(name.strip()))
    year = None
    date = _first(item.get("date"))
    if date:
        match = re.search(r"(?<!\d)(1[5-9]\d{2}|20\d{2})(?!\d)", date)
        year = int(match.group(1)) if match else None
    title = _first(item.get("title"))
    journal = _first(item.get("container-title")) or _first(item.get("journal"))
    return {
        "title": title.rstrip(" .,;:") if title else None,
        "authors": authors,
        "year": year,
        "journal": journal.rstrip(" .,;:") if journal else None,
    }


def suspicion(row: dict) -> List[str]:
    """Lý do một entry ``\\bibitem`` có thể đã bị tách sai (rỗng nếu trông ổn)."""
    reasons = []
    title, authors, journal = row.get("title"), row.get("authors") or [], row.get("journal")
    if not title:
        reasons.append("no_title")
    elif _VENUE_LIKE.match(title.strip()):
        reasons.append("title_is_venue")
    elif journal and _folded(title) == _folded(journal):
        reasons.append("title_equals_journal")
    if not authors:
        reasons.append("no_authors")
    elif any(len(name.split()) > 5 or re.search(r"\d|\bIn$", name) for name in authors):
        reasons.append("author_too_long")
    return reasons


# ---------------------------------------------------------------------------
# Hậu xử lý
# ---------------------------------------------------------------------------


def _debug_rows(partition: Path) -> Dict[str, dict]:
    path = partition / "bibtex_debug.jsonl"
    rows = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            rows[row.get("citation_key")] = row
    return rows


def refine_bibliography(
    output_dir,
    *,
    worker: str = "local",
    cache_dir=None,
    use_anystyle: bool = True,
    anystyle_runner: Optional[Callable[[Sequence[str]], List[dict]]] = None,
    anystyle_command: str = "anystyle",
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Tinh chỉnh ``tables/<worker>/bibtex.parquet``; trả về thống kê của lần chạy.

    ``anystyle_runner`` thay được (để kiểm thử). Khi không có lệnh ``anystyle``,
    bước này được bỏ qua và ghi lý do vào thống kê thay vì dừng; bảng vẫn được
    ghi lại (giống bản do parser tạo) để đầu ra luôn ở cùng một trạng thái.
    """
    root = Path(output_dir).resolve()
    tables = root / "tables" / worker
    reports = root / "report_exports" / worker
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "refine_cache"
    current, parsed = tables / "bibtex.parquet", tables / "bibtex_parsed.parquet"
    if not current.is_file():
        raise FileNotFoundError(f"{current} not found; run run_parser first")
    metadata = pq.read_schema(current).metadata or {}
    # bibtex.parquet mới do run_parser ghi thì lưu lại làm bản gốc; nếu đã tinh chỉnh thì
    # đọc bản gốc đã lưu, để chạy lại nhiều lần cho cùng kết quả.
    if _REFINED_KEY not in metadata:
        shutil.copyfile(current, parsed)
    schema = table_schemas()["bibtex"]
    rows = pq.read_table(parsed).to_pylist()
    say = progress or (lambda message: None)
    stats: Dict[str, Any] = {"rows": len(rows), "anystyle": {}}
    changes: List[dict] = []

    def change(row, field, value, reason):
        changes.append(
            {
                "arxiv_id": row["arxiv_id"],
                "citation_key": row["citation_key"],
                "field": field,
                "old": row.get(field),
                "new": value,
                "source": "anystyle",
                "reason": reason,
            }
        )
        row[field] = value

    if use_anystyle:
        started = time.perf_counter()
        debug: Dict[str, Dict[str, dict]] = {}
        candidates = []
        for index, row in enumerate(rows):
            if row["arxiv_id"] not in debug:
                debug[row["arxiv_id"]] = _debug_rows(tables / "papers" / _paper_name(row["arxiv_id"]))
            raw = debug[row["arxiv_id"]].get(row["citation_key"], {}).get("raw_text")
            reasons = suspicion(row) if raw else []
            if reasons:
                candidates.append((index, _plain_reference(raw), reasons))
        stats["anystyle"]["candidates"] = len(candidates)
        runner = anystyle_runner
        if runner is None and candidates:
            if anystyle_available(anystyle_command):
                stats["anystyle"]["version"] = anystyle_version(anystyle_command)
                runner = lambda texts: run_anystyle(texts, command=anystyle_command)  # noqa: E731
            else:
                stats["anystyle"]["skipped"] = f"command {anystyle_command!r} not found"
        if runner is not None and candidates:
            cache_path = cache_dir / "anystyle.jsonl"
            cache = _load_cache(cache_path)
            keys = {text: hashlib.sha256(text.encode("utf-8")).hexdigest() for _, text, _ in candidates}
            missing = list(dict.fromkeys(text for text, key in keys.items() if key not in cache))
            say(f"AnyStyle: {len(candidates):,} candidates, {len(missing):,} not cached")
            fresh = dict(zip((keys[text] for text in missing), runner(missing))) if missing else {}
            _append_cache(cache_path, fresh)
            cache.update(fresh)
            stats["anystyle"].update(parsed=len(missing), cache_hits=len(candidates) - len(missing))
            for index, text, reasons in candidates:
                row, found = rows[index], _anystyle_fields(cache.get(keys[text]) or {})
                why = ",".join(reasons)
                title_wrong = bool({"no_title", "title_is_venue", "title_equals_journal"} & set(reasons))
                # Tiêu đề AnyStyle phải xuất hiện trong chính văn bản \bibitem.
                if (
                    title_wrong
                    and found["title"]
                    and not _VENUE_LIKE.match(found["title"])
                    and _folded(found["title"]) in _folded(text)
                ):
                    old = row.get("title")
                    change(row, "title", found["title"], why)
                    if old and "title_is_venue" in reasons and not row.get("journal"):
                        change(row, "journal", old, "title_is_venue")
                if {"no_authors", "author_too_long"} & set(reasons) and found["authors"]:
                    change(row, "authors", found["authors"], why)
                if row.get("year") is None and found["year"] is not None:
                    change(row, "year", found["year"], "missing")
                if not row.get("journal") and found["journal"] and found["journal"] != row.get("title"):
                    change(row, "journal", found["journal"], "missing")
        stats["anystyle"]["sec"] = round(time.perf_counter() - started, 3)

    stats["changes_by_field"] = {}
    for item in changes:
        stats["changes_by_field"][item["field"]] = stats["changes_by_field"].get(item["field"], 0) + 1
    stats["rows_changed"] = len({(c["arxiv_id"], c["citation_key"]) for c in changes})

    table = pa.Table.from_pylist(rows, schema=schema).replace_schema_metadata({_REFINED_KEY: b"1"})
    temporary = tables / "bibtex.parquet.tmp"
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, current)
    log = tables / "bibtex_refinement.jsonl.tmp"
    with log.open("w", encoding="utf-8") as stream:
        for item in changes:
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
    os.replace(log, tables / "bibtex_refinement.jsonl")
    reports.mkdir(parents=True, exist_ok=True)
    _atomic_json(reports / "bibtex_refinement_summary.json", stats)
    return stats
