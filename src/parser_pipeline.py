"""Điều phối parser theo bài báo, giữ riêng dữ liệu bảng và chẩn đoán.

Không chạy corpus, ghi tệp hay nhập pandas. Khóa phiên bản trong ``key_map``
là chuỗi để kết quả giữ nguyên sau vòng ghi/đọc JSON. ``versions_error`` gồm
cả ``missing_dir`` và ``not_found``; hai bộ đếm riêng là tập con của nó.
"""

from __future__ import annotations

import json
import os
import re
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .bibtex import BibliographyMetadataWarning, BibliographyWarning, extract_bibtex, unify_publication
    from .gather_clean_tex import MissingBibliographyWarning, cleanup_latex, gather_tex
    from .hierarchy import HierarchyDeduplicationWarning, build_element_tables, citation_keys, construct_hierarchy, rewrite_citations
    from .utils import get_paper_dir, list_files
else:
    from bibtex import BibliographyMetadataWarning, BibliographyWarning, extract_bibtex, unify_publication
    from gather_clean_tex import MissingBibliographyWarning, cleanup_latex, gather_tex
    from hierarchy import HierarchyDeduplicationWarning, build_element_tables, citation_keys, construct_hierarchy, rewrite_citations
    from utils import get_paper_dir, list_files

__all__ = ["discover_publications", "parse_publication"]


def _arxiv_id(value: str) -> str:
    """Đổi tên thư mục do get_paper_dir tạo thành ID chuẩn, không nhận hậu tố vN."""
    if not isinstance(value, str):
        raise ValueError(f"Invalid arXiv ID: {value!r}")
    modern = re.fullmatch(r"(\d{2}(?:0[1-9]|1[0-2]))[.-](\d{4,5})", value)
    if modern:
        return ".".join(modern.groups())
    old = re.fullmatch(r"([a-z][a-z-]*(?:[.-][A-Z]{2})?)[/-](\d{2}(?:0[1-9]|1[0-2])\d{3})", value)
    if old:
        archive, number = old.groups()
        archive = re.sub(r"-([A-Z]{2})$", r".\1", archive)
        return f"{archive}/{number}"
    raise ValueError(f"Invalid arXiv ID: {value!r}")


def _version(value: str) -> int:
    """Đọc số phiên bản dương ở dạng N hoặc vN."""
    if not isinstance(value, str) or not re.fullmatch(r"v?[1-9]\d*", value):
        raise ValueError(f"Invalid version name: {value!r}")
    return int(value.removeprefix("v"))


def _read_json(path: Path) -> dict:
    """Đọc đối tượng JSON và từ chối khóa trùng thay vì âm thầm ghi đè."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    data = json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=pairs)
    if not isinstance(data, dict):
        raise ValueError("JSON content must be an object")
    return data


def _expected(data: dict, label: str, errors: list[str]) -> set[int]:
    """Lấy miền phiên bản đã khai báo; phân biệt thiếu trường với trường hỏng."""
    if "num_versions" not in data:
        return set()
    count = data["num_versions"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        errors.append(f"{label}: num_versions must be a positive integer")
        return set()
    return set(range(1, count + 1))


def discover_publications(data_dir, target_ids=None, source_state_paths=None, *,
                          existing_only=False, limit=None) -> list[dict]:
    """Tìm bài và hợp miền phiên bản từ metadata, trạng thái tải và thư mục.

    ``None`` hợp ID hợp lệ từ thư mục bài và trạng thái nguồn, kể cả bài chưa
    có thư mục. Danh sách đích tường minh chỉ giữ các ID được yêu cầu, kể cả
    bài vắng mặt; chưa biết phiên bản nào thì dùng v1 để ghi nhận thiếu đầu vào.
    Mặc định đọc source_state.json ở gốc và dưới .state. Danh sách đường dẫn
    tường minh thay thế mặc định; đường dẫn tương đối tính từ data_dir.
    Lỗi trạng thái toàn cục được gắn vào mọi job; khi không có job để chứa lỗi,
    hàm phát sinh ValueError. Không chọn ngầm một trạng thái đang xung đột.

    ``existing_only`` chỉ chọn bài có thư mục trong kho. Khi có ID đích, kiểm
    tra trực tiếp tên thư mục chuẩn và tên dùng dấu chấm, không liệt kê cả kho.
    Chế độ này mặc định bỏ qua source_state; vẫn đọc nếu truyền đường dẫn cụ thể.
    ``limit`` giới hạn số bài trước khi đọc metadata và liệt kê phiên bản.
    """
    root = Path(data_dir)
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer or None")
    ids = None if target_ids is None else sorted({_arxiv_id(v) for v in target_ids})
    directories = defaultdict(list)
    global_errors = []
    if root.exists() and not root.is_dir():
        raise ValueError(f"Data path is not a directory: {root}")
    if root.is_dir():
        if existing_only and ids is not None:
            candidates = sorted({root / name for identity in ids for name in
                                 (get_paper_dir(identity).name, identity.replace("/", "-"))})
            for directory in candidates:
                try:
                    identity = _arxiv_id(directory.name)
                except ValueError:
                    continue
                if directory.is_dir():
                    directories[identity].append(directory)
        else:
            # scandir lấy loại mục từ readdir, tránh một lần stat cho mỗi mục
            # (rất chậm trên ổ mạng như Google Drive). Có ID đích thì bỏ qua
            # ngay các thư mục ngoài phạm vi trước khi kiểm tra loại mục.
            wanted = None if ids is None else set(ids)
            with os.scandir(root) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    try:
                        identity = _arxiv_id(entry.name)
                    except ValueError:
                        continue
                    if wanted is not None and identity not in wanted:
                        continue
                    if entry.is_dir():
                        directories[identity].append(root / entry.name)
    elif target_ids is None:
        raise FileNotFoundError(f"Missing data directory: {root}")

    if existing_only and source_state_paths is None:
        source_state_paths = []
    if source_state_paths is None:
        paths = ([root / "source_state.json"] if (root / "source_state.json").exists() else [])
        if (root / ".state").is_dir():
            paths.extend(sorted((root / ".state").rglob("source_state.json")))
    else:
        if isinstance(source_state_paths, (str, Path)):
            source_state_paths = [source_state_paths]
        paths = [Path(p) if Path(p).is_absolute() else root / p for p in source_state_paths]

    states = defaultdict(list)
    for path in sorted(set(paths)):
        try:
            document = _read_json(path)
            papers = document.get("papers")
            if not isinstance(papers, dict):
                raise ValueError("The papers field must be an object")
            for name, entry in sorted(papers.items()):
                try:
                    identity = _arxiv_id(name)
                except ValueError as exc:
                    global_errors.append(f"{path}: {exc}")
                    continue
                states[identity].append((path, entry))
        except (OSError, ValueError) as exc:
            global_errors.append(f"Could not read state {path}: {exc}")

    if ids is None:
        ids = sorted(set(directories) | set(states))
    if existing_only:
        ids = [identity for identity in ids if identity in directories]
    if limit is not None:
        ids = ids[:limit]

    jobs = []
    for identity in ids:
        errors = list(global_errors)
        candidates = sorted(directories.get(identity, []),
                            key=lambda p: (p.name != get_paper_dir(identity).name, p.name))
        paper_dir = candidates[0] if candidates else None
        if len(candidates) > 1:
            errors.append(f"Duplicate paper directories for {identity}: " + ", ".join(map(str, candidates)))
        metadata = {}
        versions = set()
        counts = set()
        folders = defaultdict(list)
        for directory in candidates:
            meta_path = directory / "metadata.json"
            if meta_path.exists():
                try:
                    data = _read_json(meta_path)
                    if directory == paper_dir:
                        metadata = data
                    versions.update(_expected(data, str(meta_path), errors))
                    if type(data.get("num_versions")) is int and data["num_versions"] > 0:
                        counts.add(data["num_versions"])
                    if "arxiv_id" in data and _arxiv_id(data["arxiv_id"]) != identity:
                        errors.append(f"{meta_path}: arxiv_id conflicts with the directory name")
                    if "categories" in data and (not isinstance(data["categories"], list)
                                                  or any(not isinstance(c, str) for c in data["categories"])):
                        errors.append(f"{meta_path}: categories must be a list of strings")
                except (OSError, ValueError) as exc:
                    errors.append(f"Invalid metadata {meta_path}: {exc}")
            with os.scandir(directory) as entries:
                for entry in sorted(entries, key=lambda e: e.name):
                    try:
                        number = _version(entry.name)
                    except ValueError:
                        continue
                    if not entry.is_dir():
                        continue
                    folders[number].append(directory / entry.name)
                    versions.add(number)

        status_sets = defaultdict(set)
        for path, entry in states.get(identity, []):
            if not isinstance(entry, dict):
                errors.append(f"{path}: state for paper {identity} must be an object")
                continue
            versions.update(_expected(entry, str(path), errors))
            if type(entry.get("num_versions")) is int and entry["num_versions"] > 0:
                counts.add(entry["num_versions"])
            records = entry.get("versions", {})
            if not isinstance(records, dict):
                errors.append(f"{path}: versions for {identity} must be an object")
                continue
            seen = set()
            for key, record in sorted(records.items()):
                try:
                    number = _version(key)
                except ValueError as exc:
                    errors.append(f"{path}: {exc}")
                    continue
                versions.add(number)
                if number in seen:
                    errors.append(f"{path}: duplicate version v{number} in state")
                seen.add(number)
                status = record.get("status") if isinstance(record, dict) else None
                if not isinstance(status, str) or status not in {"done", "no_tex", "not_found", "error", "pending", "failed", "downloading"}:
                    errors.append(f"{path}: invalid state for v{number}: {status!r}")
                    status_sets[number].add("invalid")
                else:
                    status_sets[number].add(status)
        if len(counts) > 1:
            errors.append(f"Conflicting num_versions for {identity}: {sorted(counts)}")
        if counts and versions and max(versions) > max(counts):
            errors.append(f"Version v{max(versions)} exceeds num_versions={max(counts)} for {identity}")
        version_dirs = {}
        for number in sorted(versions or {1}):
            choices = sorted(folders.get(number, []), key=lambda p: (p.name != f"v{number}", str(p)))
            version_dirs[number] = choices[0] if choices else None
            if len(choices) > 1:
                errors.append(f"Duplicate version directories for v{number}: " + ", ".join(map(str, choices)))
        statuses = {}
        for number, values in sorted(status_sets.items()):
            if len(values) > 1:
                errors.append(f"Conflicting state for v{number}: {sorted(values)}")
                statuses[number] = "conflict"
            else:
                statuses[number] = next(iter(values))
        jobs.append(dict(arxiv_id=identity, paper_dir=paper_dir, metadata=metadata,
                         version_dirs=version_dirs, source_status_by_version=statuses,
                         discovery_errors=errors))
    if not jobs and global_errors:
        raise ValueError("; ".join(global_errors))
    return jobs


def parse_publication(job, *, enrich=False, fuzzy_threshold=0.92) -> dict:
    """Parse mọi phiên bản rồi hợp nhất, viết lại trích dẫn và tạo bảng đúng một lần.

    Lỗi từng pha được giữ trong timings và errors của phiên bản, cảnh báo giữ
    trong warnings. Cảnh báo API lỗi thời vẫn được đếm nhưng không hạ trạng
    thái dữ liệu. Chỉ thiếu thư mục không chứng minh được not_found. Thư mục
    không có TeX là bản rỗng; nếu trạng thái nói done thì đó là lỗi đầu vào.
    input_bytes đếm tệp .tex/.bib trong các thư mục phiên bản, kể cả tệp dư,
    mỗi đường dẫn thực một lần. elements_before đếm nút trước khử trùng lặp.
    """
    started = time.perf_counter()
    identity = job["arxiv_id"]
    paper_dir = job.get("paper_dir")
    # Thông điệp chỉ ghi đường dẫn tính từ thư mục dữ liệu: báo cáo không chứa
    # đường dẫn máy chạy và giống nhau dù đọc từ Drive hay từ bản giải nén.
    prefixes = []
    if paper_dir is not None:
        root = Path(paper_dir).parent
        prefixes = sorted({str(root) + os.sep, str(root.resolve()) + os.sep}, key=len, reverse=True)

    def relative(message):
        for prefix in prefixes:
            message = message.replace(prefix, "")
        return message

    discovery_errors = [relative(message) for message in job.get("discovery_errors", [])]
    # Cảnh báo lỗi thời của thư viện (DeprecationWarning, ...) không nói gì về
    # dữ liệu: chỉ đếm theo loại, không ghi từng dòng vào warnings.
    dependency_warnings = Counter()
    result = dict(elements=[], hierarchy=[], bibtex=[], version_status=[],
                  bibtex_debug=[], key_map={}, stats={}, timings=[], warnings=[])
    rows = {}
    trees, entries = {}, {}
    directories = job.get("version_dirs", {})
    source_status = job.get("source_status_by_version", {})
    numbers = sorted(set(directories) | set(source_status)) or [1]
    input_bytes = 0
    counted = set()
    fatal = []

    def warning(stage, message, version=None, category="ParserWarning", *, affects_status=True,
                requires_review=None):
        if requires_review is None:
            requires_review = affects_status
        result["warnings"].append(dict(arxiv_id=identity, stage=stage, version=version,
                                       category=category, affects_status=affects_status,
                                       severity="warning" if affects_status else "notice",
                                       requires_review=requires_review,
                                       message=f"Warning: {relative(message)}"))
        affected = [rows[version]] if version in rows else [r for r in rows.values() if r["status"] in {"ok", "partial"}]
        for row in affected:
            row["warning_count"] += 1
            row["blocking_warning_count"] += int(affects_status)
            row["notice_count"] += int(not affects_status)
            row["review_required"] |= requires_review
            if affects_status and row["status"] == "ok":
                row["status"] = "partial"

    def phase(stage, operation, version=None):
        """Đo thời gian thực và thu cảnh báo ngay cả khi thao tác phát sinh lỗi."""
        begin = datetime.now(timezone.utc).isoformat()
        tick = time.perf_counter()
        value, error = None, None
        warning_start = len(result["warnings"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                value = operation()
            except Exception as exc:
                error = relative(f"Error in {stage}: {type(exc).__name__}: {exc}")
            finally:
                duration = time.perf_counter() - tick
                end = datetime.now(timezone.utc).isoformat()
        for item in caught:
            category = item.category
            if issubclass(category, (DeprecationWarning, PendingDeprecationWarning, FutureWarning)):
                dependency_warnings[category.__name__] += 1
                continue
            retained = issubclass(category, (HierarchyDeduplicationWarning, BibliographyMetadataWarning))
            optional_enrichment = issubclass(category, BibliographyWarning) and str(item.message).startswith(
                ("CrossRef result rejected", "CrossRef enrichment failed", "LLM fallback failed"))
            optional_bibliography = issubclass(category, MissingBibliographyWarning)
            blocking = not (retained or optional_enrichment or optional_bibliography)
            # Hai trường hợp khử trùng đã được xử lý không mất dữ liệu, không cần người xem lại.
            safely_separated = issubclass(category, HierarchyDeduplicationWarning) and (
                str(item.message).startswith("Identical content occurs with different node kinds")
                or "have multiple parents in version" in str(item.message))
            warning(stage, str(item.message), version, item.category.__name__,
                    affects_status=blocking, requires_review=not safely_separated)
        new_warnings = result["warnings"][warning_start:]
        incomplete = any(w["affects_status"] for w in new_warnings)
        timing = dict(arxiv_id=identity, stage=stage, started_at=begin, ended_at=end,
                      duration_sec=duration, status="error" if error else ("partial" if incomplete else "ok"),
                      message=error or ("Processed with warnings" if new_warnings else "Processed"))
        if version is not None:
            timing["version"] = version
        result["timings"].append(timing)
        if error:
            if version is None:
                fatal.append(error)
            else:
                rows[version]["errors"].append(error)
        return error is None, value

    for number in numbers:
        row = dict(arxiv_id=identity, version=number, status="ok", message="",
                   missing_count=0, warning_count=0, source_status=source_status.get(number, ""),
                   bibliography_status="not_run", bibliography_missing=[], required_missing_count=0,
                   blocking_warning_count=0, notice_count=0, review_required=False, missing=[], errors=[])
        rows[number] = row
        trees[number], entries[number] = [], []
        directory = directories.get(number)

        def inspect_input():
            nonlocal input_bytes
            status = row["source_status"]
            exists = directory is not None and Path(directory).is_dir()
            if not exists:
                if status == "not_found":
                    row.update(status="not_found", message="Download state confirms the version was not found")
                    return False
                if status == "no_tex":
                    row.update(status="empty", message="Download state confirms no TeX source is available")
                    return False
                row.update(status="missing_dir", missing_count=1, missing=[f"v{number}"])
                raise FileNotFoundError(f"Missing version directory v{number}")
            files = list_files(directory)
            for path in files:
                if path.suffix.lower() in {".tex", ".bib"}:
                    resolved = path.resolve()
                    if resolved not in counted:
                        input_bytes += path.stat().st_size
                        counted.add(resolved)
            if status == "not_found":
                row.update(status="not_found", message="Download state confirms the version was not found")
                if any(p.suffix.lower() == ".tex" for p in files):
                    row["errors"].append("Conflict: a not_found version still contains TeX sources")
                return False
            return True

        ready, has_directory = phase("input", inspect_input, number)
        if not ready or not has_directory:
            if row["status"] == "ok":
                row["status"] = "error"
            result["timings"][-1].update(status=row["status"],
                                         message="; ".join(filter(None, [row["message"], *row["errors"]])))
            continue
        ok, tex = phase("gather", lambda: gather_tex(Path(directory)), number)
        if not ok:
            row["status"] = "error"
            continue
        if tex is None:
            if row["source_status"] in {"done", "conflict", "invalid", "error", "failed", "pending", "downloading"}:
                row["status"] = "error"
                row["errors"].append("No TeX source found, but download state does not confirm an empty version")
            else:
                row.update(status="empty", message="Version directory contains no TeX sources")
            result["timings"][-1].update(status=row["status"],
                                         message=row["message"] or "; ".join(row["errors"]))
            continue
        row["missing"] = list(tex.missing)
        row["missing_count"] = len(tex.missing)
        optional_missing = [name for name in tex.missing if Path(name).suffix.lower() in {".bib", ".bbl"}]
        required_missing = [name for name in tex.missing if name not in optional_missing]
        row["bibliography_missing"] = optional_missing
        row["required_missing_count"] = len(required_missing)
        row["bibliography_status"] = "missing_sources" if optional_missing else "ok"
        if required_missing:
            warning("gather", "Missing content files: " + ", ".join(required_missing), number)
        if optional_missing:
            warning("gather", "Skipped bibliography with missing sources: " + ", ".join(optional_missing), number,
                    affects_status=False, requires_review=True)
        for note in tex.notes:
            intentional = note.startswith(("skipped by \\includeonly:", "skipped system font mapping:"))
            decoded = note.startswith("encoding fallback (latin-1):")
            missing_bibliography = note.startswith("bibliography not found:") or any(
                note.startswith(f"include not found: {name}") for name in optional_missing)
            warning("gather", f"Source gathering note: {note}", number,
                    affects_status=not (intentional or decoded or missing_bibliography), requires_review=not intentional)
        if row["source_status"] and row["source_status"] != "done":
            warning("input", f"Existing TeX sources conflict with download state {row['source_status']}", number)
        clean_ok, cleaned = phase("cleanup", lambda: cleanup_latex(tex.source), number)
        if clean_ok:
            tree_ok, nodes = phase("construct_hierarchy", lambda: construct_hierarchy(cleaned), number)
            if tree_ok:
                trees[number] = nodes
        # Giữ GatherTex gốc: cleanup có thể làm mất cấu trúc bibitem cần trích xuất.
        bib_ok, bibliography = phase("extract_bibtex", lambda: extract_bibtex(tex, identity, enrich=enrich), number)
        if bib_ok:
            entries[number] = bibliography
        else:
            row["bibliography_status"] = "error"
        if row["errors"]:
            row["status"] = "partial" if trees[number] or entries[number] else "error"

    elements_before = sum(len(nodes) for nodes in trees.values())
    bibtex_before = sum(len(items) for items in entries.values())

    def unify():
        bibliography = unify_publication(identity, entries, fuzzy_threshold=fuzzy_threshold)
        result["bibtex"] = [item.to_row() for item in bibliography.entries]
        result["bibtex_debug"] = [item.to_debug_row() for item in bibliography.entries]
        result["key_map"] = {str(v): mapping for v, mapping in bibliography.key_map.items()}
        for item in bibliography.entries:
            if item.conflicts:
                ambiguous = any(conflict["field"] == "ambiguous_citation_key" for conflict in item.conflicts)
                for v in item.versions_present:
                    warning("unify_publication", f"{'Ambiguous citation key' if ambiguous else 'Metadata selected according to the merge policy'}: {item.citation_key}", v,
                            affects_status=ambiguous, requires_review=True)
        return bibliography

    unified_ok, bibliography = phase("unify_publication", unify)
    if unified_ok:
        def rewrite():
            return {v: rewrite_citations(nodes, bibliography.key_map.get(v, {})) for v, nodes in trees.items()}

        rewritten_ok, rewritten = phase("rewrite_citations", rewrite)
        if rewritten_ok:
            tables_ok, tables = phase("build_element_tables", lambda: build_element_tables(identity, rewritten))
            if tables_ok:
                result["elements"], result["hierarchy"] = tables
    # Đánh dấu entry có được \cite trong nội dung đã parse (\nocite{*} tính là tất cả).
    # Bảng bibtex vẫn giữ mọi entry; cờ chỉ nằm trong bản debug và thống kê.
    cited = set()
    for row in result["elements"]:
        cited |= citation_keys(row["content"])
    for row in result["bibtex_debug"]:
        row["cited"] = "*" in cited or row["citation_key"] in cited

    for number, row in rows.items():
        messages = ([row["message"]] if row["message"] else []) + row["errors"]
        messages.extend(w["message"] for w in result["warnings"] if w["affects_status"] and w["version"] in {None, number})
        row["message"] = "; ".join(messages) or (
            f"Version processed; {row['notice_count']} notices (see warnings.csv)" if row["notice_count"] else "Version processed")
        result["version_status"].append(row)
    counts = Counter(r["status"] for r in rows.values())
    valid = counts["ok"] + counts["empty"]
    error_count = sum(counts[s] for s in ("error", "missing_dir", "not_found"))
    has_data = bool(elements_before or bibtex_before)
    missing_input = (job.get("paper_dir") is None and not any(source_status.values()) and not has_data)
    if fatal:
        status = "error"
    elif missing_input:
        status = "missing_input"
    elif discovery_errors or error_count or counts["partial"]:
        status = "partial" if has_data or valid else "error"
    else:
        status = "empty" if counts["empty"] == len(numbers) else "ok"
    metadata = job.get("metadata", {})
    categories = metadata.get("categories", [])
    category = categories[0] if isinstance(categories, list) and categories else metadata.get("category", "")
    messages = discovery_errors + fatal
    messages.extend(f"v{r['version']}: {r['message']}" for r in rows.values() if r["status"] not in {"ok", "empty"})
    blocking_count = sum(w["affects_status"] for w in result["warnings"])
    review_count = sum(w["requires_review"] for w in result["warnings"])
    result["stats"] = dict(
        arxiv_id=identity, status=status, paper_success=status in {"ok", "empty"},
        versions_expected=len(numbers), versions_ok=counts["ok"], versions_empty=counts["empty"],
        versions_partial=counts["partial"], versions_error=error_count,
        versions_not_found=counts["not_found"], versions_missing_dir=counts["missing_dir"],
        elements_before=elements_before, elements_after=len(result["elements"]),
        bibtex_before=bibtex_before, bibtex_after=len(result["bibtex"]),
        elapsed_sec=time.perf_counter() - started, input_bytes=input_bytes,
        category=category, warnings_count=len(result["warnings"]),
        bibtex_cited=sum(row["cited"] for row in result["bibtex_debug"]),
        dependency_warning_count=sum(dependency_warnings.values()),
        bibliography_status=("error" if any(r["bibliography_status"] == "error" for r in rows.values()) else
            "missing_sources" if any(r["bibliography_status"] == "missing_sources" for r in rows.values()) else
            "ok" if any(r["bibliography_status"] == "ok" for r in rows.values()) else "not_run"),
        versions_missing_bibliography=sum(r["bibliography_status"] == "missing_sources" for r in rows.values()),
        blocking_warning_count=blocking_count, notice_count=len(result["warnings"]) - blocking_count,
        review_required=bool(review_count), review_count=review_count,
        message="; ".join(messages) or ("All versions are empty" if status == "empty" else
            ("Processing complete; review diagnostics in warnings.csv" if review_count else "Processing complete")))
    return result
