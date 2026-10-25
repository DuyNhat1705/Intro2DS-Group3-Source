"""Download full LaTeX sources (every version) of arXiv papers.

Workflow (see scrape_batch):
    get_source (download) -> save_source (store archive) -> sizes + time/memory stats
Extraction is separate: use ipynb_workflow(), or call extract_full / extract_tex_bib on demand.

Progress is stored in DATA_DIR/source_state.json after every version, so a run
interrupted by rate limits / blocks / Ctrl+C resumes where it stopped.
"""
import functools
import gzip
import io
import json
import logging
import os
import math
import re
import shutil
import sys
import tarfile
import time
import tracemalloc
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import arxiv
import requests

from utils import DATA_DIR, get_paper_dir

# ====== EDIT HERE: papers to scrape ======
# {arxiv_id: num_versions}; None = read num_versions from data/<id>/metadata.json (get_metadata.py),
# falling back to the arxiv API if that file does not exist.
PAPERS = {
    "2608.04007": 1,
    "2608.04008": None,
    "2608.04009": None,
}
# =========================================
# False = delete the archive after extraction (saves disk); True = keep it in the version folder.
KEEP_ARCHIVE = False

SRC_URL = "https://export.arxiv.org/src/{id}"  # never use arxiv.org (robot detection)
USER_AGENT = "Intro2DS-Group3-Scraper/1.0 (student project; contact: duynhata0@gmail.com)"
REQUEST_DELAY = 4.0  # seconds between two requests
STATE_FILE = DATA_DIR / "source_state.json"
WANTED_EXT = (".tex", ".bib")
TERMINAL = {"done", "no_tex", "not_found"}  # statuses that are never retried
ARCHIVE_NAME = {
    "tar.gz": "source.tar.gz",
    "tar": "source.tar",
    "gz": "source.gz",  # single gzipped .tex file
    "zip": "source.zip",
    "tex": "source.tex",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("get_full_source")


class DownloadError(Exception):
    """Transient failure; the version stays un-finished and is retried next run."""


class RateLimitError(DownloadError):
    """Still 429/5xx after all retries: stop and resume later."""


class BlockedError(DownloadError):
    """HTTP 403: we are blocked. Stop immediately."""


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
_last_request_end = 0.0
_total_slept = 0.0

# Sleep luồng chạy
def _sleep(seconds: float) -> None:
    global _total_slept
    if seconds > 0:
        time.sleep(seconds)
        _total_slept += seconds

# Chờ cho đến khi `delay` giây trôi qua kể từ khi request trước đó kết thúc.
def _wait_for_slot(delay: float) -> None:
    """Block until `delay` seconds passed since the previous request ended."""
    _sleep(delay - (time.monotonic() - _last_request_end))

# Đánh dấu thời điểm kết thúc của request hiện tại.
def _mark_request_end() -> None:
    global _last_request_end
    _last_request_end = time.monotonic()


# --------------------------------------------------------------------------- #
# Source type detection
# --------------------------------------------------------------------------- #

# Kiểm tra xem tên file có chứa .tex hoặc .bib (hậu tố đã khai báo trong WANTED_EXT).
def _wanted(name: str) -> bool:
    return name.lower().endswith(WANTED_EXT)

# Kiểm tra bytes đầu của dữ liệu để xác định xem nó có giống như một file LaTeX hay không.
def _looks_like_tex(data: bytes) -> bool:
    head = data[:200_000]
    if b"\x00" in head[:2000]:
        return False
    text = head.decode("utf-8", "ignore")
    return any(m in text for m in ("\\documentclass", "\\begin{document}", "\\usepackage", "\\section", "\\input"))

# Kiểm tra loại source còn đang archive và xác định xem nó có chứa nội dung .tex/.bib hay không.
def identify_source(data: bytes) -> Tuple[str, bool]:
    """Return (kind, usable). usable = contains .tex/.bib content worth keeping.

    kinds: tar.gz, tar, gz (single gzipped tex), zip, tex, pdf, docx, other.
    """
    if data[:4] == b"%PDF":
        return "pdf", False
    if data[:2] == b"\x1f\x8b":
        try:
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
                return "tar.gz", any(m.isfile() and _wanted(m.name) for m in tf.getmembers())
        except (tarfile.TarError, EOFError, OSError):
            pass
        try:
            inner = gzip.decompress(data)
        except (OSError, EOFError):
            return "other", False
        if inner[:4] == b"%PDF":
            return "pdf", False
        return "gz", _looks_like_tex(inner)
    if data[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                names = zf.namelist()
        except zipfile.BadZipFile:
            return "other", False
        if any(_wanted(n) for n in names):
            return "zip", True
        return ("docx" if "word/document.xml" in names else "other"), False
    if tarfile.is_tarfile(io.BytesIO(data)):
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            return "tar", any(m.isfile() and _wanted(m.name) for m in tf.getmembers())
    if _looks_like_tex(data):
        return "tex", True
    return "other", False


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
# Lấy số lượng version của bài báo (từ metadata.json nếu có, nếu không thì truy vấn API của arXiv).
def get_num_versions(arxiv_id: str, delay: float = REQUEST_DELAY) -> int:
    """Number of versions: read from data/<id>/metadata.json (written by get_metadata.py,
    field `num_versions`); only if missing, fall back to the arxiv API."""
    meta_file = get_paper_dir(arxiv_id) / "metadata.json"
    if meta_file.exists():
        try:
            n = json.loads(meta_file.read_text(encoding="utf-8")).get("num_versions")
            if isinstance(n, int) and n >= 1:
                return n
        except (json.JSONDecodeError, OSError):
            log.warning("%s: cannot read metadata.json, falling back to arxiv API", arxiv_id)
    log.info("%s: no metadata.json, querying arxiv API for num_versions", arxiv_id)
 
    # num_retries=0: retry của arxiv.py chỉ cách nhau ~delay giây, quá ngắn sau khi bị 429,
    # nên tự retry với thời gian chờ tăng dần (30s, 60s, 120s, 240s).
    client = arxiv.Client(delay_seconds=delay, num_retries=0)
    max_attempts = 4
    for attempt in range(1, max_attempts + 1):
        _wait_for_slot(delay)
        try:
            result = next(client.results(arxiv.Search(id_list=[arxiv_id])), None)
        except arxiv.HTTPError as e:
            status = getattr(e, "status", None)
            if status == 429 or (status or 0) >= 500:
                if attempt == max_attempts:
                    raise RateLimitError(f"arXiv API still returns HTTP {status} for {arxiv_id}") from e
                wait = 30 * 2 ** (attempt - 1)
                log.warning("%s: arXiv API HTTP %s, waiting %ds (attempt %d/%d)",
                            arxiv_id, status, wait, attempt, max_attempts)
                _sleep(wait)
                continue
            raise DownloadError(f"arXiv API HTTP {status} for {arxiv_id}") from e
        finally:
            _mark_request_end()
        if result is None:
            raise DownloadError(f"{arxiv_id}: not found in arXiv API")
        m = re.search(r"v(\d+)$", result.entry_id)
        return int(m.group(1)) if m else 1
    raise RateLimitError(f"Could not get num_versions for {arxiv_id}")

# Download một version cụ thể của bài báo từ arXiv. Nếu version không tồn tại (404), trả về None.
def download_version(arxiv_id: str, version: int, delay: float = REQUEST_DELAY,
                     session: Optional[requests.Session] = None, max_retries: int = 4) -> Optional[bytes]:
    """Download raw source of one version. None if the version does not exist (404)."""
    session = session or requests.Session()
    url = SRC_URL.format(id=f"{arxiv_id}v{version}")
    last_status = None
    for attempt in range(1, max_retries + 1):
        _wait_for_slot(delay)
        try:
            r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
        except requests.RequestException as e:
            _mark_request_end()
            last_status = None  # attempt này không có HTTP status; không để 429 cũ quyết định loại lỗi
            log.warning("%s: network error (%s), attempt %d/%d", url, e, attempt, max_retries)
            _sleep(delay * 2 ** attempt)
            continue
        _mark_request_end()
        last_status = r.status_code
        if r.status_code == 200:
            return r.content
        if r.status_code == 404:
            return None
        if r.status_code == 403:
            raise BlockedError(f"403 for {url}: blocked by arXiv, stop and retry later")
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("Retry-After", ""))
            except ValueError:
                wait = delay * 2 ** attempt
            log.warning("%s: HTTP %d, waiting %.0fs (attempt %d/%d)", url, r.status_code, wait, attempt, max_retries)
            _sleep(wait)
            continue
        raise DownloadError(f"Unexpected HTTP {r.status_code} for {url}")
    exc = RateLimitError if last_status in (429,) or (last_status or 0) >= 500 else DownloadError
    raise exc(f"Giving up on {url} (last status: {last_status})")

# Tạo một generator để tải xuống từng version của bài báo, bỏ qua các version đã hoàn thành (trong skip_versions).
def get_source(arXiv_id: str, num_versions: int, skip_versions: Iterable[int] = (),
               delay: float = REQUEST_DELAY,
               session: Optional[requests.Session] = None) -> Iterator[Tuple[int, Optional[bytes]]]:
    """Yield (version, source_bytes | None) for v1..v{num_versions}.

    A generator, so the caller can save each version (and the state) as soon as it
    arrives. Versions in `skip_versions` (already finished) are not requested.
    """
    skip = set(skip_versions)
    session = session or requests.Session()
    for v in range(1, num_versions + 1):
        if v in skip:
            continue
        yield v, download_version(arXiv_id, v, delay=delay, session=session)


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
# Lây đường dẫn đến thư mục của một version cụ thể của bài báo.
def _version_dir(arxiv_id: str, version: int) -> Path:
    return get_paper_dir(arxiv_id) / f"v{version}"

# Lưu trữ source đã tải xuống vào thư mục data/<id>/v<version>/source.<ext>. Nếu source không chứa .tex/.bib, chỉ tạo một thư mục rỗng.
def _write_source(arxiv_id: str, version: int, source: bytes, kind: str, usable: bool) -> Optional[Path]:
    vdir = _version_dir(arxiv_id, version)
    vdir.mkdir(parents=True, exist_ok=True)  # empty dir = empty record for pdf/docx-only versions
    if not usable:
        log.info("%s v%d: no .tex/.bib (%s) -> empty record", arxiv_id, version, kind)
        return None
    path = vdir / ARCHIVE_NAME[kind]
    path.write_bytes(source)
    return path

# Lưu trữ source đã tải xuống vào thư mục data/<id>/v<version>/source.<ext>. Nếu source không chứa .tex/.bib, chỉ tạo một thư mục rỗng.
def save_source(arXiv_id: str, version: int, source: bytes) -> None:
    """Save the raw archive to data/<id>/v<version>/source.<ext>.

    pdf / docx / other versions without .tex/.bib only get an empty directory.
    """
    kind, usable = identify_source(source)
    _write_source(arXiv_id, version, source, kind, usable)


# --------------------------------------------------------------------------- #
# State (resume)
# --------------------------------------------------------------------------- #
# Đọc source_state.json để lấy trạng thái hiện tại của các bài báo đã tải xuống. Nếu file bị hỏng hoặc không tồn tại, trả về trạng thái mặc định.
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.error("Corrupted state file %s, starting fresh", STATE_FILE)
    return {"papers": {}}

# Lưu trạng thái hiện tại vào file source_state.json.
def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, STATE_FILE)  # atomic: never leaves a half-written state

# Kiểm tra xem tất cả các version của bài báo đã hoàn thành hay chưa
def _is_complete(entry: dict) -> bool:
    n = entry.get("num_versions")
    vs = entry.get("versions", {})
    return bool(n) and all(vs.get(str(v), {}).get("status") in TERMINAL for v in range(1, n + 1))


# --------------------------------------------------------------------------- #
# Extraction (never called from scrape; used by ipynb_workflow or on demand)
# --------------------------------------------------------------------------- #
# Chuẩn hóa đường dẫn tương đối của file trong archive, loại bỏ các phần tử không hợp lệ (như "", ".", "..") 
# và kiểm tra xem đường dẫn có phải là tuyệt đối hay không. Nếu hợp lệ, trả về PurePosixPath; nếu không, trả về None.
def _safe_rel(name: str) -> Optional[PurePosixPath]:
    p = PurePosixPath(name.replace("\\", "/"))
    parts = [x for x in p.parts if x not in ("", ".")]
    if not parts or p.is_absolute() or ".." in parts:
        return None
    return PurePosixPath(*parts)

# Tìm file archive trong thư mục version.
def _find_archive(vdir: Path) -> Optional[Path]:
    return next((vdir / n for n in ARCHIVE_NAME.values() if (vdir / n).exists()), None)

# Lấy danh sách các file đã được giải nén trong thư mục version, loại trừ file archive gốc.
def _extracted_files(vdir: Path) -> List[Path]:
    """Files in the version dir that are not the raw archive (i.e. already extracted)."""
    archive_names = set(ARCHIVE_NAME.values())
    return [f for f in vdir.rglob("*")
            if f.is_file() and not (f.parent == vdir and f.name in archive_names)]

# Lấy danh sách các thư mục version của bài báo, lọc theo danh sách version mong muốn nếu được cung cấp.
def _version_dirs(arxiv_id: str, versions: Optional[Iterable[int]]) -> List[Tuple[int, Path]]:
    paper_dir = get_paper_dir(arxiv_id)
    wanted = set(versions) if versions is not None else None
    dirs = sorted(paper_dir.glob("v[0-9]*"), key=lambda p: int(p.name[1:])) if paper_dir.exists() else []
    return [(int(d.name[1:]), d) for d in dirs if wanted is None or int(d.name[1:]) in wanted]

# Dung lượng chiếm trên ổ đĩa: mỗi file được cấp phát theo cụm (cluster), nên làm tròn lên bội số của cluster.
# Đây là ước lượng (không tính metadata thư mục/MFT; NTFS có thể lưu file rất nhỏ ngay trong MFT).
@functools.lru_cache(maxsize=1)
def _cluster_size() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        import ctypes
        spc, bps, free, total = (ctypes.c_ulong() for _ in range(4))
        root = os.path.splitdrive(os.path.abspath(DATA_DIR))[0] + "\\"
        if ctypes.windll.kernel32.GetDiskFreeSpaceW(
                ctypes.c_wchar_p(root), ctypes.byref(spc), ctypes.byref(bps), ctypes.byref(free), ctypes.byref(total)):
            return spc.value * bps.value  # sectors/cluster x bytes/sector
        return 4096
    return os.statvfs(DATA_DIR).f_frsize


def _disk(size: int) -> int:
    """Bytes a file of `size` bytes occupies on disk (0 for empty files)."""
    c = _cluster_size()
    return math.ceil(size / c) * c if size > 0 else 0


# Giải nén archive vào thư mục version trong MỘT lượt đọc, ghi file theo luồng (không đọc cả file vào RAM).
# Trả về (danh sách file đã ghi, tổng byte của TOÀN BỘ file hợp lệ trong archive, tổng byte chiếm trên đĩa của chúng),
# kể cả hình không được ghi ra.
# only_tex_bib=True thì chỉ ghi .tex/.bib nhưng vẫn cộng kích thước mọi file -> có luôn số liệu "full" mà không cần extract full.
def _copy_stream(src, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1024 * 1024)
    return dest.stat().st_size


def _extract_archive(vdir: Path, archive: Path, only_tex_bib: bool) -> Tuple[List[str], int, int]:
    extracted: List[str] = []
    full_bytes = 0
    full_disk = 0

    def handle(name: str, size: int, opener):
        nonlocal full_bytes, full_disk
        rel = _safe_rel(name)
        if rel is None:  # đường dẫn không an toàn: bỏ qua cả ghi lẫn tính size
            return
        full_bytes += size
        full_disk += _disk(size)
        if only_tex_bib and not _wanted(rel.name):
            return
        with opener() as src:
            _copy_stream(src, vdir.joinpath(*rel.parts))
        extracted.append(str(rel))

    if archive.name in ("source.tar.gz", "source.tar"):
        with tarfile.open(archive, "r:*") as tf:
            for m in tf:
                if m.isfile():  # bỏ symlink, thiết bị
                    handle(m.name, m.size, lambda m=m: tf.extractfile(m))
    elif archive.name == "source.zip":
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    handle(info.filename, info.file_size, lambda i=info: zf.open(i))
    elif archive.name == "source.gz":  # một file tex đơn lẻ được nén
        with gzip.open(archive, "rb") as src:
            n = _copy_stream(src, vdir / "main.tex")
        full_bytes += n
        full_disk += _disk(n)
        extracted.append("main.tex")
    elif archive.name == "source.tex":
        with open(archive, "rb") as src:
            n = _copy_stream(src, vdir / "main.tex")
        full_bytes += n
        full_disk += _disk(n)
        extracted.append("main.tex")
    return extracted, full_bytes, full_disk


# Xóa mọi file không phải .tex/.bib trong thư mục version (trừ file archive) và các thư mục rỗng. Trả về danh sách file giữ lại.
def _prune(vdir: Path) -> List[str]:
    kept = []
    for f in _extracted_files(vdir):
        if _wanted(f.name):
            kept.append(f.relative_to(vdir).as_posix())
        else:
            f.unlink()
    for d in sorted((p for p in vdir.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return kept


# Giải nén tất cả các file từ archive vào thư mục version (figures, .sty, .bbl, ...). Ghi sizes["full"] vào state.
def extract_full(arxiv_id: str, versions: Optional[Iterable[int]] = None,
                 remove_archive: bool = False) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for v, vdir in _version_dirs(arxiv_id, versions):
        archive = _find_archive(vdir)
        if archive is None:
            continue
        out[v], _, _ = _extract_archive(vdir, archive, only_tex_bib=False)
        if remove_archive:
            archive.unlink()
        log.info("%s v%d: extracted %d files (full)", arxiv_id, v, len(out[v]))
    if out:
        record_sizes(arxiv_id, "full")
    return out


# Chỉ giữ .tex và .bib (giữ cấu trúc thư mục). Nếu đã extract full thì xóa tại chỗ file thừa, nếu chưa thì extract thẳng
# .tex/.bib từ archive. Ghi sizes["tex_bib"] vào state.
def extract_tex_bib(arxiv_id: str, versions: Optional[Iterable[int]] = None,
                    remove_archive: bool = False) -> Dict[int, List[str]]:
    out: Dict[int, List[str]] = {}
    for v, vdir in _version_dirs(arxiv_id, versions):
        archive = _find_archive(vdir)
        if _extracted_files(vdir):
            out[v] = _prune(vdir)
        elif archive is not None:
            out[v], _, _ = _extract_archive(vdir, archive, only_tex_bib=True)
        else:
            continue
        if remove_archive and archive is not None and archive.exists():
            archive.unlink()
        log.info("%s v%d: %d tex/bib files kept", arxiv_id, v, len(out[v]))
    if out:
        record_sizes(arxiv_id, "tex_bib")
    return out


# --------------------------------------------------------------------------- #
# Size / time / memory
# --------------------------------------------------------------------------- #
# Kích thước của một version (byte). Mỗi chỉ số có hai dạng: dung lượng logic và dung lượng chiếm trên ổ đĩa (*_disk_bytes):
#   archive_bytes   - file nén lúc tải (nếu archive đã xóa thì lấy từ size_bytes trong state)
#   extracted_bytes - các file đã giải nén, không tính archive
#   total_bytes     - thực tế trên đĩa lúc này (file đã giải nén + archive nếu còn)
# So sánh trước/sau extract_tex_bib thì so extracted_bytes (hoặc extracted_disk_bytes).
_SIZE_KEYS = ("archive_bytes", "extracted_bytes", "total_bytes",
              "archive_disk_bytes", "extracted_disk_bytes", "total_disk_bytes")


def get_version_size(arxiv_id: str, version: int, state: Optional[dict] = None) -> Dict[str, int]:
    vdir = _version_dir(arxiv_id, version)
    if not vdir.exists():
        return dict.fromkeys(_SIZE_KEYS, 0)
    archive = _find_archive(vdir)
    sizes = [f.stat().st_size for f in _extracted_files(vdir)]
    extracted, extracted_disk = sum(sizes), sum(_disk(x) for x in sizes)
    if archive is not None:
        archive_bytes = on_disk = archive.stat().st_size
        on_disk_disk = _disk(on_disk)
    else:  # archive đã xóa: lấy kích thước lúc tải từ state
        state = state if state is not None else load_state()
        info = state.get("papers", {}).get(arxiv_id, {}).get("versions", {}).get(str(version), {})
        archive_bytes = info.get("size_bytes", 0) if info.get("status") == "done" else 0
        on_disk = on_disk_disk = 0
    return {"archive_bytes": archive_bytes, "extracted_bytes": extracted, "total_bytes": extracted + on_disk,
            "archive_disk_bytes": _disk(archive_bytes), "extracted_disk_bytes": extracted_disk,
            "total_disk_bytes": extracted_disk + on_disk_disk}


def _with_totals(versions: Dict[int, Dict[str, int]]) -> dict:
    totals = {k: sum(x[k] for x in versions.values()) for k in _SIZE_KEYS}
    return {"versions": versions, "totals": totals}


# Kích thước từng version + tổng của cả bài (đọc từ đĩa).
def get_paper_size(arxiv_id: str, state: Optional[dict] = None) -> dict:
    state = state if state is not None else load_state()
    versions = {v: get_version_size(arxiv_id, v, state) for v, _ in _version_dirs(arxiv_id, None)}
    return _with_totals(versions)


# Lưu snapshot kích thước vào source_state.json theo giai đoạn: sizes["archive_only"], sizes["full"], sizes["tex_bib"].
# state=None: tự đọc/ghi file state; snapshot=None: tự đo từ đĩa.
def record_sizes(arxiv_id: str, stage: str, state: Optional[dict] = None, snapshot: Optional[dict] = None) -> dict:
    own = state is None
    state = load_state() if own else state
    entry = state["papers"].setdefault(arxiv_id, {"versions": {}})
    sizes = entry.get("sizes", {})
    if "totals" in sizes:  # định dạng cũ (một snapshot duy nhất) -> chuyển vào "after_download"
        sizes = {"after_download": sizes}
    sizes[stage] = snapshot if snapshot is not None else get_paper_size(arxiv_id, state)
    entry["sizes"] = sizes
    if own:
        save_state(state)
    return sizes[stage]


# Đo thời gian thực và bộ nhớ đỉnh (tracemalloc) của khối mã được bao quanh.
@contextmanager
def measure():
    result: dict = {}
    was_tracing = tracemalloc.is_tracing()
    if was_tracing:
        tracemalloc.reset_peak()
    else:
        tracemalloc.start()
    slept0, t0 = _total_slept, time.perf_counter()
    try:
        yield result
    finally:
        elapsed = time.perf_counter() - t0
        sleep = _total_slept - slept0
        result["elapsed_sec"] = round(elapsed, 2)
        result["sleep_sec"] = round(sleep, 2)
        result["active_sec"] = round(elapsed - sleep, 2)
        result["peak_memory_mb"] = round(tracemalloc.get_traced_memory()[1] / 1024 / 1024, 2)
        if not was_tracing:
            tracemalloc.stop()


# --------------------------------------------------------------------------- #
# Scrape one paper / batch (chỉ tải + lưu archive, KHÔNG extract)
# --------------------------------------------------------------------------- #
def scrape_paper(arxiv_id: str, num_versions: Optional[int] = None, state: Optional[dict] = None,
                 delay: float = REQUEST_DELAY, session: Optional[requests.Session] = None) -> dict:
    """Download + save every unfinished version of one paper, updating the state file
    after each version. Returns the paper's state entry."""
    state = state if state is not None else load_state()
    entry = state["papers"].setdefault(arxiv_id, {"versions": {}})
    if num_versions is None:
        num_versions = entry.get("num_versions") or get_num_versions(arxiv_id, delay=delay)
    entry["num_versions"] = num_versions
    done = [int(v) for v, i in entry["versions"].items() if i.get("status") in TERMINAL]

    try:
        with measure() as m:
            for version, data in get_source(arxiv_id, num_versions, skip_versions=done, delay=delay, session=session):
                if data is None:
                    info = {"status": "not_found", "kind": None, "size_bytes": 0}
                    _version_dir(arxiv_id, version).mkdir(parents=True, exist_ok=True)
                else:
                    kind, usable = identify_source(data)
                    _write_source(arxiv_id, version, data, kind, usable)
                    info = {"status": "done" if usable else "no_tex", "kind": kind, "size_bytes": len(data)}
                entry["versions"][str(version)] = info
                save_state(state)  # checkpoint after EVERY version
                log.info("%s v%d: %s", arxiv_id, version, info)
    finally:
        # runs also on 403/429/Ctrl+C; stats are cumulative over resumed runs
        st = entry.setdefault("stats", {"elapsed_sec": 0.0, "sleep_sec": 0.0, "active_sec": 0.0,
                                        "peak_memory_mb": 0.0, "runs": 0})
        for k in ("elapsed_sec", "sleep_sec", "active_sec"):
            st[k] = round(st[k] + m.get(k, 0.0), 2)
        st["peak_memory_mb"] = max(st["peak_memory_mb"], m.get("peak_memory_mb", 0.0))
        st["runs"] += 1
        entry["completed"] = _is_complete(entry)
        save_state(state)
    return entry


def scrape_batch(papers: Union[Iterable[str], Dict[str, Optional[int]]], delay: float = REQUEST_DELAY) -> dict:
    """Workflow: scrape -> save archive -> record sizes["archive_only"]. Does NOT extract.

    papers: list of arXiv ids, or {arxiv_id: num_versions | None}.
    Stops cleanly (state kept) on 403 / persistent 429; rerun the same call to resume.
    """
    items = papers.items() if isinstance(papers, dict) else ((p, None) for p in papers)
    state = load_state()
    session = requests.Session()
    for arxiv_id, nv in items:
        if state["papers"].get(arxiv_id, {}).get("completed"):
            log.info("%s: already completed, skipping", arxiv_id)
            continue
        try:
            scrape_paper(arxiv_id, nv, state=state, delay=delay, session=session)
            record_sizes(arxiv_id, "archive_only", state)
            save_state(state)
        except (BlockedError, RateLimitError) as e:
            log.error("Stopping batch: %s. Run again later to resume.", e)
            save_state(state)
            break
        except Exception:
            log.exception("%s: failed, will be retried next run", arxiv_id)
            save_state(state)
    return state

def process_paper(arxiv_id: str, state: Optional[dict] = None, keep_archive: Optional[bool] = None) -> dict:
    """Paper đã tải xong: extract .tex/.bib và ghi sizes["full"] + sizes["tex_bib"]. Không tải gì."""
    keep = KEEP_ARCHIVE if keep_archive is None else keep_archive
    own = state is None
    state = load_state() if own else state
    entry = state["papers"].get(arxiv_id)
    if not entry or not entry.get("completed"):
        return {"completed": False}

    old = entry.get("sizes", {}).get("full", {}).get("versions", {})
    full_versions = {int(k): v for k, v in old.items()}  # key là str nếu đọc từ JSON
    for v, vdir in _version_dirs(arxiv_id, None):
        archive = _find_archive(vdir)
        if archive is None:  # đã xử lý ở lần trước, hoặc version không có tex/bib
            continue
        archive_bytes = archive.stat().st_size
        _, full_bytes, full_disk = _extract_archive(vdir, archive, only_tex_bib=True)
        full_versions[v] = {"archive_bytes": archive_bytes, "extracted_bytes": full_bytes,
                            "total_bytes": archive_bytes + full_bytes,
                            "archive_disk_bytes": _disk(archive_bytes), "extracted_disk_bytes": full_disk,
                            "total_disk_bytes": _disk(archive_bytes) + full_disk}
        if not keep:
            archive.unlink()

    if full_versions:
        record_sizes(arxiv_id, "full", state, snapshot=_with_totals(full_versions))
    record_sizes(arxiv_id, "tex_bib", state)
    if own:
        save_state(state)
    return {"completed": True, "sizes": entry["sizes"]}
# --------------------------------------------------------------------------- #
# Notebook workflow
# --------------------------------------------------------------------------- #
# scrape -> lưu archive -> với mỗi version: MỘT lượt đọc archive vừa tính kích thước "full" (không ghi hình ra đĩa)
# vừa extract .tex/.bib -> xóa file thừa -> (xóa archive nếu keep_archive=False).
# Ghi vào source_state.json: sizes["archive_only"], sizes["full"], sizes["tex_bib"].
def ipynb_workflow(papers, delay: float = REQUEST_DELAY, keep_archive: Optional[bool] = None) -> dict:
    """papers: list id hoặc {id: num_versions | None} (mặc định PAPERS).
    keep_archive: giữ lại file nén sau cùng hay không (mặc định theo KEEP_ARCHIVE).

    Chạy lại an toàn: bài đã tải xong thì không tải lại; version đã mất archive (keep_archive=False) thì giữ nguyên số liệu cũ.
    Trả về {id: {"completed", "stats", "archive_only", "full", "tex_bib", "saved_by_dropping_non_tex_bytes"}}
    (kích thước là totals của bài, đơn vị byte).
    """
    keep = KEEP_ARCHIVE if keep_archive is None else keep_archive
    ids = list(papers)

    state = scrape_batch(papers, delay=delay)  # chỉ tải + lưu archive

    summary = {}
    for arxiv_id in ids:
        entry = state["papers"].get(arxiv_id, {})
        if not entry.get("completed"):
            log.warning("%s: chưa tải xong (bị dừng hoặc lỗi), bỏ qua extract. Chạy lại để tiếp tục.", arxiv_id)
            summary[arxiv_id] = {"completed": False, "stats": entry.get("stats")}
            continue

        full_versions: Dict[int, Dict[str, int]] = {}
        for v, vdir in _version_dirs(arxiv_id, None):
            archive = _find_archive(vdir)
            if archive is None:  # đã xử lý ở lần chạy trước, hoặc version không có tex/bib
                continue
            archive_bytes = archive.stat().st_size
            _, full_bytes, full_disk = _extract_archive(vdir, archive, only_tex_bib=True)
            _prune(vdir)  # dọn file thừa nếu trước đó đã extract full
            full_versions[v] = {"archive_bytes": archive_bytes, "extracted_bytes": full_bytes,
                                "total_bytes": archive_bytes + full_bytes,
                                "archive_disk_bytes": _disk(archive_bytes), "extracted_disk_bytes": full_disk,
                                "total_disk_bytes": _disk(archive_bytes) + full_disk}
            if not keep:
                archive.unlink()

        if full_versions:
            record_sizes(arxiv_id, "full", state, snapshot=_with_totals(full_versions))
        record_sizes(arxiv_id, "tex_bib", state)
        save_state(state)

        tot = {k: v["totals"] for k, v in entry["sizes"].items()}
        full_b = tot.get("full", {}).get("extracted_bytes")
        tex_b = tot.get("tex_bib", {}).get("extracted_bytes")
        full_d = tot.get("full", {}).get("extracted_disk_bytes")
        tex_d = tot.get("tex_bib", {}).get("extracted_disk_bytes")
        summary[arxiv_id] = {
            "completed": True,
            "stats": entry.get("stats"),
            "archive_only": tot.get("archive_only"),
            "full": tot.get("full"),
            "tex_bib": tot.get("tex_bib"),
            "saved_by_dropping_non_tex_bytes": (full_b - tex_b) if full_b is not None and tex_b is not None else None,
            "saved_by_dropping_non_tex_disk_bytes": (full_d - tex_d) if full_d is not None and tex_d is not None else None,
        }
        log.info("%s: full=%s B -> tex_bib=%s B", arxiv_id, full_b, tex_b)
    return summary

def generate_papers_dict(year: int, month: int, start_idx: int, end_idx: int) -> Dict[str, Optional[int]]:
    """Generate a dictionary of arXiv IDs for papers in a specific year and month.

    Args:
        year (int): The year of the papers.
        month (int): The month of the papers.
        start_idx (int): The starting index of the papers to include.
        end_idx (int): The ending index of the papers to include.

    Returns:
        Dict[str, Optional[int]]: A dictionary where keys are arXiv IDs and values are None.
    """
    papers_dict = {}
    for idx in range(start_idx, end_idx + 1):
        arxiv_id = f"{year}{month:02d}.{idx:05d}"
        papers_dict[arxiv_id] = None  # None indicates that num_versions will be determined later
    return papers_dict


if __name__ == "__main__":
    paper_dict = generate_papers_dict(year=26, month=8, start_idx=4007, end_idx=4012)  # Adjust the range as needed
    # print(paper_dict)
    # Giả định có num versions cho tất cả các bài báo, đặt giá trị là 1 để test (thực tế phải chạy metadata để lấy num_versions)
    for key in paper_dict:
        paper_dict[key] = 1

    
    # Kết quả performance và kích thước sẽ được lưu trong source_state.json
    ipynb_workflow(paper_dict, delay=REQUEST_DELAY, keep_archive=KEEP_ARCHIVE)