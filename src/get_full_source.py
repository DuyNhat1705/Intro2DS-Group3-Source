"""Download the full LaTeX sources (every version) of arXiv papers, keeping only .tex/.bib.

One version = one streamed GET of export.arxiv.org/src/<id>v<n>; its .tex/.bib files are extracted
straight from memory into data/<id>/v<n>/, so archives and figures never touch the disk:
- Pacer: at least REQUEST_DELAY seconds between the STARTS of two arXiv requests (the download time
  counts towards the gap instead of being added to it). One Pacer can be shared by every arXiv call of
  a process (metadata API and /src/), so the 3-second etiquette holds across modules. After a 429 / 5xx
  it slows down, and it returns to REQUEST_DELAY once requests succeed again.
- A PDF-only version is recognised from its Content-Type and its body is never downloaded.
- The same extraction pass measures the whole archive ("size before removing figures").

Progress is stored in STATE_FILE (default DATA_DIR/source_state.json) after every version, so a run
interrupted by rate limits / blocks / Ctrl+C resumes where it stopped. A notebook may point STATE_FILE
elsewhere or replace load_state / save_state: both are looked up at call time.
"""
import functools
import gzip
import io
import json
import logging
import math
import os
import queue
import re
import shutil
import sys
import tarfile
import threading
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterable, List, Optional, Union

import arxiv
import requests

from utils import DATA_DIR, get_paper_dir

# ====== EDIT HERE: papers to scrape when running this file directly ======
# {arxiv_id: num_versions}; None = read num_versions from data/<id>/metadata.json (get_metadata.py),
# falling back to the arxiv API if that file does not exist.
PAPERS = {
    "2608.04007": 1,
    "2608.04008": None,
    "2608.04009": None,
}
# =========================================
# True = also keep the raw archive next to the extracted .tex/.bib (costs disk).
KEEP_ARCHIVE = False

SRC_URL = "https://export.arxiv.org/src/{id}"  # never use arxiv.org (robot detection)
USER_AGENT = "Intro2DS-Group3-Scraper/1.0 (student project; contact: duynhata0@gmail.com)"
REQUEST_DELAY = 3.0                # seconds between the STARTS of two arXiv requests (etiquette: >= 3)
MAX_REQUEST_DELAY = 6.0            # upper bound of the delay after 429 / 5xx (= REQUEST_DELAY: never slow down)
RELAX_AFTER = 100                  # successful requests before the delay steps back towards REQUEST_DELAY
RATE_LIMIT_WAITS = (30, 60, 120)   # waits after 429 / 5xx before retrying; then RateLimitError
TIMEOUT = (30, 300)                # (connect, read) seconds
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

# Guards every change to the state dict (a notebook may flush it from another thread).
STATE_LOCK = threading.RLock()

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
# Pacing
# --------------------------------------------------------------------------- #
class Pacer:
    """At least `interval` seconds between the starts of two requests to the same service.

    throttled() (after a 429 / 5xx) adds one second, up to `max_interval`; after RELAX_AFTER
    successful requests in a row, succeeded() takes half a second off, down to the base interval.
    `events` counts the retries of download_version (rate_limited, server_error, cut, network_error).
    """

    def __init__(self, interval: float = REQUEST_DELAY, max_interval: float = MAX_REQUEST_DELAY):
        self.base = self.interval = interval
        self.max_interval = max(interval, max_interval)
        self.last_start = 0.0
        self.slept = 0.0  # cumulative seconds spent waiting (pacing + back-off)
        self.events = Counter()
        self._ok = 0
        self._lock = threading.Lock()

    def count(self, kind: str) -> None:
        with self._lock:
            self.events[kind] += 1

    def wait(self) -> None:
        with self._lock:
            pause = self.interval - (time.monotonic() - self.last_start)
            if pause > 0:
                time.sleep(pause)
                self.slept += pause
            self.last_start = time.monotonic()

    def sleep(self, seconds: float) -> None:
        """Back-off after an error, counted as sleep."""
        time.sleep(seconds)
        self.slept += seconds

    def call(self, fn: Callable, *args, **kwargs):
        """Pace a call that makes its own requests (arxiv.py): the gap restarts when it returns."""
        self.wait()
        try:
            return fn(*args, **kwargs)
        finally:
            with self._lock:
                self.last_start = time.monotonic()

    def throttled(self) -> None:
        with self._lock:
            self._ok = 0
            if self.interval < self.max_interval:
                self.interval = min(self.max_interval, self.interval + 1.0)
                log.warning("arXiv pacing slowed down to %.1fs", self.interval)

    def succeeded(self) -> None:
        with self._lock:
            self._ok += 1
            if self.interval > self.base and self._ok >= RELAX_AFTER:
                self.interval = max(self.base, self.interval - 0.5)
                self._ok = 0
                log.info("arXiv pacing back to %.1fs", self.interval)


_DEFAULT_PACER = Pacer()


# --------------------------------------------------------------------------- #
# Source type helpers
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


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
# Lấy số lượng version của bài báo (từ metadata.json nếu có, nếu không thì truy vấn API của arXiv).
def get_num_versions(arxiv_id: str, pacer: Optional[Pacer] = None) -> int:
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

    pacer = pacer or _DEFAULT_PACER
    # num_retries=0: retry của arxiv.py chỉ cách nhau vài giây, quá ngắn sau khi bị 429, nên tự retry.
    client = arxiv.Client(delay_seconds=pacer.interval, num_retries=0)
    attempts = len(RATE_LIMIT_WAITS) + 1
    for attempt in range(1, attempts + 1):
        try:
            result = pacer.call(lambda: next(client.results(arxiv.Search(id_list=[arxiv_id])), None))
        except arxiv.HTTPError as e:
            status = getattr(e, "status", None)
            if status != 429 and (status or 0) < 500:
                raise DownloadError(f"arXiv API HTTP {status} for {arxiv_id}") from e
            pacer.throttled()
            if attempt == attempts:
                raise RateLimitError(f"arXiv API still returns HTTP {status} for {arxiv_id}") from e
            log.warning("%s: arXiv API HTTP %s, attempt %d/%d", arxiv_id, status, attempt, attempts)
            pacer.sleep(RATE_LIMIT_WAITS[attempt - 1])
            continue
        pacer.succeeded()
        if result is None:
            raise DownloadError(f"{arxiv_id}: not found in arXiv API")
        m = re.search(r"v(\d+)$", result.entry_id)
        return int(m.group(1)) if m else 1
    raise RateLimitError(f"Could not get num_versions for {arxiv_id}")


@dataclass
class Fetched:
    """One downloaded version. data is empty when the body was skipped (PDF) or really empty."""
    data: bytes
    size: int                 # archive size in bytes (Content-Length when the body was skipped)
    skipped_pdf: bool = False


# Download một version cụ thể của bài báo. None nếu version không tồn tại (404).
def download_version(arxiv_id: str, version: int, pacer: Optional[Pacer] = None,
                     session: Optional[requests.Session] = None) -> Optional[Fetched]:
    """GET /src/<id>v<version>. None if the version does not exist (404)."""
    pacer = pacer or _DEFAULT_PACER
    session = session or requests.Session()
    url = SRC_URL.format(id=f"{arxiv_id}v{version}")
    attempts = len(RATE_LIMIT_WAITS) + 1
    last_status = None
    for attempt in range(1, attempts + 1):
        pacer.wait()
        try:
            with session.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, stream=True) as r:
                last_status = r.status_code
                if r.status_code == 200:
                    pacer.succeeded()
                    if "pdf" in r.headers.get("Content-Type", "").lower():
                        # PDF-only version: nothing to keep, do not download the body
                        return Fetched(b"", int(r.headers.get("Content-Length") or 0), skipped_pdf=True)
                    data = r.content
                    return Fetched(data, len(data))
                if r.status_code == 404:
                    pacer.succeeded()
                    return None
                if r.status_code == 403:
                    raise BlockedError(f"403 for {url}: blocked by arXiv, stop and retry later")
                if r.status_code != 429 and r.status_code < 500:
                    raise DownloadError(f"Unexpected HTTP {r.status_code} for {url}")
                pacer.count("rate_limited" if r.status_code == 429 else "server_error")
                pacer.throttled()
                try:
                    wait = float(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = RATE_LIMIT_WAITS[min(attempt, len(RATE_LIMIT_WAITS)) - 1]
                log.warning("%s: HTTP %d, attempt %d/%d", url, r.status_code, attempt, attempts)
        except requests.RequestException as e:
            last_status = None  # attempt này không có HTTP status; không để 429 cũ quyết định loại lỗi
            # Body bị cắt giữa chừng (archive lớn, mạng chậm): kết nối vẫn tốt, thử lại sớm.
            # Lỗi mạng khác (không kết nối được, timeout): chờ lâu dần.
            cut = isinstance(e, requests.exceptions.ChunkedEncodingError)
            pacer.count("cut" if cut else "network_error")
            wait = 5 * attempt if cut else 10 * 2 ** attempt
            log.warning("%s: network error (%s), attempt %d/%d", url, e, attempt, attempts)
        if attempt < attempts:
            pacer.sleep(wait)
    exc = RateLimitError if last_status == 429 or (last_status or 0) >= 500 else DownloadError
    raise exc(f"Giving up on {url} (last status: {last_status})")


# --------------------------------------------------------------------------- #
# Paths and sizes
# --------------------------------------------------------------------------- #
# Lấy đường dẫn đến thư mục của một version cụ thể của bài báo.
def _version_dir(arxiv_id: str, version: int) -> Path:
    return get_paper_dir(arxiv_id) / f"v{version}"


# Chuẩn hóa đường dẫn tương đối của file trong archive, loại bỏ các phần tử không hợp lệ (như "", ".", "..")
# và kiểm tra xem đường dẫn có phải là tuyệt đối hay không. Nếu hợp lệ, trả về PurePosixPath; nếu không, trả về None.
def _safe_rel(name: str) -> Optional[PurePosixPath]:
    p = PurePosixPath(name.replace("\\", "/"))
    parts = [x for x in p.parts if x not in ("", ".")]
    if not parts or p.is_absolute() or ".." in parts:
        return None
    return PurePosixPath(*parts)


# Archive còn nằm trong thư mục version (KEEP_ARCHIVE, hoặc dữ liệu của bản cũ chỉ tải mà chưa extract).
def _find_archive(vdir: Path) -> Optional[Path]:
    return next((vdir / n for n in ARCHIVE_NAME.values() if (vdir / n).exists()), None)


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


# Kích thước (byte) theo từng version; mỗi chỉ số có hai dạng: dung lượng logic và dung lượng chiếm trên đĩa (*_disk_bytes):
#   archive_bytes   - file nén lúc tải
#   extracted_bytes - các file giải nén (toàn bộ archive ở snapshot "full", chỉ .tex/.bib ở "tex_bib")
#   total_bytes     - archive + extracted
_SIZE_KEYS = ("archive_bytes", "extracted_bytes", "total_bytes",
              "archive_disk_bytes", "extracted_disk_bytes", "total_disk_bytes")


def _with_totals(versions: Dict[int, Dict[str, int]]) -> dict:
    totals = {k: sum(x[k] for x in versions.values()) for k in _SIZE_KEYS}
    return {"versions": versions, "totals": totals}


def _size_row(archive: int, extracted: int, extracted_disk: int) -> dict:
    return {"archive_bytes": archive, "extracted_bytes": extracted, "total_bytes": archive + extracted,
            "archive_disk_bytes": _disk(archive), "extracted_disk_bytes": extracted_disk,
            "total_disk_bytes": _disk(archive) + extracted_disk}


def sizes_from_versions(entry: dict) -> dict:
    """Size snapshots of a paper (archive_only / full / tex_bib) from the numbers recorded per version.
    Only versions with TeX count (PDF-only versions keep an empty record)."""
    stages = {"archive_only": {}, "full": {}, "tex_bib": {}}
    for v, info in entry["versions"].items():
        ok = info.get("status") == "done"
        archive = info.get("size_bytes", 0) if ok else 0
        stages["archive_only"][int(v)] = _size_row(archive, 0, 0)
        stages["full"][int(v)] = _size_row(archive, info.get("full_bytes", 0) if ok else 0,
                                           info.get("full_disk_bytes", 0) if ok else 0)
        row = _size_row(0, info.get("tex_bib_bytes", 0) if ok else 0, info.get("tex_bib_disk_bytes", 0) if ok else 0)
        row["archive_bytes"], row["archive_disk_bytes"] = archive, _disk(archive)  # size at download time
        stages["tex_bib"][int(v)] = row
    return {stage: _with_totals(rows) for stage, rows in stages.items()}


# --------------------------------------------------------------------------- #
# Extraction (in memory, one pass)
# --------------------------------------------------------------------------- #
@dataclass
class Extracted:
    kind: str                                   # tar.gz, tar, gz, zip, tex, pdf, docx, other
    files: List[str] = field(default_factory=list)
    full_bytes: int = 0                         # every file of the archive (before removing figures)
    full_disk_bytes: int = 0
    tex_bib_bytes: int = 0                      # the .tex/.bib files kept
    tex_bib_disk_bytes: int = 0


def extract_tex_bib(data: bytes, vdir: Path) -> Extracted:
    """Identify the archive in `data` and write only its .tex/.bib files into `vdir` (paths kept)."""
    out = Extracted("other")

    def keep(rel, src) -> None:
        dest = vdir.joinpath(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(src, f, 1024 * 1024)
        n = dest.stat().st_size
        out.tex_bib_bytes += n
        out.tex_bib_disk_bytes += _disk(n)
        out.files.append(rel.as_posix())

    def member(name: str, size: int, opener) -> None:
        rel = _safe_rel(name)
        if rel is None:  # unsafe path: neither written nor counted
            return
        out.full_bytes += size
        out.full_disk_bytes += _disk(size)
        if _wanted(rel.name):
            with opener() as src:
                keep(rel, src)

    def single_file(content: bytes) -> None:
        """A bare (possibly gzipped) file: keep it as main.tex if it is LaTeX."""
        out.full_bytes, out.full_disk_bytes = len(content), _disk(len(content))
        if _looks_like_tex(content):
            keep(_safe_rel("main.tex"), io.BytesIO(content))

    if data[:4] == b"%PDF":
        out.kind = "pdf"
        return out
    if data[:2] == b"\x1f\x8b":
        try:
            tf = tarfile.open(fileobj=io.BytesIO(data), mode="r:gz")
        except tarfile.ReadError:  # a gzipped single file, not a tar
            tf = None
        if tf is not None:
            out.kind = "tar.gz"
            with tf:
                for m in tf:
                    if m.isfile():  # skips symlinks and devices
                        member(m.name, m.size, lambda m=m: tf.extractfile(m))
            return out
        inner = gzip.decompress(data)  # corrupted gzip raises: retried later instead of an empty record
        if inner[:4] == b"%PDF":
            out.kind = "pdf"
        else:
            out.kind = "gz"
            single_file(inner)
        return out
    if data[:2] == b"PK":
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            return out
        with zf:
            names = zf.namelist()
            if not any(_wanted(n) for n in names):
                out.kind = "docx" if "word/document.xml" in names else "other"
                return out
            out.kind = "zip"
            for info in zf.infolist():
                if not info.is_dir():
                    member(info.filename, info.file_size, lambda i=info: zf.open(i))
        return out
    if tarfile.is_tarfile(io.BytesIO(data)):
        out.kind = "tar"
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tf:
            for m in tf:
                if m.isfile():
                    member(m.name, m.size, lambda m=m: tf.extractfile(m))
        return out
    single_file(data)
    out.kind = "tex" if out.files else "other"
    return out


# --------------------------------------------------------------------------- #
# State (resume)
# --------------------------------------------------------------------------- #
# Đọc STATE_FILE để lấy trạng thái các bài đã tải. File hỏng hoặc chưa có thì trả về trạng thái rỗng.
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.error("Corrupted state file %s, starting fresh", STATE_FILE)
    return {"papers": {}}


# Ghi trạng thái (JSON gọn, không thụt lề: file nhỏ hơn, ghi nhanh hơn trên Drive).
def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    with STATE_LOCK:
        text = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, STATE_FILE)  # atomic: never leaves a half-written state


# Kiểm tra xem tất cả các version của bài báo đã hoàn thành hay chưa
def _is_complete(entry: dict) -> bool:
    n = entry.get("num_versions")
    vs = entry.get("versions", {})
    return bool(n) and all(vs.get(str(v), {}).get("status") in TERMINAL for v in range(1, n + 1))


def paper_entry(state: dict, arxiv_id: str, num_versions: int) -> dict:
    with STATE_LOCK:
        entry = state["papers"].setdefault(arxiv_id, {"versions": {}})
        entry["num_versions"] = num_versions
        return entry


def store_version(entry: dict, arxiv_id: str, version: int, fetched: Optional[Fetched],
                  keep_archive: bool = KEEP_ARCHIVE) -> dict:
    """Write one version into data/<id>/v<version>/ and record it in `entry` (a paper of the state).

    fetched=None means the version does not exist (404). An empty directory is the empty record
    of a version without TeX (PDF, docx, ...). Raises DownloadError if the archive is corrupted.
    """
    vdir = _version_dir(arxiv_id, version)
    if vdir.exists():  # leftovers of an interrupted attempt (only unfinished versions get here)
        shutil.rmtree(vdir)
    vdir.mkdir(parents=True)
    if fetched is None:
        info = {"status": "not_found", "kind": None, "size_bytes": 0}
    elif fetched.skipped_pdf:
        info = {"status": "no_tex", "kind": "pdf", "size_bytes": fetched.size}
    elif not fetched.data:  # e.g. a withdrawn version served as 0 bytes
        info = {"status": "no_tex", "kind": "empty", "size_bytes": 0}
    else:
        try:
            ex = extract_tex_bib(fetched.data, vdir)
        except Exception as e:
            shutil.rmtree(vdir, ignore_errors=True)
            raise DownloadError(f"{arxiv_id}v{version}: cannot extract ({e!r})") from e
        if keep_archive and ex.files and ex.kind in ARCHIVE_NAME:
            (vdir / ARCHIVE_NAME[ex.kind]).write_bytes(fetched.data)
        info = {"status": "done" if ex.files else "no_tex", "kind": ex.kind, "size_bytes": fetched.size,
                "full_bytes": ex.full_bytes, "full_disk_bytes": ex.full_disk_bytes,
                "tex_bib_bytes": ex.tex_bib_bytes, "tex_bib_disk_bytes": ex.tex_bib_disk_bytes,
                "n_files": len(ex.files)}
    with STATE_LOCK:
        entry["versions"][str(version)] = info
        entry["completed"] = _is_complete(entry)
        if entry["completed"]:
            entry["sizes"] = sizes_from_versions(entry)
    log.info("%s v%d: %s", arxiv_id, version, {k: info[k] for k in ("status", "kind", "size_bytes")})
    return info


def _todo(entry: dict, arxiv_id: str, version: int) -> Optional[str]:
    """'download', 'adopt' (an older run left the archive on disk without extracting it) or None (finished)."""
    info = entry["versions"].get(str(version))
    if not info or info.get("status") not in TERMINAL:
        return "download"
    if (info["status"] == "done" and "full_bytes" not in info and "tex_bib" not in entry.get("sizes", {})
            and _find_archive(_version_dir(arxiv_id, version)) is not None):
        return "adopt"
    return None


def versions_left(entry: dict, arxiv_id: str, versions: Iterable[int]) -> List[int]:
    return [v for v in versions if _todo(entry, arxiv_id, v)]


# --------------------------------------------------------------------------- #
# Background storing
# --------------------------------------------------------------------------- #
class Writer:
    """One background thread that extracts and stores downloaded versions, so the next download starts
    while the previous archive is still being written (useful for big archives and slow disks such as
    Google Drive). Jobs run in submission order; at most `max_pending` downloaded archives wait in memory
    (submit blocks when the queue is full)."""

    def __init__(self, max_pending: int = 3):
        self._queue = queue.Queue(maxsize=max(1, max_pending))
        self.busy_sec = 0.0  # time spent storing
        self.jobs = 0
        self._thread = threading.Thread(target=self._run, name="source-writer", daemon=True)
        self._thread.start()

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, fn: Callable, *args) -> None:
        self._queue.put((fn, args))

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                t0 = time.perf_counter()
                try:
                    job[0](*job[1])
                except Exception:  # jobs report their own errors; never let the thread die
                    log.exception("writer job failed")
                self.busy_sec += time.perf_counter() - t0
                self.jobs += 1
            finally:
                self._queue.task_done()

    def join(self) -> None:
        """Wait until every submitted job is done."""
        self._queue.join()

    def close(self) -> None:
        self.join()
        self._queue.put(None)
        self._thread.join()


# --------------------------------------------------------------------------- #
# Scrape one paper / batch
# --------------------------------------------------------------------------- #
def scrape_paper(arxiv_id: str, num_versions: Optional[int] = None, state: Optional[dict] = None,
                 pacer: Optional[Pacer] = None, session: Optional[requests.Session] = None,
                 keep_archive: bool = KEEP_ARCHIVE, writer: Optional[Writer] = None,
                 on_finished: Optional[Callable[[dict, Optional[Exception]], None]] = None) -> dict:
    """Download every unfinished version v1..v<num_versions> and keep its .tex/.bib, checkpointing the
    state after each version. Returns the paper's state entry (`completed` once every version is finished).

    Without a writer each version is stored before the next download, and on_finished(entry, None) is
    called before returning. With a writer, versions are stored on the writer thread while the next ones
    download: the function returns once every version is DOWNLOADED, and on_finished(entry, error) runs on
    the writer thread once they are all stored (error = the first storing error, e.g. a corrupted archive).

    Raises BlockedError (403), RateLimitError (429 / 5xx after the in-module waits) or DownloadError
    (also for a corrupted archive when there is no writer); on_finished is not called then. Versions
    finished before the error stay finished, so calling again resumes.
    """
    state = state if state is not None else load_state()
    pacer = pacer or _DEFAULT_PACER
    session = session or requests.Session()
    if num_versions is None:
        with STATE_LOCK:
            num_versions = state["papers"].get(arxiv_id, {}).get("num_versions")
        num_versions = num_versions or get_num_versions(arxiv_id, pacer)
    entry = paper_entry(state, arxiv_id, num_versions)
    errors: List[Exception] = []

    def store(v: int, fetched: Optional[Fetched]) -> None:
        try:
            store_version(entry, arxiv_id, v, fetched, keep_archive)
            save_state(state)  # checkpoint after every version
        except Exception as e:
            if writer is None:
                raise
            log.error("%s v%d: %r", arxiv_id, v, e)
            errors.append(e)

    def finish() -> None:
        if on_finished is not None:
            on_finished(entry, errors[0] if errors else None)

    t0, slept0 = time.perf_counter(), pacer.slept
    try:
        for v in range(1, num_versions + 1):
            todo = _todo(entry, arxiv_id, v)
            if todo == "adopt":
                archive = _find_archive(_version_dir(arxiv_id, v)).read_bytes()
                fetched = Fetched(archive, len(archive))
            elif todo == "download":
                fetched = download_version(arxiv_id, v, pacer, session)
            else:
                continue
            if writer is None:
                store(v, fetched)
            else:
                writer.submit(store, v, fetched)
    except BaseException:
        if writer is not None:
            writer.join()  # lưu xong các version đã tải trước khi báo lỗi, để lần gọi lại không tải trùng
        raise
    finally:
        # runs also on 403/429/Ctrl+C; stats are cumulative over resumed runs
        elapsed, slept = time.perf_counter() - t0, pacer.slept - slept0
        with STATE_LOCK:
            st = entry.setdefault("stats", {"elapsed_sec": 0.0, "sleep_sec": 0.0, "active_sec": 0.0, "runs": 0})
            st["elapsed_sec"] = round(st["elapsed_sec"] + elapsed, 2)
            st["sleep_sec"] = round(st["sleep_sec"] + slept, 2)
            st["active_sec"] = round(st["active_sec"] + elapsed - slept, 2)
            st["runs"] = st.get("runs", 0) + 1
            entry["completed"] = _is_complete(entry)
        save_state(state)
    if writer is None:
        finish()
    else:
        writer.submit(finish)
    return entry


def scrape_batch(papers: Union[Iterable[str], Dict[str, Optional[int]]], pacer: Optional[Pacer] = None,
                 keep_archive: bool = KEEP_ARCHIVE) -> dict:
    """scrape_paper for many papers with one pacer and one HTTP session.

    papers: list of arXiv ids, or {arxiv_id: num_versions | None}.
    Stops cleanly (state kept) on 403 / persistent 429; rerun the same call to resume.
    """
    items = papers.items() if isinstance(papers, dict) else ((p, None) for p in papers)
    state = load_state()
    pacer = pacer or _DEFAULT_PACER
    session = requests.Session()
    for arxiv_id, nv in items:
        if state["papers"].get(arxiv_id, {}).get("completed"):
            log.info("%s: already completed, skipping", arxiv_id)
            continue
        try:
            scrape_paper(arxiv_id, nv, state=state, pacer=pacer, session=session, keep_archive=keep_archive)
        except (BlockedError, RateLimitError) as e:
            log.error("Stopping batch: %s. Run again later to resume.", e)
            break
        except Exception:
            log.exception("%s: failed, will be retried next run", arxiv_id)
    return state


if __name__ == "__main__":
    # Kết quả (trạng thái từng version, kích thước trước/sau khi bỏ hình, thời gian) nằm trong STATE_FILE
    final = scrape_batch(PAPERS)
    for pid in PAPERS:
        e = final["papers"].get(pid, {})
        print(pid, "completed" if e.get("completed") else "unfinished",
              e.get("sizes", {}).get("tex_bib", {}).get("totals"))
