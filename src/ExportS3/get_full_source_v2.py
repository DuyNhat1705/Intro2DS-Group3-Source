"""Full-text source, v2: every version of a paper through export.arxiv.org/src, keeping only .tex/.bib.

What changes compared with get_full_source (v1), whose helpers this module reuses:
- Pacing is measured from the START of the previous request (Pacer), and one Pacer is shared by every
  arXiv call of a lane (API metadata and /src/), so the 3-second etiquette holds across modules.
- The GET is streamed: a PDF-only version is recognised from its Content-Type and its body (often
  several MB) is never downloaded.
- .tex/.bib are extracted straight from memory in one pass that also measures the size of the whole
  archive ("before removing figures"), so archives never land on disk (unless keep_archive=True).
- store_version() is shared with the S3 lane (s3_source), so a version read from the bulk bucket is
  stored and accounted for exactly like a version downloaded from export.
- HTTP 429 / 5xx: a few long in-module waits, then RateLimitError (the notebook cools down and retries).

The state file (DATA_DIR/source_state.json) keeps the v1 schema, so existing progress and the v1
analysis code keep working. v2 adds per version: `via` ("export" or "s3:<chunk>") and the sizes
(`full_bytes`, `tex_bib_bytes`, ...); `sizes` of a paper is filled once all its versions are finished.
"""
import gzip
import io
import logging
import shutil
import tarfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import requests

import get_full_source as gfs
from get_full_source import (ARCHIVE_NAME, TERMINAL, BlockedError, DownloadError, RateLimitError,
                             _disk, _looks_like_tex, _safe_rel, _version_dir, _wanted, _with_totals)

SRC_URL = gfs.SRC_URL  # export.arxiv.org, never arxiv.org
USER_AGENT = gfs.USER_AGENT
REQUEST_DELAY = 3.0                # seconds between the starts of two arXiv requests
RATE_LIMIT_WAITS = (30, 60, 120)   # waits after 429 / 5xx before retrying; then RateLimitError
TIMEOUT = (30, 300)                # (connect, read) seconds

# Guards every change to the shared state dict (export lane and S3 lane write different papers of it).
STATE_LOCK = threading.RLock()

log = logging.getLogger("get_full_source_v2")


# --------------------------------------------------------------------------- #
# Pacing
# --------------------------------------------------------------------------- #
class Pacer:
    """At least `interval` seconds between the starts of two requests to the same service."""

    def __init__(self, interval: float = REQUEST_DELAY):
        self.interval = interval
        self.last_start = 0.0
        self.slept = 0.0  # cumulative seconds spent waiting (pacing + back-off)
        self._lock = threading.Lock()

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


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
@dataclass
class Fetched:
    """One downloaded version. data is empty when the body was skipped (PDF) or really empty."""
    data: bytes
    size: int                 # archive size in bytes (Content-Length when the body was skipped)
    skipped_pdf: bool = False


def download_version(arxiv_id: str, version: int, pacer: Pacer,
                     session: Optional[requests.Session] = None) -> Optional[Fetched]:
    """GET /src/<id>v<version>. None if the version does not exist (404)."""
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
                    if "pdf" in r.headers.get("Content-Type", "").lower():
                        # PDF-only version: nothing to keep, do not download the body
                        return Fetched(b"", int(r.headers.get("Content-Length") or 0), skipped_pdf=True)
                    data = r.content
                    return Fetched(data, len(data))
                if r.status_code == 404:
                    return None
                if r.status_code == 403:
                    raise BlockedError(f"403 for {url}: blocked by arXiv, stop and retry later")
                if r.status_code != 429 and r.status_code < 500:
                    raise DownloadError(f"Unexpected HTTP {r.status_code} for {url}")
                try:
                    wait = float(r.headers.get("Retry-After", ""))
                except ValueError:
                    wait = RATE_LIMIT_WAITS[min(attempt, len(RATE_LIMIT_WAITS)) - 1]
                log.warning("%s: HTTP %d, attempt %d/%d", url, r.status_code, attempt, attempts)
        except requests.RequestException as e:  # includes a body cut off while streaming
            last_status = None
            wait = 10 * 2 ** attempt
            log.warning("%s: network error (%s), attempt %d/%d", url, e, attempt, attempts)
        if attempt < attempts:
            pacer.sleep(wait)
    exc = RateLimitError if last_status == 429 or (last_status or 0) >= 500 else DownloadError
    raise exc(f"Giving up on {url} (last status: {last_status})")


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
# State
# --------------------------------------------------------------------------- #
def save_state(state: dict) -> None:
    """gfs.save_state under STATE_LOCK (looked up at call time, so a notebook may replace it)."""
    with STATE_LOCK:
        gfs.save_state(state)


def paper_entry(state: dict, arxiv_id: str, num_versions: int) -> dict:
    with STATE_LOCK:
        entry = state["papers"].setdefault(arxiv_id, {"versions": {}})
        entry["num_versions"] = num_versions
        return entry


def _size_row(archive: int, extracted: int, extracted_disk: int) -> dict:
    return {"archive_bytes": archive, "extracted_bytes": extracted, "total_bytes": archive + extracted,
            "archive_disk_bytes": _disk(archive), "extracted_disk_bytes": extracted_disk,
            "total_disk_bytes": _disk(archive) + extracted_disk}


def sizes_from_versions(entry: dict) -> dict:
    """The v1 size snapshots (archive_only / full / tex_bib) from the numbers recorded per version.
    Only versions with TeX count, as in v1 (PDF-only versions keep an empty record)."""
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


def store_version(entry: dict, arxiv_id: str, version: int, fetched: Optional[Fetched], via: str,
                  keep_archive: bool = False) -> dict:
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
    info["via"] = via
    with STATE_LOCK:
        entry["versions"][str(version)] = info
        entry["completed"] = gfs._is_complete(entry)
        if entry["completed"]:
            entry["sizes"] = sizes_from_versions(entry)
    log.info("%s v%d via %s: %s", arxiv_id, version, via, {k: info[k] for k in ("status", "kind", "size_bytes")})
    return info


def _todo(entry: dict, arxiv_id: str, version: int) -> Optional[str]:
    """'download', 'adopt' (v1 left the archive on disk without extracting it) or None (finished)."""
    info = entry["versions"].get(str(version))
    if not info or info.get("status") not in TERMINAL:
        return "download"
    if (info["status"] == "done" and "full_bytes" not in info and "tex_bib" not in entry.get("sizes", {})
            and gfs._find_archive(_version_dir(arxiv_id, version)) is not None):
        return "adopt"
    return None


def versions_left(entry: dict, arxiv_id: str, versions: Iterable[int]) -> List[int]:
    return [v for v in versions if _todo(entry, arxiv_id, v)]


# --------------------------------------------------------------------------- #
# One paper through export
# --------------------------------------------------------------------------- #
def scrape_paper(arxiv_id: str, num_versions: int, versions: Optional[Iterable[int]] = None,
                 state: Optional[dict] = None, pacer: Optional[Pacer] = None,
                 session: Optional[requests.Session] = None, keep_archive: bool = False) -> dict:
    """Download through export every unfinished version in `versions` (default v1..v<num_versions>),
    storing each one and checkpointing the state after it. Returns the paper's state entry.

    Raises BlockedError (403), RateLimitError (429 / 5xx after the in-module waits) or DownloadError;
    versions finished before the error stay finished, so calling again resumes.
    """
    state = state if state is not None else gfs.load_state()
    pacer = pacer or Pacer()
    session = session or requests.Session()
    entry = paper_entry(state, arxiv_id, num_versions)
    wanted = range(1, num_versions + 1) if versions is None else versions
    t0, slept0 = time.perf_counter(), pacer.slept
    try:
        for v in wanted:
            todo = _todo(entry, arxiv_id, v)
            if todo == "adopt":
                archive = gfs._find_archive(_version_dir(arxiv_id, v)).read_bytes()
                fetched = Fetched(archive, len(archive))
                via = entry["versions"][str(v)].get("via", "export")
            elif todo == "download":
                fetched, via = download_version(arxiv_id, v, pacer, session), "export"
            else:
                continue
            store_version(entry, arxiv_id, v, fetched, via, keep_archive)
            save_state(state)  # checkpoint after every version
    finally:
        elapsed, slept = time.perf_counter() - t0, pacer.slept - slept0
        with STATE_LOCK:
            st = entry.setdefault("stats", {"elapsed_sec": 0.0, "sleep_sec": 0.0, "active_sec": 0.0,
                                            "peak_memory_mb": 0.0, "runs": 0})
            st["elapsed_sec"] = round(st["elapsed_sec"] + elapsed, 2)
            st["sleep_sec"] = round(st["sleep_sec"] + slept, 2)
            st["active_sec"] = round(st["active_sec"] + elapsed - slept, 2)
            st["runs"] += 1
        save_state(state)
    return entry


def store_fetched(arxiv_id: str, num_versions: int, version: int, fetched: Fetched, via: str,
                  keep_archive: bool = False, state: Optional[dict] = None) -> dict:
    """Store a version obtained elsewhere (S3 lane) and checkpoint the state."""
    state = state if state is not None else gfs.load_state()
    entry = paper_entry(state, arxiv_id, num_versions)
    info = store_version(entry, arxiv_id, version, fetched, via, keep_archive)
    save_state(state)
    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Smoke test: 2608.04007 has one version with TeX
    e = scrape_paper("2608.04007", 1)
    print(e["versions"], e.get("sizes", {}).get("tex_bib", {}).get("totals"))
