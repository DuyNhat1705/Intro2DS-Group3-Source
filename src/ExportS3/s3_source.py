"""Bulk source lane: arXiv's requester-pays S3 bucket (s3://arxiv/src/).

The bucket stores the source of every paper in tar chunks (src/arXiv_src_YYMM_NNN.tar, ~500 MB each),
listed in src/arXiv_src_manifest.xml together with the first and last item of each chunk. A chunk is
read as a stream: nothing is written to disk except the .tex/.bib of the wanted papers, which are
stored with get_full_source_v2.store_fetched exactly like versions downloaded from export.

- The stream stops as soon as every wanted paper of the chunk has been seen.
- A broken or interrupted stream resumes at the last member boundary (HTTP Range), also in a later
  session (DATA_DIR/.state/s3_chunks.json), so no byte is paid twice.
- A wanted paper that the chunk does not contain is reported back as `missing` (export fetches it).

Requester pays: the AWS account owning the credentials pays the egress (~0.09 USD/GB). Credentials
are read the usual boto3 way, e.g. the environment variables AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY.
"""
import json
import logging
import os
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import utils

BUCKET = "arxiv"
MANIFEST_KEY = "src/arXiv_src_manifest.xml"
REGION = "us-east-1"
EGRESS_USD_PER_GB = 0.09  # S3 us-east-1 -> internet (first 10 TB / month)
STREAM_RETRIES = 5        # broken streams per chunk before giving up on it in this session
DENIED = {"AccessDenied", "403", "401", "InvalidAccessKeyId", "SignatureDoesNotMatch",
          "AllAccessDisabled", "ExpiredToken", "InvalidToken", "AccountProblem"}

# member names look like "2608/2608.04007.gz" (or .pdf); a version suffix is accepted if present
MEMBER_RE = re.compile(r"(\d{4}\.\d{4,5})(?:v(\d+))?\.[A-Za-z0-9.]+$")

log = logging.getLogger("s3_source")


class S3Unavailable(Exception):
    """No usable credentials or no permission: the lane stops and export takes over."""


def _state_dir() -> Path:
    return Path(utils.DATA_DIR) / ".state"


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def make_client():
    import boto3
    from botocore.config import Config
    return boto3.client("s3", region_name=REGION,
                        config=Config(retries={"max_attempts": 10, "mode": "standard"},
                                      connect_timeout=30, read_timeout=120))


def _call(fn, **kwargs):
    """Run one S3 call; credential / permission problems become S3Unavailable."""
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    try:
        return fn(**kwargs)
    except (NoCredentialsError, PartialCredentialsError) as e:
        raise S3Unavailable(f"no AWS credentials ({e})") from e
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code"))
        if code in DENIED:
            raise S3Unavailable(f"{code}: {e.response.get('Error', {}).get('Message')}") from e
        raise


def check_access(client) -> None:
    """One HEAD on the manifest (fractions of a cent); raises S3Unavailable when it is refused."""
    _call(client.head_object, Bucket=BUCKET, Key=MANIFEST_KEY, RequestPayer="requester")


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Chunk:
    filename: str      # S3 key, e.g. src/arXiv_src_2608_012.tar
    yymm: str
    first: int         # number of the first / last paper, e.g. 4007 for 2608.04007
    last: int
    size: int          # bytes
    timestamp: str     # when the chunk was built, as written in the manifest
    num_items: int


def _item_number(item: str, yymm: str) -> Optional[int]:
    m = re.fullmatch(rf"{yymm}\.(\d{{4,5}})(?:v\d+)?", (item or "").strip())
    return int(m.group(1)) if m else None


def parse_manifest(xml_bytes: bytes) -> List[Chunk]:
    chunks = []
    for f in ET.fromstring(xml_bytes).iter("file"):
        d = {c.tag: (c.text or "").strip() for c in f}
        yymm = d.get("yymm", "")
        first, last = _item_number(d.get("first_item"), yymm), _item_number(d.get("last_item"), yymm)
        if first is None or last is None:  # old-style ids (astro-ph0001001): never one of ours
            continue
        chunks.append(Chunk(d["filename"], yymm, first, last, int(d.get("size") or 0),
                            d.get("timestamp", ""), int(d.get("num_items") or 0)))
    return chunks


def load_manifest(client, max_age_days: float = 7.0) -> List[Chunk]:
    """Manifest from the bucket (a few MB), cached in DATA_DIR/.state for `max_age_days`."""
    cache = _state_dir() / "s3_manifest.xml"
    if not cache.exists() or time.time() - cache.stat().st_mtime > max_age_days * 86400:
        body = _call(client.get_object, Bucket=BUCKET, Key=MANIFEST_KEY, RequestPayer="requester")["Body"]
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_bytes(body.read())
        os.replace(tmp, cache)
    return parse_manifest(cache.read_bytes())


class ChunkIndex:
    """Which chunk holds a given arXiv id."""

    def __init__(self, chunks: List[Chunk]):
        self.by_name = {c.filename: c for c in chunks}
        self._months: Dict[str, Tuple[List[int], List[Chunk]]] = {}
        grouped = defaultdict(list)
        for c in chunks:
            grouped[c.yymm].append(c)
        for yymm, cs in grouped.items():
            cs.sort(key=lambda c: c.first)
            self._months[yymm] = ([c.first for c in cs], cs)

    def find(self, arxiv_id: str) -> Optional[Chunk]:
        yymm, num = arxiv_id.split(".")
        firsts, cs = self._months.get(yymm, ([], []))
        i = bisect_right(firsts, int(num)) - 1
        return cs[i] if i >= 0 and cs[i].first <= int(num) <= cs[i].last else None

    def latest_month(self) -> Optional[str]:
        return max(self._months) if self._months else None


# --------------------------------------------------------------------------- #
# Progress of chunks (resume across sessions)
# --------------------------------------------------------------------------- #
def load_chunk_state() -> dict:
    """{chunk filename: {status: partial|done, resume_at, bytes_read, seconds, stored, missing}}"""
    path = _state_dir() / "s3_chunks.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_chunk_state(state: dict) -> None:
    path = _state_dir() / "s3_chunks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def bytes_read_total(chunk_state: dict) -> int:
    return sum(c.get("bytes_read", 0) for c in chunk_state.values())


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
@dataclass
class Plan:
    work: Dict[str, Dict[str, int]] = field(default_factory=dict)  # chunk filename -> {arxiv_id: version}
    to_export: List[str] = field(default_factory=list)             # targets S3 will not serve
    gb_to_read: float = 0.0                                        # upper bound (whole remaining chunks)

    @property
    def n_papers(self) -> int:
        return sum(len(w) for w in self.work.values())

    @property
    def cost_usd(self) -> float:
        return self.gb_to_read * EGRESS_USD_PER_GB


def plan(index: ChunkIndex, targets: Dict[str, int], chunk_state: dict, max_gb: Optional[float] = None) -> Plan:
    """Split `targets` {arxiv_id: version wanted from S3} between chunks and export.

    Export gets the ids that are in no chunk (not on S3 yet), whose chunk was finished in an earlier
    session (the paper was missing or failed there), or that do not fit in the budget `max_gb`
    (total GB read from S3 over all sessions; chunks with the most wanted papers per GB go first).
    """
    out, by_chunk = Plan(), defaultdict(dict)
    for aid, version in targets.items():
        c = index.find(aid)
        if c is None or chunk_state.get(c.filename, {}).get("status") == "done":
            out.to_export.append(aid)
        else:
            by_chunk[c.filename][aid] = version
    budget = None if max_gb is None else max_gb * 1e9 - bytes_read_total(chunk_state)

    def remaining(name: str) -> int:
        return max(index.by_name[name].size - chunk_state.get(name, {}).get("resume_at", 0), 0)

    for name in sorted(by_chunk, key=lambda n: -len(by_chunk[n]) / max(remaining(n), 1)):
        need = remaining(name)
        if budget is not None and need > budget:
            out.to_export += list(by_chunk[name])
            continue
        if budget is not None:
            budget -= need
        out.work[name] = by_chunk[name]
        out.gb_to_read += need / 1e9
    out.work = dict(sorted(out.work.items()))  # read chunks in id order
    return out


# --------------------------------------------------------------------------- #
# Reading a chunk
# --------------------------------------------------------------------------- #
class _CountingReader:
    """File-like wrapper around the S3 body that counts the bytes read (= bytes paid)."""

    def __init__(self, body):
        self.body, self.count = body, 0

    def read(self, size: int = -1) -> bytes:
        data = self.body.read(None if size is None or size < 0 else size)
        self.count += len(data)
        return data


def _blocks(size: int) -> int:
    return (size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE * tarfile.BLOCKSIZE


@dataclass
class ChunkResult:
    chunk: str
    stored: List[str] = field(default_factory=list)   # stored through on_member
    missing: List[str] = field(default_factory=list)  # not in the chunk, or a different version
    failed: List[str] = field(default_factory=list)   # on_member raised (e.g. corrupted archive)
    bytes_read: int = 0
    seconds: float = 0.0
    finished: bool = False                            # False: stopped early, resumes next time


def process_chunk(client, chunk: Chunk, wanted: Dict[str, int],
                  on_member: Callable[[str, int, bytes], None], chunk_state: dict,
                  should_stop: Callable[[], bool] = lambda: False,
                  save: Callable[[dict], None] = save_chunk_state) -> ChunkResult:
    """Stream `chunk` and call on_member(arxiv_id, version, data) for every wanted paper.

    wanted: {arxiv_id: version expected}. A member name without version is taken to be that
    version; with a different version it counts as missing. Progress (resume offset, bytes read)
    is kept in chunk_state and persisted with `save` (default: s3_chunks.json) after the chunk
    and on every stream error.
    """
    st = chunk_state.setdefault(chunk.filename, {"status": "partial", "resume_at": 0, "bytes_read": 0,
                                                 "seconds": 0.0, "stored": 0, "missing": []})
    res, remaining = ChunkResult(chunk.filename), dict(wanted)
    t0, errors, exhausted = time.perf_counter(), 0, False
    while remaining and not exhausted and not should_stop():
        base = st["resume_at"]
        if base >= chunk.size:
            exhausted = True
            break
        reader = None
        try:
            body = _call(client.get_object, Bucket=BUCKET, Key=chunk.filename, RequestPayer="requester",
                         Range=f"bytes={base}-")["Body"]
            reader = _CountingReader(body)
            try:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    for m in tf:
                        hit = MEMBER_RE.search(m.name.rsplit("/", 1)[-1]) if m.isfile() else None
                        aid = hit.group(1) if hit else None
                        if aid in remaining:
                            data = tf.extractfile(m).read()  # a broken stream here keeps aid wanted
                            version = remaining.pop(aid)
                            if hit.group(2) and int(hit.group(2)) != version:
                                res.missing.append(aid)
                            else:
                                try:
                                    on_member(aid, version, data)
                                    res.stored.append(aid)
                                except Exception as e:  # one bad archive must not stop the chunk
                                    log.warning("%s: %s v%d failed: %r", chunk.filename, aid, version, e)
                                    res.failed.append(aid)
                        st["resume_at"] = base + m.offset_data + _blocks(m.size)  # next member header
                        if not remaining or should_stop():
                            break
                    else:
                        exhausted = True
            finally:
                body.close()
        except S3Unavailable:
            raise
        except tarfile.ReadError as e:
            if "empty file" in str(e):  # resumed exactly at the end-of-archive blocks
                exhausted = True
            else:
                errors = _stream_error(chunk, e, errors)
        except Exception as e:  # broken stream: resume from the last member boundary
            errors = _stream_error(chunk, e, errors)
        finally:
            if reader is not None:
                st["bytes_read"] += reader.count
                res.bytes_read += reader.count
            save(chunk_state)
    if exhausted:
        res.missing += list(remaining)
    res.finished = exhausted or not remaining
    res.seconds = time.perf_counter() - t0
    st["seconds"] = round(st.get("seconds", 0.0) + res.seconds, 2)
    st["stored"] = st.get("stored", 0) + len(res.stored)
    st["missing"] = sorted(set(st.get("missing", [])) | set(res.missing) | set(res.failed))
    if res.finished:
        st["status"] = "done"
    save(chunk_state)
    log.info("%s: stored %d, missing %d, failed %d, %.1f MB in %.0fs%s", chunk.filename, len(res.stored),
             len(res.missing), len(res.failed), res.bytes_read / 1e6, res.seconds,
             "" if res.finished else " (stopped, resumes later)")
    return res


def _stream_error(chunk: Chunk, e: Exception, errors: int) -> int:
    errors += 1
    if errors > STREAM_RETRIES:
        raise RuntimeError(f"{chunk.filename}: stream failed {errors} times, last: {e!r}") from e
    wait = min(120, 5 * 2 ** errors)
    log.warning("%s: stream error %r, resuming in %ds (%d/%d)", chunk.filename, e, wait, errors, STREAM_RETRIES)
    time.sleep(wait)
    return errors
