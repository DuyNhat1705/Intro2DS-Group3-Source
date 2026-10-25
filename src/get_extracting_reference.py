"""Semantic‑Scholar reference extractor.

Typical usage from a notebook or another script::

    from src.get_extracting_reference import (
        get_extracting_references,
        save_extracting_references,
        load_done,
        mark_done,
    )

    arxiv_id = "2103.00020"
    if arxiv_id not in load_done():
        refs = get_extracting_references(arxiv_id)
        if refs is None:
            mark_done(arxiv_id, "not_found", 0)
        elif not refs:
            mark_done(arxiv_id, "no_refs", 0)
        else:
            save_extracting_references(arxiv_id, refs)
            mark_done(arxiv_id, "ok", len(refs))

The module provides a small CLI (``python -m src.get_extracting_reference``)
to process many IDs in batch mode.  No network activity happens at import
time.
"""

from __future__ import annotations

# utils import – works both as a package import and as a script import.
try:
    from .utils import DATA_DIR, get_paper_dir
except ImportError:  # pragma: no cover
    from utils import DATA_DIR, get_paper_dir

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

import requests

# --------------------------------------------------------------------------- #
# Public exception
# --------------------------------------------------------------------------- #
class QuotaExhausted(Exception):
    """Raised when the Semantic Scholar API repeatedly returns HTTP 429."""
    pass


# --------------------------------------------------------------------------- #
# Module‑level state
# --------------------------------------------------------------------------- #
_SESSION = requests.Session()
_SESSION.headers["User-Agent"] = "Intro2DS-Group3 (github.com/DuyNhat1705/Intro2DS-Group3-Source)"
_SESSION.headers["Accept"] = "application/json"
_api_key = os.getenv("S2_API_KEY")
if _api_key:
    _SESSION.headers["x-api-key"] = _api_key

_TIMEOUT_GET = 30  # seconds
_TIMEOUT_POST = 60  # seconds
_BASE_REF_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{id}/references"
_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

# throttle: remember the start time of the previous request
_last_req_start: float = 0.0
_MIN_INTERVAL = 3.0  # seconds between *starts* of requests


def _throttle() -> None:
    """Sleep if the time since the previous request start is < 3 s."""
    global _last_req_start
    now = time.monotonic()
    elapsed = now - _last_req_start
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_req_start = time.monotonic()


_VERSION_RE = re.compile(r"v\d+$")


def _normalize_id(arxiv_id: str) -> str:
    """Remove a trailing version suffix (e.g. ``v2``) from an arXiv identifier."""
    return _VERSION_RE.sub("", arxiv_id.strip())


def _request(
    method: str,
    url: str,
    *,
    params: Optional[Mapping[str, Any]] = None,
    json_body: Optional[Mapping[str, Any]] = None,
    timeout: int,
    ok_statuses: tuple[int, ...] = (200, 404),
) -> requests.Response:
    """Perform a request with throttling and retry handling."""
    retry_429 = 5
    backoff_429 = [10, 20, 40, 80]  # seconds
    retry_other = 3
    backoff_other = [1, 2, 4]       # seconds

    for attempt in range(max(retry_429, retry_other) + 1):
        _throttle()
        try:
            if method == "GET":
                resp = _SESSION.get(url, params=params, timeout=timeout)
            else:  # POST
                resp = _SESSION.post(url, params=params, json=json_body, timeout=timeout)
        except (requests.Timeout, requests.RequestException) as exc:
            if attempt < retry_other:
                time.sleep(backoff_other[min(attempt, len(backoff_other) - 1)])
                continue
            raise RuntimeError(f"Network error after retries: {exc}") from exc

        # Successful or not‑found response
        if resp.status_code in ok_statuses:
            return resp

        # 429 – rate‑limit handling
        if resp.status_code == 429:
            if attempt >= retry_429:
                raise QuotaExhausted("Semantic Scholar quota exhausted")
            retry_after = resp.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = int(retry_after)
            else:
                wait = backoff_429[min(attempt, len(backoff_429) - 1)]
            time.sleep(wait)
            continue

        # 5xx – transient server errors
        if 500 <= resp.status_code < 600:
            if attempt < retry_other:
                time.sleep(backoff_other[min(attempt, len(backoff_other) - 1)])
                continue
            raise RuntimeError(f"Server error {resp.status_code}: {resp.text}")

        # Anything else is fatal
        raise RuntimeError(f"Unexpected status {resp.status_code}: {resp.text}")

    # Should never reach here
    raise RuntimeError("Exhausted retry loops without a decisive response")


def _map_ref(arxiv_id: str, cited: Mapping[str, Any]) -> Mapping[str, Any]:
    """Convert a Semantic Scholar ``citedPaper`` dict to the required output format."""
    return {
        "arxiv_id": arxiv_id,
        "ref_arxiv_id": (cited.get("externalIds") or {}).get("ArXiv"),
        "ref_title": cited.get("title"),
        "ref_authors": [
            a.get("name") for a in (cited.get("authors") or []) if a.get("name")
        ],
        "ref_submission_date": cited.get("publicationDate"),
        "ref_s2_id": cited.get("paperId"),
    }


def generate_ids(ranges: List[Tuple[str, int, int]]) -> Iterator[str]:
    """Yield unique arXiv IDs from ``ranges`` lazily.

    ``ranges`` is a list of ``(yymm, start, end)`` tuples, inclusive.
    IDs are formatted ``f"{yymm}.{n:05d}"``.  Duplicates are skipped.
    """
    seen: set[str] = set()
    for yymm, start, end in ranges:
        if not (isinstance(yymm, str) and len(yymm) == 4 and yymm.isdigit()):
            raise ValueError(f"Invalid YYMM '{yymm}'; must be 4 digits")
        if start > end:
            raise ValueError(f"Start ({start}) > end ({end})")
        for n in range(start, end + 1):
            cur = f"{yymm}.{n:05d}"
            if cur not in seen:
                seen.add(cur)
                yield cur


def get_extracting_references(arxiv_id: str) -> Optional[List[dict]]:
    """Fetch the reference list for *arxiv_id* via the single‑paper endpoint.

    Returns ``None`` for HTTP 404, otherwise a list (possibly empty) of
    reference dictionaries with exactly six keys.
    """
    arxiv_id = _normalize_id(arxiv_id)
    url = _BASE_REF_URL.format(id=arxiv_id)
    params = {
        "fields": "paperId,title,authors,publicationDate,externalIds",
        "limit": 1000,
    }
    results: List[dict] = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = _request(
            "GET", url, params=params, timeout=_TIMEOUT_GET, json_body=None
        )
        if resp.status_code == 404:
            return None
        data = resp.json()
        items = data.get("data", [])
        for item in items:
            cited = item.get("citedPaper")
            if not cited:
                continue
            if not cited.get("paperId"):
                continue
            results.append(_map_ref(arxiv_id, cited))

        nxt = data.get("next")
        if not nxt:
            break
        offset = int(nxt)

    return results


def _fetch_batch(ids: List[str]) -> dict[str, Optional[List[dict]]]:
    """Batch request for a list of (already‑normalized) arXiv IDs.

    Returns a mapping ``id -> references`` where ``references`` follows the
    same shape as :func:`get_extracting_references`.  ``None`` indicates a
    404 for that ID.
    """
    # Build payload
    payload = {"ids": [f"ARXIV:{_normalize_id(i)}" for i in ids]}
    params = {
        "fields": "references.paperId,references.title,references.authors,"
        "references.publicationDate,references.externalIds"
    }
    resp = _request(
        "POST",
        _BATCH_URL,
        params=params,
        json_body=payload,
        timeout=_TIMEOUT_POST,
        ok_statuses=(200, 400, 404, 413),
    )
    if resp.status_code == 404:
        raise RuntimeError("Batch endpoint returned 404; check URL or API status")

    if resp.status_code in (400, 413):
        if len(ids) == 1:
            return {ids[0]: get_extracting_references(ids[0])}
        mid = len(ids) // 2
        left = _fetch_batch(ids[:mid])
        right = _fetch_batch(ids[mid:])
        left.update(right)
        return left

    # At this point status should be 200
    if resp.status_code != 200:
        raise RuntimeError(f"Unexpected status {resp.status_code}: {resp.text}")

    # 404 already handled above

    payload_resp = resp.json()
    out: dict[str, Optional[List[dict]]] = {}
    for orig_id, entry in zip(ids, payload_resp):
        if entry is None:
            out[orig_id] = None
            continue
        refs_raw = entry.get("references") or []
        if len(refs_raw) >= 1000:
            # Fall back to paginated endpoint for full list
            out[orig_id] = get_extracting_references(orig_id)
            continue
        mapped = [_map_ref(_normalize_id(orig_id), r) for r in refs_raw if r and r.get("paperId")]
        out[orig_id] = mapped
    return out


def save_extracting_references(arxiv_id: str, references: List[dict]) -> None:
    """Atomically write ``references`` to ``<paper_dir>/references.json``."""
    arxiv_id = _normalize_id(arxiv_id)
    paper_dir = Path(get_paper_dir(arxiv_id))
    paper_dir.mkdir(parents=True, exist_ok=True)

    target = paper_dir / "references.json"
    tmp = target.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False)
    os.replace(tmp, target)


def load_done() -> set[str]:
    """Read ``refs_done.jsonl`` and return a set of already‑processed IDs."""
    state_dir = Path(DATA_DIR) / ".state"
    state_file = state_dir / "refs_done.jsonl"
    done: set[str] = set()
    if not state_file.is_file():
        return done
    with open(state_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pid = obj.get("id")
                if isinstance(pid, str):
                    done.add(pid)
            except json.JSONDecodeError:
                continue
    return done


def mark_done(arxiv_id: str, status: str, n_refs: int) -> None:
    """Append a line to ``refs_done.jsonl`` describing the processing result."""
    arxiv_id = _normalize_id(arxiv_id)
    state_dir = Path(DATA_DIR) / ".state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / "refs_done.jsonl"
    record = {"id": arxiv_id, "status": status, "n_refs": n_refs}
    with open(state_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_range(arg: str) -> Tuple[str, int, int]:
    """Parse ``YYMM:START-END`` into a tuple."""
    try:
        yymm, rest = arg.split(":")
        start_str, end_str = rest.split("-")
        start = int(start_str)
        end = int(end_str)
        return yymm, start, end
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Invalid range '{arg}': {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download reference lists from Semantic Scholar."
    )
    parser.add_argument(
        "--ranges",
        type=_parse_range,
        nargs="*",
        default=[
            ("2609", 17834, 22833),
            ("2609", 27834, 32833),
            ("2608", 4007, 9006),
            ("2608", 14007, 19006),
            ("2511", 6709, 11708),
        ],
        help="One or more YYMM:START-END specifications.",
    )
    parser.add_argument(
        "--max-new",
        type=int,
        default=1000,
        help="Maximum number of *new* papers to process this run (0 = unlimited).",
    )
    parser.add_argument(
        "--per-range",
        type=int,
        default=0,
        help="Maximum new papers per range (0 = ignore).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="How many IDs to fetch per batch request.",
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "single"],
        default="batch",
        help="Use the batch endpoint or fall back to single‑paper calls.",
    )
    args = parser.parse_args(argv)

    if args.mode == "single":
        args.batch_size = 1
    done = load_done()
    new_ids: list[str] = []
    stats = {"ok": 0, "no_refs": 0, "not_found": 0}
    start_time = time.time()
    consecutive_errors = 0

    try:
        for yymm, start, end in args.ranges:
            count_this_range = 0
            for arxiv_id in generate_ids([(yymm, start, end)]):
                if arxiv_id in done:
                    continue
                # Respect per‑range limit
                if args.per_range and count_this_range >= args.per_range:
                    continue
                new_ids.append(arxiv_id)
                count_this_range += 1
                # Stop when global max‑new reached
                if args.max_new and len(new_ids) >= args.max_new:
                    break
            if args.max_new and len(new_ids) >= args.max_new:
                break

        # Process in batches
        for i in range(0, len(new_ids), args.batch_size):
            batch_ids = new_ids[i : i + args.batch_size]
            try:
                if args.mode == "batch":
                    batch_res = _fetch_batch(batch_ids)
                else:
                    batch_res = {
                        pid: get_extracting_references(pid) for pid in batch_ids
                    }
            except QuotaExhausted as exc:
                print(f"Quota exhausted at batch starting with {batch_ids[0]!r}: {exc}",
                      file=sys.stderr)
                return 2
            except KeyboardInterrupt:
                print("\nInterrupted by user.", file=sys.stderr)
                return 130
            except Exception as exc:
                print(f"Error processing batch starting with {batch_ids[0]!r}: {exc}",
                      file=sys.stderr)
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    return 1
                continue  # skip this batch

            consecutive_errors = 0  # reset on success

            for pid in batch_ids:
                refs = batch_res.get(pid)
                if refs is None:
                    mark_done(pid, "not_found", 0)
                    stats["not_found"] += 1
                elif not refs:
                    mark_done(pid, "no_refs", 0)
                    stats["no_refs"] += 1
                else:
                    save_extracting_references(pid, refs)
                    mark_done(pid, "ok", len(refs))
                    stats["ok"] += 1

                # progress output every 500 new papers
                processed = sum(stats.values())
                if processed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = elapsed / processed if processed else 0
                    print(
                        f"Processed {processed} papers – "
                        f"{stats['ok']} ok, {stats['no_refs']} no_refs, "
                        f"{stats['not_found']} not_found – "
                        f"{rate:.2f}s per paper"
                    )
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        return 130

    total = sum(stats.values())
    elapsed = time.time() - start_time
    print(
        f"Finished. Total papers: {total} (ok={stats['ok']}, "
        f"no_refs={stats['no_refs']}, not_found={stats['not_found']}). "
        f"Elapsed {elapsed:.1f}s, avg {elapsed / max(total, 1):.2f}s per paper."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
