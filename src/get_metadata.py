import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
import requests

try:
    from src.utils import DATA_DIR, get_paper_dir
except ImportError:
    from utils import DATA_DIR, get_paper_dir

# log hiển thị thời gian thực và mức độ thông báo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# giới hạn request và thông số retry
MIN_DELAY_SECONDS = 3.0
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


_ARXIV_CLIENT = arxiv.Client(
    page_size=100,
    delay_seconds=MIN_DELAY_SECONDS,
    num_retries=1,
)

# Lưu lại thời điểm gọi API gần nhất để kiểm soát delay giữa các request
_last_request_time: float = 0.0


def _respect_rate_limit(min_delay: float = MIN_DELAY_SECONDS) -> None:
    global _last_request_time
    elapsed = time.time() - _last_request_time
    if elapsed < min_delay:
        time.sleep(min_delay - elapsed)
    _last_request_time = time.time()


def _strip_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id.strip())


def _extract_version_num(entry_id_or_url: str) -> int:
    match = re.search(r"v(\d+)$", entry_id_or_url.strip())
    return int(match.group(1)) if match else 1


def _fetch_arxiv_results_with_retry(
    id_list: List[str],
    progress_prefix: str = "",
) -> List[arxiv.Result]:
    search = arxiv.Search(id_list=id_list)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            _respect_rate_limit()
            return list(_ARXIV_CLIENT.results(search))

        except arxiv.HTTPError as e:
            status_code = getattr(e, "status", None)
            wait_time = MIN_DELAY_SECONDS * (BACKOFF_FACTOR ** attempt)

            # Kiểm tra mã lỗi HTTP
            if status_code == 429:
                logger.warning(
                    f"{progress_prefix}[HTTP 429] Quá giới hạn request khi tải {id_list}. "
                    f"Thử lại ({attempt}/{MAX_RETRIES}) sau {wait_time:.1f}s..."
                )
            elif status_code == 503:
                logger.warning(
                    f"{progress_prefix}[HTTP 503] Server arXiv đang bận khi tải {id_list}. "
                    f"Thử lại ({attempt}/{MAX_RETRIES}) sau {wait_time:.1f}s..."
                )
            elif status_code in RETRYABLE_STATUS_CODES:
                logger.warning(
                    f"{progress_prefix}[HTTP {status_code}] Lỗi server khi tải {id_list}. "
                    f"Thử lại ({attempt}/{MAX_RETRIES}) sau {wait_time:.1f}s..."
                )
            else:
                # Các lỗi 400, 404 thì báo lỗi và dừng luôn không retry
                logger.error(f"{progress_prefix}[HTTP {status_code}] Lỗi khi tải {id_list}: {e}")
                raise

            if attempt == MAX_RETRIES:
                logger.error(f"{progress_prefix}Quá số lần thử lại cho {id_list} (HTTP {status_code}).")
                raise
            time.sleep(wait_time)

        except (requests.exceptions.RequestException, ConnectionError, TimeoutError) as e:
            # lỗi mất kết nối mạng/timeout giữa chừng
            wait_time = MIN_DELAY_SECONDS * (BACKOFF_FACTOR ** attempt)
            logger.warning(
                f"{progress_prefix}[Lỗi mạng] {e}. Thử lại ({attempt}/{MAX_RETRIES}) sau {wait_time:.1f}s..."
            )
            if attempt == MAX_RETRIES:
                logger.error(f"{progress_prefix}Thất bại sau {MAX_RETRIES} lần thử cho {id_list}.")
                raise
            time.sleep(wait_time)

        except Exception as e:
            wait_time = MIN_DELAY_SECONDS * (BACKOFF_FACTOR ** attempt)
            logger.warning(
                f"{progress_prefix}[Lỗi] {e}. Thử lại ({attempt}/{MAX_RETRIES}) sau {wait_time:.1f}s..."
            )
            if attempt == MAX_RETRIES:
                raise
            time.sleep(wait_time)

    return []


def get_entry_discovery(
    arXiv_id: str,
    current_idx: Optional[int] = None,
    total_count: Optional[int] = None,
) -> Dict[str, Any]:
    base_id = _strip_version(arXiv_id)

    progress_prefix = f"[{current_idx}/{total_count}] " if (current_idx and total_count) else ""

    logger.info(f"{progress_prefix}Đang lấy metadata cho {base_id}...")


    latest_results = _fetch_arxiv_results_with_retry([base_id], progress_prefix=progress_prefix)
    if not latest_results:
        raise ValueError(f"{progress_prefix}Không tìm thấy bài báo: {base_id}")

    latest_paper = latest_results[0]
    
    # Hậu tố vN ở entry_id của bản mới nhất chính là tổng số version của bài báo
    num_versions = _extract_version_num(latest_paper.entry_id)


    # Lấy submission_date và revised_dates
    if num_versions == 1:
        submission_date = latest_paper.published.date().isoformat() if latest_paper.published else ""
        revised_dates: List[str] = []
    else:
        # Nếu có nhiều version, gom toàn bộ ID từ v1 -> vN vào 1 lần gọi API để lấy ngày của từng bản
        version_ids = [f"{base_id}v{v}" for v in range(1, num_versions + 1)]
        logger.info(f"{progress_prefix}Bài {base_id} có {num_versions} versions, đang lấy lịch sử ngày...")
        version_results = _fetch_arxiv_results_with_retry(version_ids, progress_prefix=progress_prefix)

        # Sắp xếp lại các version theo thứ tự
        version_results_sorted = sorted(
            version_results,
            key=lambda r: _extract_version_num(r.entry_id),
        )


        # Trích xuất ngày của từng version sang YYYY-MM-DD
        version_dates: List[str] = []
        for res in version_results_sorted:
            dt = res.updated or res.published
            if dt:
                version_dates.append(dt.date().isoformat())

        if version_dates:
            # Bản v1 là ngày nộp ban đầu, từ v2 trở đi là các ngày chỉnh sửa (revised_dates)
            submission_date = version_dates[0]
            revised_dates = version_dates[1:]
        else:
            # Fallback nếu API không trả đủ danh sách version
            submission_date = latest_paper.published.date().isoformat() if latest_paper.published else ""
            revised_dates = (
                [latest_paper.updated.date().isoformat()]
                if (latest_paper.updated and latest_paper.updated != latest_paper.published)
                else []
            )

    # lấy venue, không có thì để None 
    venue: Optional[str] = latest_paper.journal_ref if latest_paper.journal_ref else None


    clean_title = re.sub(r"\s+", " ", latest_paper.title).strip() if latest_paper.title else ""
    clean_abstract = re.sub(r"\s+", " ", latest_paper.summary).strip() if latest_paper.summary else ""


    metadata: Dict[str, Any] = {
        "arxiv_id": base_id,
        "title": clean_title,
        "authors": [author.name for author in latest_paper.authors],
        "submission_date": submission_date,
        "revised_dates": revised_dates,
        "venue": venue,
        "num_versions": num_versions,
        "categories": list(latest_paper.categories) if latest_paper.categories else [],
        "abstract": clean_abstract,
    }

    logger.info(
        f"{progress_prefix}Xong {base_id} (v{num_versions}, ngày nộp: {submission_date}, sửa: {len(revised_dates)} lần)."
    )
    return metadata


def save_metadata(arXiv_id: str, metadata: Dict[str, Any]) -> Path:
    base_id = _strip_version(arXiv_id)
   
    paper_dir = get_paper_dir(base_id)
    paper_dir.mkdir(parents=True, exist_ok=True)

    output_path = paper_dir / "metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

    return output_path


def process_batch_metadata(
    arxiv_ids: List[str],
    skip_existing: bool = True,
    state_file: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = len(arxiv_ids)
    papers_table: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0
    skipped_count = 0


    if state_file is None:
        state_file = DATA_DIR / "metadata_crawl_state.json"

    logger.info(f"Bắt đầu cào metadata cho {total} bài báo...")

    for idx, raw_id in enumerate(arxiv_ids, start=1):
        base_id = _strip_version(raw_id)
        
        progress_str = f"[{idx}/{total} - {(idx / total) * 100:.1f}%]"

        paper_dir = get_paper_dir(base_id)
        metadata_file = paper_dir / "metadata.json"

        # Kiểm tra lưu state: nếu bài này đã có file metadata.json hợp lệ thì đọc lên luôn, không gọi API lại
        if skip_existing and metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
                if existing_meta.get("arxiv_id") and existing_meta.get("title"):
                    papers_table.append(existing_meta)
                    skipped_count += 1
                    success_count += 1
                    logger.info(f"{progress_str} Bỏ qua {base_id} (đã có metadata.json).")
                    continue
            except Exception as e:
                logger.warning(f"{progress_str} File cũ của {base_id} lỗi ({e}), tải lại...")

        
        try:
            metadata = get_entry_discovery(base_id, current_idx=idx, total_count=total)
            save_metadata(base_id, metadata)
            papers_table.append(metadata)
            success_count += 1

            # Cập nhật file state sau mỗi bài thành công để biết lần chạy gần nhất dừng ở đâu
            state_data = {
                "last_processed_id": base_id,
                "current_index": idx,
                "total": total,
                "success_count": success_count,
                "fail_count": fail_count,
                "skipped_count": skipped_count,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            with open(state_file, "w", encoding="utf-8") as sf:
                json.dump(state_data, sf, ensure_ascii=False, indent=4)

        except Exception as e:
            fail_count += 1
            logger.error(f"{progress_str} Thất bại bài {base_id}: {e}")

    papers_table_path = DATA_DIR / "papers.json"
    with open(papers_table_path, "w", encoding="utf-8") as pf:
        json.dump(papers_table, pf, ensure_ascii=False, indent=4)

    logger.info(
        f"Hoàn tất: {success_count}/{total} bài "
        f"(Tải mới: {success_count - skipped_count}, Có sẵn: {skipped_count}, Lỗi: {fail_count})."
    )
    return papers_table


if __name__ == "__main__":
    START_ID = "2310.00001"
    END_ID = "2310.00050"
    id_list = generate_arxiv_ids(START_ID, END_ID)

    results = process_batch_metadata(id_list, skip_existing=True)
