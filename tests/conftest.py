"""Cấu hình chung: cho phép `import src.*` và dọn các cache cấp module giữa các test."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(autouse=True)
def _clear_caches():
    from src import bibtex, hierarchy

    hierarchy._SAT_CACHE.clear()
    bibtex._enrich_with_crossref.cache_clear()
    bibtex._fallback_parse_with_llm.cache_clear()
    yield


def fake_split(text, **kwargs):
    """Thay SaT: mỗi đoạn là một câu, đủ để kiểm tra tích hợp mà không tải mô hình."""
    return [text]


def write_paper(base, arxiv_id, versions, num_versions=None):
    """Tạo thư mục bài theo bố cục Milestone 1: <id>/metadata.json và <id>/vN/*.tex."""
    import json

    paper = Path(base) / arxiv_id.replace(".", "-")
    paper.mkdir(parents=True)
    count = num_versions or len(versions)
    (paper / "metadata.json").write_text(json.dumps({"arxiv_id": arxiv_id, "num_versions": count}), encoding="utf-8")
    for number, files in versions.items():
        folder = paper / f"v{number}"
        folder.mkdir()
        for name, content in files.items():
            (folder / name).write_text(content, encoding="utf-8")
    return paper
