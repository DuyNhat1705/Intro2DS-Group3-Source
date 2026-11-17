"""Checkpoint: kiểm tra theo kích thước hoặc băm đầy đủ, và bản chụp manifest."""

import json

import pytest

from conftest import write_paper
from src.parser_pipeline import discover_publications
from src.parser_reporting import _SNAPSHOT, _paper_name, run_parser

# Không có văn xuôi nên không cần mô hình SaT.
DOC = (
    r"\documentclass{article}\begin{document}\section{Method}"
    "\n"
    r"\begin{equation}a=b+%d\end{equation}"
    "\n"
    r"\end{document}"
)


def corpus(base, count=3):
    for number in range(count):
        write_paper(base, f"2609.2784{number}", {1: {"main.tex": DOC % number}})
    return discover_publications(base)


def run(jobs, out, **kwargs):
    return run_parser(jobs, out, worker="w", monitor_interval=0.05, **kwargs)["summary"]["run"]


def test_second_run_reuses_from_snapshot_without_state_files(tmp_path):
    jobs = corpus(tmp_path / "data")
    first = run(jobs, tmp_path / "out")
    assert first["checkpoint"]["not_reusable"] == 3 and (tmp_path / "out" / ".state" / "w" / _SNAPSHOT).is_file()
    second = run(jobs, tmp_path / "out")
    assert second["papers_reused"] == 3
    assert second["checkpoint"]["reused_from_snapshot"] == 3 and second["checkpoint"]["reused_from_state_files"] == 0
    assert second["checkpoint"]["sec"] >= 0


def test_interrupted_session_falls_back_to_state_files(tmp_path):
    jobs = corpus(tmp_path / "data")
    run(jobs, tmp_path / "out")
    # Phiên bị ngắt đột ngột không ghi lại bản chụp (bản chụp bị xóa lúc bắt đầu phiên).
    (tmp_path / "out" / ".state" / "w" / _SNAPSHOT).unlink()
    again = run(jobs, tmp_path / "out")
    assert again["papers_reused"] == 3 and again["checkpoint"]["reused_from_state_files"] == 3


def test_snapshot_keeps_papers_outside_the_current_run(tmp_path):
    jobs = corpus(tmp_path / "data")
    run(jobs, tmp_path / "out")
    run(jobs[:1], tmp_path / "out")
    third = run(jobs, tmp_path / "out")
    assert third["checkpoint"]["reused_from_snapshot"] == 3


@pytest.mark.parametrize("mode, reused", [("size", 3), ("full", 2)])
def test_same_size_corruption_is_caught_only_by_full_verification(tmp_path, mode, reused):
    jobs = corpus(tmp_path / "data")
    run(jobs, tmp_path / "out")
    partition = tmp_path / "out" / "tables" / "w" / "papers" / _paper_name(jobs[0]["arxiv_id"])
    key_map = partition / "key_map.json"
    original = key_map.read_bytes()
    key_map.write_bytes(original.replace(b"{", b"[", 1).replace(b"}", b"]", 1))  # cùng kích thước, khác nội dung
    assert key_map.stat().st_size == len(original) and key_map.read_bytes() != original
    assert run(jobs, tmp_path / "out", verify_checkpoints=mode)["papers_reused"] == reused


def test_size_change_is_caught_in_size_mode(tmp_path):
    jobs = corpus(tmp_path / "data")
    run(jobs, tmp_path / "out")
    partition = tmp_path / "out" / "tables" / "w" / "papers" / _paper_name(jobs[0]["arxiv_id"])
    with (partition / "warnings.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    assert run(jobs, tmp_path / "out")["papers_reused"] == 2


def test_old_record_without_sizes_is_fully_verified_then_upgraded(tmp_path):
    jobs = corpus(tmp_path / "data", count=1)
    run(jobs, tmp_path / "out")
    state = tmp_path / "out" / ".state" / "w"
    stamp = state / f"{_paper_name(jobs[0]['arxiv_id'])}.json"
    record = json.loads(stamp.read_text(encoding="utf-8"))
    del record["file_sizes"]
    stamp.write_text(json.dumps(record), encoding="utf-8")
    (state / _SNAPSHOT).unlink()
    assert run(jobs, tmp_path / "out")["papers_reused"] == 1
    snapshot = json.loads((state / _SNAPSHOT).read_text(encoding="utf-8"))["records"]
    assert set(snapshot[_paper_name(jobs[0]["arxiv_id"])]["file_sizes"]) >= {"elements.parquet", "warnings.jsonl"}


def test_data_preparation_and_invalid_mode(tmp_path):
    jobs = corpus(tmp_path / "data", count=1)
    info = run(jobs, tmp_path / "out", data_preparation={"data_source": "tar", "extract_sec": 12.5})
    assert info["data_preparation"] == {"data_source": "tar", "extract_sec": 12.5}
    with pytest.raises(ValueError):
        run(jobs, tmp_path / "out", verify_checkpoints="fast")
