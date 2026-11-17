"""Kiểm thử tích hợp: parse một bài, chạy run_parser tuần tự/song song và timeout."""

import time
from unittest.mock import patch

import pyarrow.parquet as pq

from conftest import fake_split, write_paper
from src.parser_parallel import iter_parallel_results
from src.parser_pipeline import discover_publications, parse_publication
from src.parser_reporting import run_parser


def test_parse_publication_shares_sentence_and_merges_references(tmp_path):
    for version, title, key, fields in [
        (1, "Introduction", "old", "year={2017}"),
        (2, "Background", "new", "doi={10.1000/attention}"),
    ]:
        body = (
            r"\documentclass{article}\begin{document}"
            + rf"\section{{{title}}}We study attention \cite{{{key}}}."
            + r"\bibliography{refs}\end{document}"
        )
        bib = f"@article{{{key},title={{Attention Is All You Need}},author={{Vaswani, Ashish}},{fields}}}"
        if version == 1:
            paper = write_paper(tmp_path, "2301.00001", {1: {"main.tex": body, "refs.bib": bib}}, num_versions=3)
        else:
            (paper / "v2").mkdir()
            (paper / "v2" / "main.tex").write_text(body, encoding="utf-8")
            (paper / "v2" / "refs.bib").write_text(bib, encoding="utf-8")
    (paper / "v3").mkdir()
    with patch("src.hierarchy._sentence_model") as model:
        model.return_value.split.side_effect = fake_split
        result = parse_publication(discover_publications(tmp_path)[0], enrich=False)
    assert len(result["bibtex"]) == 1
    assert result["bibtex"][0]["year"] == 2017 and result["bibtex"][0]["doi"] == "10.1000/attention"
    sentences = [row for row in result["elements"] if row["kind"] == "sentence"]
    assert len(sentences) == 1
    edges = [row for row in result["hierarchy"] if row["element_id"] == sentences[0]["element_id"]]
    assert len(edges) == 2 and edges[0]["parent_id"] != edges[1]["parent_id"]
    assert any(row["version"] == 3 and row["status"] == "empty" for row in result["version_status"])
    # Entry được \cite có cờ cited; thông điệp không chứa đường dẫn tuyệt đối của máy.
    assert result["bibtex_debug"][0]["cited"] is True and result["stats"]["bibtex_cited"] == 1
    assert all(str(tmp_path) not in w["message"] for w in result["warnings"])
    assert all(w["category"] != "PyparsingDeprecationWarning" for w in result["warnings"])


# Không có văn xuôi nên tiến trình con không cần tải mô hình SaT.
STRUCTURE_ONLY = (
    r"\documentclass{article}\begin{document}\section{Method}"
    "\n"
    r"\begin{equation}a=b+%d\end{equation}"
    "\n"
    r"\end{document}"
)


def _corpus(base, count=4):
    for number in range(count):
        write_paper(
            base,
            f"2609.2784{number}",
            {1: {"main.tex": STRUCTURE_ONLY % number}, 2: {"main.tex": STRUCTURE_ONLY % (number + 10)}},
        )
    return discover_publications(base)


def _tables(result):
    return {name: pq.read_table(path).to_pylist() for name, path in result["tables"].items()}


def test_parallel_background_writer_matches_sequential(tmp_path):
    jobs = _corpus(tmp_path / "data")
    sequential = run_parser(jobs, tmp_path / "seq", num_workers=1, resume=False)
    parallel = run_parser(
        jobs, tmp_path / "par", num_workers=2, max_workers=2, adaptive_ram=False, resume=False, paper_timeout_sec=300
    )
    assert sequential["summary"]["run"]["stop_reason"] == "completed"
    assert parallel["summary"]["run"]["stop_reason"] == "completed"
    assert parallel["summary"]["papers_success"] == len(jobs)
    assert _tables(parallel) == _tables(sequential)
    # Chạy lại dùng checkpoint, không parse lại bài nào.
    again = run_parser(jobs, tmp_path / "par", num_workers=2, max_workers=2, adaptive_ram=False)
    assert again["summary"]["run"]["papers_reused"] == len(jobs)


def slow_or_quick_worker(connection, enrich, fuzzy_threshold):
    """Worker giả: bài 'slow' treo mãi, bài khác trả kết quả ngay."""
    while True:
        job = connection.recv()
        if job is None:
            return
        if job["arxiv_id"] == "slow":
            time.sleep(3600)
        connection.send(("result", {"done": job["arxiv_id"]}, 0))


def test_paper_timeout_stops_hung_worker_and_continues():
    jobs = [{"arxiv_id": "slow", "version_dirs": {1: None}}] + [
        {"arxiv_id": f"quick{i}", "version_dirs": {1: None}} for i in range(3)
    ]
    events = []
    started = time.perf_counter()
    # Thời hạn tính cả lúc khởi động tiến trình con (spawn + import), nên không đặt quá nhỏ.
    results = dict(
        iter_parallel_results(
            jobs,
            num_workers=2,
            adaptive_ram=False,
            job_timeout_sec=20,
            on_event=events.append,
            _worker_target=slow_or_quick_worker,
        )
    )
    assert time.perf_counter() - started < 120
    assert results[0]["stats"]["status"] == "error"
    assert "time limit" in results[0]["stats"]["message"]
    assert [results[i] for i in (1, 2, 3)] == [{"done": f"quick{i}"} for i in range(3)]
    summary = next(e for e in events if e["event"] == "parallel_summary")
    assert summary["job_timeouts"] == 1 and summary["worker_errors"] == 1
