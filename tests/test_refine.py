"""Hậu xử lý bibtex bằng AnyStyle, với runner giả lập (không cần Ruby)."""

import json
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import fake_split, write_paper
from src.bibtex_refine import refine_bibliography, suspicion
from src.parser_pipeline import discover_publications
from src.parser_reporting import _paper_name, run_parser, table_schemas


def row(key, title=None, authors=(), year=None, doi=None, journal=None, arxiv_id="2609.00001"):
    return dict(
        arxiv_id=arxiv_id,
        citation_key=key,
        entry_type="article",
        title=title,
        authors=list(authors),
        year=year,
        doi=doi,
        journal=journal,
        url=None,
    )


def make_output(tmp_path, rows, debug):
    """Bố cục run_parser tối thiểu: bảng gộp và bibtex_debug.jsonl của từng partition."""
    tables = tmp_path / "tables" / "w"
    tables.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=table_schemas()["bibtex"]), tables / "bibtex.parquet")
    for arxiv_id in {r["arxiv_id"] for r in rows}:
        partition = tables / "papers" / _paper_name(arxiv_id)
        partition.mkdir(parents=True)
        with (partition / "bibtex_debug.jsonl").open("w", encoding="utf-8") as stream:
            for r in rows:
                if r["arxiv_id"] == arxiv_id:
                    stream.write(json.dumps({**r, **debug.get(r["citation_key"], {})}) + "\n")
    return tables


@pytest.mark.parametrize(
    "r, reasons",
    [
        (row("k", "Advances in Neural Information Processing Systems", ["A. B"]), ["title_is_venue"]),
        (row("k", None, ["A. B"]), ["no_title"]),
        (row("k", "Fine title", ["A. B", "C. Hartvigsen. 2024. Are language models useful"]), ["author_too_long"]),
        (row("k", "Fine title", [], journal="X"), ["no_authors"]),
        (row("k", "Fine title", ["A. B"]), []),
    ],
)
def test_suspicion(r, reasons):
    assert suspicion(r) == reasons


ANYSTYLE = {
    "title": ["Deep evidential regression"],
    "author": [{"family": "Amini", "given": "A."}, {"family": "Rus", "given": "D."}],
    "date": ["2020"],
    "container-title": ["Advances in Neural Information Processing Systems"],
}
RAW = (
    r"A.~Amini and D.~Rus. Deep evidential regression. In \emph{Advances in Neural Information "
    r"Processing Systems}, 2020."
)


def test_anystyle_fixes_only_suspicious_bibitems_in_one_batch(tmp_path):
    tables = make_output(
        tmp_path,
        [
            row(
                "amini",
                "Advances in Neural Information Processing Systems",
                ["A. Amini", "D. Rus. Deep evidential regression. In"],
            ),
            row("bibfile", None, []),  # entry .bib: không có raw_text nên không gửi đi
            row("ok", "A good title", ["X. Y"]),
        ],
        {"amini": {"raw_text": RAW}, "ok": {"raw_text": "X. Y. A good title. 2020."}},
    )
    calls = []

    def runner(texts):
        calls.append(list(texts))
        return [ANYSTYLE for _ in texts]

    stats = refine_bibliography(tmp_path, worker="w", anystyle_runner=runner)
    out = {r["citation_key"]: r for r in pq.read_table(tables / "bibtex.parquet").to_pylist()}
    assert len(calls) == 1 and len(calls[0]) == 1 and "\\" not in calls[0][0]  # một lô, văn bản thuần
    assert out["amini"]["title"] == "Deep evidential regression"
    assert out["amini"]["authors"] == ["A. Amini", "D. Rus"] and out["amini"]["year"] == 2020
    assert out["amini"]["journal"] == "Advances in Neural Information Processing Systems"
    assert out["ok"]["title"] == "A good title" and out["bibfile"]["title"] is None
    assert stats["anystyle"]["candidates"] == 1 and stats["rows_changed"] == 1
    log = [json.loads(line) for line in (tables / "bibtex_refinement.jsonl").read_text().splitlines()]
    # Tên hội nghị từng bị lấy làm tiêu đề được chuyển sang journal.
    assert {(c["field"], c["source"]) for c in log} == {
        ("title", "anystyle"),
        ("journal", "anystyle"),
        ("authors", "anystyle"),
        ("year", "anystyle"),
    }


def test_rerun_uses_cache_and_original_table(tmp_path):
    tables = make_output(tmp_path, [row("amini", None, ["A. Amini"])], {"amini": {"raw_text": RAW}})
    calls = []

    def runner(texts):
        calls.append(texts)
        return [ANYSTYLE for _ in texts]

    refine_bibliography(tmp_path, worker="w", anystyle_runner=runner)
    refined = pq.read_table(tables / "bibtex.parquet").to_pylist()
    again = refine_bibliography(tmp_path, worker="w", anystyle_runner=runner)
    assert len(calls) == 1 and again["anystyle"]["cache_hits"] == 1
    assert pq.read_table(tables / "bibtex.parquet").to_pylist() == refined
    assert pq.read_table(tables / "bibtex_parsed.parquet").to_pylist()[0]["title"] is None


def test_anystyle_title_must_occur_in_source(tmp_path):
    make_output(tmp_path, [row("k", None, ["A. B"])], {"k": {"raw_text": "A. B. Something real. 2020."}})
    stats = refine_bibliography(
        tmp_path, worker="w", anystyle_runner=lambda texts: [{"title": ["Invented title"]} for _ in texts]
    )
    assert stats["rows_changed"] == 0


def test_missing_anystyle_command_is_reported_not_fatal(tmp_path):
    make_output(tmp_path, [row("k", None, ["A. B"])], {"k": {"raw_text": "A. B. Something. 2020."}})
    stats = refine_bibliography(tmp_path, worker="w", anystyle_command="no-such-anystyle-cmd")
    assert "not found" in stats["anystyle"]["skipped"] and stats["rows_changed"] == 0


DOC = (
    r"\documentclass{article}\begin{document}\section{Intro}See \cite{amini}."
    r"\begin{thebibliography}{1}\bibitem{amini} " + RAW + r"\end{thebibliography}\end{document}"
)


def test_end_to_end_after_run_parser(tmp_path):
    write_paper(tmp_path / "data", "2609.27830", {1: {"main.tex": DOC}})
    jobs = discover_publications(tmp_path / "data")
    with patch("src.hierarchy._sentence_model") as model:
        model.return_value.split.side_effect = fake_split
        run_parser(jobs, tmp_path / "out", worker="w", monitor_interval=0.05)
    debug = tmp_path / "out" / "tables" / "w" / "papers" / _paper_name("2609.27830") / "bibtex_debug.jsonl"
    assert "Deep evidential regression" in json.loads(debug.read_text(encoding="utf-8"))["raw_text"]
    stats = refine_bibliography(tmp_path / "out", worker="w", anystyle_runner=lambda texts: [ANYSTYLE] * len(texts))
    (out,) = pq.read_table(tmp_path / "out" / "tables" / "w" / "bibtex.parquet").to_pylist()
    # Parser đã tách đúng mục này, nên AnyStyle không có ứng viên và bảng giữ nguyên.
    assert stats["anystyle"]["candidates"] == 0 and out["title"] == "Deep evidential regression"
