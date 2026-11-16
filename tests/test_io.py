"""Duyệt tệp, tìm bài và chữ ký checkpoint."""
import shutil

from conftest import write_paper
from src.parser_pipeline import discover_publications
from src.parser_reporting import input_signature
from src.utils import list_files

DOC = r"\documentclass{article}\begin{document}\section{Intro}\end{document}"


def test_list_files_matches_rglob(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    for name in ("x.tex", "a/y.bib", "a/b/z.tex", "a/b/.hidden"):
        (tmp_path / name).write_text("x")
    (tmp_path / "empty").mkdir()
    expected = sorted(p for p in tmp_path.rglob("*") if p.is_file())
    assert list_files(tmp_path) == expected


def test_discover_limit_applies_before_reading_metadata(tmp_path):
    for number in range(5):
        write_paper(tmp_path, f"2609.2783{number}", {1: {"main.tex": DOC}})
    # Metadata hỏng ở bài ngoài giới hạn không được đọc tới.
    (tmp_path / "2609-27834" / "metadata.json").write_text("{broken")
    jobs = discover_publications(tmp_path, limit=2)
    assert [job["arxiv_id"] for job in jobs] == ["2609.27830", "2609.27831"]
    assert all(not job["discovery_errors"] for job in jobs)
    assert discover_publications(tmp_path, limit=5)[-1]["discovery_errors"]


def test_discover_targets_keep_missing_ids(tmp_path):
    write_paper(tmp_path, "2609.27830", {1: {"main.tex": DOC}, 2: {"main.tex": DOC}})
    write_paper(tmp_path, "2609.27831", {1: {"main.tex": DOC}})
    jobs = discover_publications(tmp_path, target_ids=["2609.27830", "2609.27999"])
    assert [job["arxiv_id"] for job in jobs] == ["2609.27830", "2609.27999"]
    assert sorted(jobs[0]["version_dirs"]) == [1, 2]
    assert jobs[1]["paper_dir"] is None and jobs[1]["version_dirs"] == {1: None}


def test_signature_ignores_location_but_not_content(tmp_path):
    drive = tmp_path / "drive"
    write_paper(drive, "2609.27830", {1: {"main.tex": DOC}})
    local = tmp_path / "local"
    shutil.copytree(drive, local)
    options = {"enrich": False}
    from_drive = input_signature(discover_publications(drive)[0], options)
    assert input_signature(discover_publications(local)[0], options) == from_drive
    (local / "2609-27830" / "v1" / "main.tex").write_text(DOC + " ")
    assert input_signature(discover_publications(local)[0], options) != from_drive
