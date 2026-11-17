"""Chọn file chính khi thư mục có bản sao trùng kiểu `main(1).tex` (do Google Drive tạo)."""

import pytest

from src.gather_clean_tex import AmbiguousRootError, gather_tex

DOC = r"\documentclass{article}\begin{document}\section{Intro}Hello.\bibliography{refs}\end{document}"


def write(folder, files):
    for name, content in files.items():
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def test_identical_copies_pick_the_unsuffixed_file(tmp_path):
    write(tmp_path, {"paper(1).tex": DOC, "paper.tex": DOC, "paper(2).tex": DOC,
                     "refs.bib": "@misc{a,title={A}}", "refs(1).bib": "@misc{a,title={A}}"})
    result = gather_tex(tmp_path)
    assert result.root.name == "paper.tex"
    assert "identical copies ignored" in result.root_reason
    assert [p.name for p in result.bib_files] == ["refs.bib"]


def test_copies_only_differing_in_comments_count_as_identical(tmp_path):
    write(tmp_path, {"main.tex": DOC, "main(1).tex": "% old note\n" + DOC})
    assert gather_tex(tmp_path).root.name == "main.tex"


def test_copies_collapse_before_the_other_tie_breakers(tmp_path):
    # Hai tài liệu khác nhau, mỗi tài liệu có bản sao; luật "tên quen thuộc" vẫn chọn main.tex.
    other = DOC.replace("Hello.", "Supplementary material.")
    write(tmp_path, {"main.tex": DOC, "main(1).tex": DOC,
                     "supplement.tex": other, "supplement(1).tex": other})
    assert gather_tex(tmp_path).root.name == "main.tex"


def test_distinct_roots_stay_ambiguous(tmp_path):
    # Cùng độ dài để luật "file lớn nhất" cũng không phân xử được.
    write(tmp_path, {"a.tex": DOC, "b.tex": DOC.replace("Hello.", "Howdy.")})
    with pytest.raises(AmbiguousRootError):
        gather_tex(tmp_path)
