r"""Tests of multi-file gathering (prompt §5.1): roots, includes, conditionals, duplicates."""

from __future__ import annotations

from pathlib import Path

from src.m2.gather import gather_files
from src.m2.io_utils import scan_version_dirs

DOC = "\\documentclass{article}\n\\begin{document}\n%s\n\\end{document}\n"


def g(files, **kw):
    return gather_files(files, **kw)


def test_single_root_and_input():
    r = g({"main.tex": DOC % "Intro \\input{sec/a} end", "sec/a.tex": "AAA text.", "unused.tex": "Z"})
    assert r.status == "ok"
    assert r.root_file == "main.tex"
    assert r.compile_files == ["main.tex", "sec/a.tex"]
    assert "unused.tex" in r.redundant_files
    assert "AAA text." in r.body


def test_commented_include_is_not_followed():
    r = g({"main.tex": DOC % "% \\input{a}\nText \\%\\input{b}", "a.tex": "AAA", "b.tex": "BBB"})
    assert "a.tex" not in r.compile_files
    assert "b.tex" in r.compile_files  # \% is an escaped percent, the include is live
    assert "AAA" not in r.body


def test_input_without_braces_and_extension():
    r = g({"main.tex": DOC % "\\input sections/intro.tex\n\\input sections/b\n",
           "sections/intro.tex": "Intro text.", "sections/b.tex": "B text."})
    assert r.compile_files == ["main.tex", "sections/intro.tex", "sections/b.tex"]


def test_include_subfile_inputifexists():
    r = g({"main.tex": DOC % "\\include{ch1}\\subfile{ch2}\\InputIfFileExists{ch3}{}{}",
           "ch1.tex": "One.", "ch2.tex": "Two.", "ch3.tex": "Three."})
    assert r.compile_files == ["main.tex", "ch1.tex", "ch2.tex", "ch3.tex"]


def test_import_and_subimport_relative_to_current_file():
    r = g({"main.tex": DOC % "\\import{parts/}{p1}",
           "parts/p1.tex": "P1 \\subimport{sub/}{p2}",
           "parts/sub/p2.tex": "P2 deep."})
    assert r.compile_files == ["main.tex", "parts/p1.tex", "parts/sub/p2.tex"]
    assert "P2 deep." in r.body


def test_relative_to_current_folder_and_case_insensitive():
    r = g({"main.tex": DOC % "\\input{Chapters/One}", "chapters/one.tex": "x \\input{two}",
           "chapters/two.tex": "deep two"})
    assert r.compile_files == ["main.tex", "chapters/one.tex", "chapters/two.tex"]


def test_include_cycle_is_guarded():
    r = g({"main.tex": DOC % "\\input{a}", "a.tex": "A \\input{b}", "b.tex": "B \\input{a}"})
    assert r.compile_files == ["main.tex", "a.tex", "b.tex"]
    assert any("cycle" in n for n in r.notes)


def test_missing_include_recorded():
    r = g({"main.tex": DOC % "\\input{fig/plot.tikz} \\input{main.bbl} \\input{gone}"})
    assert set(r.missing_includes) == {"fig/plot.tikz", "main.bbl", "gone"}
    assert r.status == "ok"


def test_iffalse_and_newif_around_includes():
    body = ("\\iffalse \\input{old} \\fi\n"
            "\\iflong \\input{longv} \\else \\input{shortv} \\fi\n"
            "\\ifxetex \\input{xe} \\else \\input{pdf} \\fi\n")
    pre = "\\documentclass{article}\n\\newif\\iflong\n\\longtrue\n"
    r = g({"main.tex": pre + "\\begin{document}\n" + body + "\\end{document}",
           "old.tex": "OLD", "longv.tex": "LONG", "shortv.tex": "SHORT", "xe.tex": "XE",
           "pdf.tex": "PDF"})
    assert r.compile_files == ["main.tex", "longv.tex", "pdf.tex"]


def test_comment_environment_removed():
    r = g({"main.tex": DOC % "\\begin{comment}\\input{a}\\end{comment} ok", "a.tex": "A"})
    assert r.compile_files == ["main.tex"]


def test_several_roots_prefers_not_included_and_main():
    files = {
        "paper.tex": DOC % "\\input{body}",
        "supp.tex": "\\documentclass{article}\n\\begin{document}\nSupplement\n\\end{document}",
        "body.tex": "Body.",
        "fig.tex": "\\documentclass{standalone}\n\\begin{document}x\\end{document}",
    }
    r = g(files)
    assert r.root_file == "paper.tex"


def test_root_included_by_another_root_is_dropped():
    files = {"main.tex": DOC % "\\input{inner}",
             "inner.tex": "\\documentclass{article}\n\\begin{document}\nInner\\end{document}"}
    assert g(files).root_file == "main.tex"


def test_standalone_dropped():
    files = {"z.tex": DOC % "Real paper.",
             "a.tex": "\\documentclass[tikz]{standalone}\\begin{document}x\\end{document}"}
    assert g(files).root_file == "z.tex"


def test_documentstyle_root():
    r = g({"old.tex": "\\documentstyle[12pt]{article}\n\\begin{document}\nOld.\n\\end{document}"})
    assert r.root_file == "old.tex" and r.status == "ok"


def test_root_fallback_without_documentclass():
    r = g({"a.tex": "Just text.", "b.tex": "\\begin{document}\nHello.\n\\end{document}"})
    assert r.status == "root_fallback"
    assert r.root_file == "b.tex"


def test_duplicate_copy_equal_after_comments_counts_once():
    files = {"main.tex": DOC % "\\input{s}", "main(1).tex": (DOC % "\\input{s}") + "% only a comment",
             "s.tex": "S.", "s(1).tex": "S."}
    r = g(files)
    assert r.root_file == "main.tex"
    assert "main(1).tex" in r.redundant_files
    assert r.compile_files == ["main.tex", "s.tex"]
    assert any(n.startswith("dup_equal") for n in r.notes)


def test_duplicate_copy_different_uses_canonical_name():
    files = {"main.tex": DOC % "\\input{s}", "s.tex": "NEW", "s(1).tex": "OLD"}
    r = g(files)
    assert "NEW" in r.body and "OLD" not in r.body
    assert "s(1).tex" in r.redundant_files


def test_only_dup_copy_exists_resolves():
    r = g({"main.tex": DOC % "\\input{s}", "s(1).tex": "ONLY"})
    assert "ONLY" in r.body


def test_includeonly():
    pre = "\\documentclass{book}\n\\includeonly{c1}\n"
    r = g({"main.tex": pre + "\\begin{document}\\include{c1}\\include{c2}\\end{document}",
           "c1.tex": "C1", "c2.tex": "C2"})
    assert r.compile_files == ["main.tex", "c1.tex"]


def test_bibliography_resolution():
    r = g({"main.tex": DOC % "\\bibliography{refs,other}", "refs.bib": "", "x.bib": ""})
    assert r.bib_names == ["refs", "other"]


def test_preamble_body_split_and_source_map():
    r = g({"main.tex": "\\documentclass{article}\n\\title{T}\n\\begin{document}\nA\n\\input{b}\n\\end{document}",
           "b.tex": "line1\nline2 MARK\n"})
    assert "\\title{T}" in r.preamble
    off = r.text.find("MARK")
    f, line = r.locate(off)
    assert f == "b.tex" and line == 2


def test_version_dir_merge(tmp_path: Path):
    v1 = tmp_path / "v1"
    v1k = tmp_path / "v1(1)"
    (v1 / "sec").mkdir(parents=True)
    (v1k / "sec").mkdir(parents=True)
    (v1 / "main.tex").write_text("A", encoding="utf-8")
    (v1k / "main.tex").write_text("B", encoding="utf-8")
    (v1k / "sec" / "x.tex").write_text("X", encoding="utf-8")
    (v1 / "fig.png").write_bytes(b"\x89PNG")
    vf = scan_version_dirs([v1, v1k])
    assert sorted(vf.files) == ["main.tex", "sec/x.tex"]
    assert vf.files["main.tex"].parent == v1
    assert any(n.startswith("merged_dir") for n in vf.notes)


def test_empty_primary_dir_uses_copy(tmp_path: Path):
    v1 = tmp_path / "v1"
    v1k = tmp_path / "v1(1)"
    v1.mkdir()
    v1k.mkdir()
    (v1k / "main.tex").write_text("A", encoding="utf-8")
    vf = scan_version_dirs([v1, v1k])
    assert list(vf.files) == ["main.tex"]
