r"""Tests of macro expansion and hierarchy construction (prompt §5.2–§5.3)."""

from __future__ import annotations

from src.m2.pipeline import parse_version_files


def doc(body: str, pre: str = "") -> dict[str, str]:
    return {"main.tex": "\\documentclass{article}\n" + pre + "\n\\begin{document}\n" + body
            + "\n\\end{document}\n"}


def tree(body: str, pre: str = ""):
    return parse_version_files(doc(body, pre)).tree


def outline(node, depth=0, out=None):
    out = [] if out is None else out
    out.append((depth, node.kind, node.content))
    for c in node.children:
        outline(c, depth + 1, out)
    return out


def headings(t):
    return [(d, k, c) for d, k, c in outline(t)
            if k in ("part", "chapter", "section", "subsection", "subsubsection", "paragraph",
                     "subparagraph", "abstract")]


def kinds(t):
    return [k for _d, k, _c in outline(t)]


def test_root_and_title():
    t = tree("\\maketitle Hello world.", pre="\\title{My Great Paper\\thanks{x}}")
    assert t.kind == "document"
    assert t.content == "My Great Paper"


def test_section_alias_and_equation_alias():
    pre = "\\def\\sect{\\section}\n\\newcommand{\\beq}{\\begin{equation}}\n\\newcommand{\\eeq}{\\end{equation}}"
    t = tree("\\sect{Intro} We have \\beq a=b \\label{x}\\eeq and more.", pre)
    o = outline(t)
    assert (1, "section", "Intro") in o
    assert (2, "formula", "\\begin{equation}a=b\\end{equation}") in o


def test_skipped_levels():
    t = tree("\\subsection{A} Text a. \\subsubsection{B} Text b. \\section{C} Text c. "
             "\\paragraph{D} Text d.")
    h = headings(t)
    assert h == [(1, "subsection", "A"), (2, "subsubsection", "B"), (1, "section", "C"),
                 (2, "paragraph", "D")]


def test_section_short_long_title_and_cleanup():
    t = tree("\\section[Short]{The \\textbf{Long} Title\\label{sec:x}\\footnote{fn}} Body.")
    assert headings(t) == [(1, "section", "The Long Title")]


def test_starred_sections_and_text_before_first_heading():
    t = tree("Preface sentence. \\section*{Intro} Body.")
    o = outline(t)
    assert o[1] == (1, "sentence", "Preface sentence.")
    assert (1, "section", "Intro") in o


def test_abstract_env_and_command():
    t = tree("\\begin{abstract} We study X. It works. \\end{abstract} \\section{I} Body.")
    o = outline(t)
    assert o[1][1] == "abstract"
    assert (2, "sentence", "We study X.") in o
    t2 = parse_version_files({"main.tex": "\\documentclass{revtex4}\n\\abstract{Short abs.}\n"
                              "\\begin{document}\\section{I} Body.\\end{document}"}).tree
    assert outline(t2)[1][1] == "abstract"


def test_appendix_and_acknowledgements_included():
    t = tree("\\section{Intro} A. \\section*{Acknowledgments} We thank B. "
             "\\begin{acks} Funding C. \\end{acks} \\appendix \\section{Proofs} Proof D.")
    h = [c for _d, k, c in headings(t)]
    assert "Acknowledgments" in h and "Proofs" in h
    o = outline(t)
    assert any(c == "Funding C." for _d, _k, c in o)


def test_references_excluded():
    t = tree("\\section{Intro} A \\cite{k}. \\section*{References} \\begin{thebibliography}{9}"
             "\\bibitem{k} A. Author. Title. 2020. \\end{thebibliography}")
    h = [c for _d, _k, c in headings(t)]
    assert "References" not in h
    assert all("Author" not in c for _d, _k, c in outline(t))
    t2 = tree("\\section{Intro} A. \\bibliographystyle{plain}\\bibliography{refs}")
    assert all(k != "section" or c == "Intro" for _d, k, c in outline(t2))


def test_nested_lists():
    t = tree("\\section{S} Intro. \\begin{itemize} \\item First point. \\item Second. "
             "\\begin{enumerate} \\item Inner one. \\end{enumerate} \\end{itemize} After.")
    o = outline(t)
    lst = [x for x in o if x[1] == "list"]
    assert len(lst) == 2
    assert (4, "sentence", "First point.") in o
    assert (6, "sentence", "Inner one.") in o
    assert o[-1] == (2, "sentence", "After.")


def test_theorem_blocks_with_newtheorem():
    pre = "\\newtheorem{mythm}{Theorem}"
    t = tree("\\section{S} \\begin{mythm} Every x is y. \\begin{equation} x=y \\end{equation}"
             "\\end{mythm} \\begin{proof} Trivial. \\end{proof}", pre)
    o = outline(t)
    blocks = [x for x in o if x[1] == "block"]
    assert len(blocks) == 2
    assert (3, "sentence", "Every x is y.") in o
    assert (3, "formula", "\\begin{equation}x=y\\end{equation}") in o


def test_figure_and_table_content():
    t = tree("\\begin{figure}[htbp]\\centering\\includegraphics[width=3cm]{img/plot.pdf}"
             "\\caption{A \\emph{nice} plot.}\\label{f}\\end{figure}"
             "\\begin{figure}\\includegraphics{img/raw_fig.png}\\end{figure}"
             "\\begin{table}[t]\\caption{Results.}\\begin{tabular}{lc}\\toprule A & B \\\\ "
             "\\midrule 1 & 2 \\\\ \\bottomrule\\end{tabular}\\end{table}")
    o = outline(t)
    figs = [c for _d, k, c in o if k == "figure"]
    tabs = [c for _d, k, c in o if k == "table"]
    assert figs == ["A nice plot.", "raw_fig.png"]
    assert tabs == ["Results.\nA | B\n1 | 2"]


def test_formula_mid_paragraph_flushes_sentence():
    t = tree("\\section{S} We define the loss as \\[ L = \\sum_i x_i \\] where x is input. Done.")
    o = [x for x in outline(t) if x[1] in ("sentence", "formula")]
    assert [k for _d, k, _c in o] == ["sentence", "formula", "sentence", "sentence"]
    assert o[0][2] == "We define the loss as"
    assert o[2][2] == "where x is input."


def test_footnote_sentences_follow_host_sentence():
    t = tree("\\section{S} We use data\\footnote{Available online. See site.} from X. Next one.")
    s = [c for _d, k, c in outline(t) if k == "sentence"]
    assert s == ["We use data from X.", "Available online.", "See site.", "Next one."]


def test_verbatim_removed_from_prose():
    t = tree("\\section{S} Code follows. \\begin{verbatim} x = 1. Y = 2. \\end{verbatim} End.")
    s = [c for _d, k, c in outline(t) if k == "sentence"]
    assert s == ["Code follows.", "End."]


def test_ieee_parstart():
    t = tree("\\section{Intro} \\IEEEPARstart{T}{his} paper is new.")
    s = [c for _d, k, c in outline(t) if k == "sentence"]
    assert s == ["This paper is new."]


def test_unbalanced_brace_is_partial_not_crash():
    vp = parse_version_files(doc("\\section{Intro} Text \\textbf{never closed. More text."))
    assert vp.status in ("ok", "partial")
    assert any(k == "sentence" for _d, k, _c in outline(vp.tree))


def test_newenvironment_alias():
    pre = "\\newenvironment{myitems}{\\begin{itemize}}{\\end{itemize}}"
    t = tree("\\begin{myitems}\\item One.\\item Two.\\end{myitems}", pre)
    assert kinds(t).count("item") == 2
