r"""Tests of LaTeX cleanup and math normalization (prompt §5.4)."""

from __future__ import annotations

import pytest

from src.m2.clean import clean_inline
from src.m2.mathnorm import normalize_block, normalize_inline
from src.m2.pipeline import parse_version_files


def formulas(body: str) -> list[str]:
    files = {"main.tex": "\\documentclass{article}\n\\begin{document}\n" + body + "\n\\end{document}"}
    out = []

    def walk(n):
        if n.kind == "formula":
            out.append(n.content)
        for c in n.children:
            walk(c)
    walk(parse_version_files(files).tree)
    return out


EQ = "\\begin{equation}%s\\end{equation}"
AL = "\\begin{equation}\\begin{aligned}%s\\end{aligned}\\end{equation}"


@pytest.mark.parametrize("src, expected", [
    ("$$ a + b $$", EQ % "a + b"),
    ("\\[ a+b \\]", EQ % "a+b"),
    ("\\begin{equation} a=1 \\label{e1} \\end{equation}", EQ % "a=1"),
    ("\\begin{equation*} a=1 \\end{equation*}", EQ % "a=1"),
    ("\\begin{displaymath} x \\end{displaymath}", EQ % "x"),
    ("\\begin{dmath} y=2 \\end{dmath}", EQ % "y=2"),
    ("\\begin{align} a &= b \\nonumber \\\\ c &= d \\notag \\end{align}", AL % "a &= b \\\\ c &= d"),
    ("\\begin{align*} a &= b \\end{align*}", AL % "a &= b"),
    ("\\begin{gather} a \\\\ b \\end{gather}", AL % "a \\\\ b"),
    ("\\begin{multline} a \\\\ b \\end{multline}", AL % "a \\\\ b"),
    ("\\begin{eqnarray} a &=& b \\end{eqnarray}", AL % "a &=& b"),
    ("\\begin{flalign} a &= b \\end{flalign}", AL % "a &= b"),
    ("\\begin{alignat}{2} a &= b \\end{alignat}", AL % "a &= b"),
    ("\\begin{equation}\\begin{split} a &= b \\\\ &= c \\end{split}\\end{equation}",
     EQ % "\\begin{split} a &= b \\\\ &= c \\end{split}"),
])
def test_block_forms(src, expected):
    assert formulas("Text. " + src + " More.") == [expected]


def test_subequations_is_transparent():
    f = formulas("\\begin{subequations}\\begin{align} a&=b \\end{align}\\begin{equation} c \\end{equation}"
                 "\\end{subequations}")
    assert f == [AL % "a&=b", EQ % "c"]


def test_normalize_block_whitespace_and_braces():
    assert normalize_block("equation", "  a  =\n  {b}  ") == EQ % "a = {b}"
    assert normalize_block("align", "x &= 1") == AL % "x &= 1"


@pytest.mark.parametrize("src", ["\\(x^2\\)", "$x^2$", "\\begin{math}x^2\\end{math}", "$ x^2 $"])
def test_inline_forms(src):
    assert clean_inline("We have " + src + " here.") == "We have $x^2$ here."


def test_normalize_inline_strips_label():
    assert normalize_inline(" a \\label{x} ") == "$a$"


@pytest.mark.parametrize("src, expected", [
    ("\\centering Text", "Text"),
    ("A \\vspace{2mm} B", "A B"),
    ("A \\hspace*{1em} B", "A B"),
    ("\\noindent Hello", "Hello"),
    ("\\textbf{Bold} and \\emph{em} and \\textit{it} and \\texttt{tt}", "Bold and em and it and tt"),
    ("\\underline{u} \\mbox{m} \\textsc{sc}", "u m sc"),
    ("See \\href{http://x.org}{our site}.", "See our site."),
    ("At \\url{https://github.com/a/b_c}.", "At https://github.com/a/b_c."),
    ("Price 50\\% \\& more \\_ \\#1", "Price 50% & more _ #1"),
    ("``quoted'' text", '"quoted" text'),
    ("pages 1--2 --- yes", "pages 1–2 — yes"),
    ("Fig.~1", "Fig. 1"),
    ("{\\bf old} {\\it style}", "old style"),
    ("\\resizebox{\\linewidth}{!}{Inner}", "Inner"),
    ("\\unknowncmd{kept} \\unknownswitch gone", "kept gone"),
    ("a\\\\b", "a b"),
    ("\\label{x}Text", "Text"),
    ("\\textcolor{red}{Red}", "Red"),
])
def test_formatting_removal(src, expected):
    assert clean_inline(src) == expected


@pytest.mark.parametrize("src, expected", [
    ("Erd\\H{o}s", "Erdős"),
    ("Schr\\\"odinger", "Schrödinger"),
    ("{\\\"o}", "ö"),
    ("\\'e\\`a\\^o", "éàô"),
    ("\\v{c}", "č"),
    ("\\c{c}", "ç"),
    ("\\'{\\i}", "í"),
    ("\\~n", "ñ"),
    ("\\ss{} \\o \\aa", "ß ø å"),
])
def test_accents(src, expected):
    assert clean_inline(src) == expected


@pytest.mark.parametrize("src, expected", [
    ("\\citep{a,b}", "\\cite{a,b}"),
    ("\\citet[p.~3]{a}", "\\cite{a}"),
    ("\\citep[see][]{a}", "\\cite{a}"),
    ("\\citealp{ a , b }", "\\cite{a,b}"),
    ("\\citeauthor{a}", "\\cite{a}"),
    ("\\parencite{a}", "\\cite{a}"),
    ("\\textcite{a}", "\\cite{a}"),
    ("\\autocite[12]{a}", "\\cite{a}"),
    ("\\cite*{a}", "\\cite{a}"),
    ("\\cite{a}", "\\cite{a}"),
    ("\\eqref{eq:1}", "\\ref{eq:1}"),
    ("\\autoref{s}", "\\ref{s}"),
    ("\\cref{a,b}", "\\ref{a,b}"),
    ("\\Cref{a}", "\\ref{a}"),
])
def test_cite_and_ref_normalization(src, expected):
    assert clean_inline(src) == expected


def test_no_residual_inline_delimiters():
    out = clean_inline("A \\(x\\) and \\begin{math}y\\end{math} and $z$.")
    assert "\\(" not in out and "\\begin{math}" not in out
    assert out == "A $x$ and $y$ and $z$."
