r"""Tests of BibTeX extraction, normalization and reference deduplication (prompt §5.6–§5.7)."""

from __future__ import annotations

import pytest

from src.m2.bib_dedup import dedup_paper, rename_cites, same_reference
from src.m2.bib_parse import BibEntry, parse_bib
from src.m2.bibitem import parse_bibitems

# ---------------------------------------------------------------------------------------
# .bib parser
# ---------------------------------------------------------------------------------------
BIB = r"""
@string{nips = "Advances in Neural Information Processing Systems"}
@STRING{ jmlr = {Journal of Machine Learning Research} }
@comment{ ignored @article{nope, title={x}} }
@preamble{ "\newcommand{\noop}[1]{}" }

@inproceedings{vaswani2017,
  title     = {Attention is All you {Need}},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle = nips # " 30",
  year      = 2017,
  month     = dec,
  pages     = {5998--6008},
}

@article{ schr,
  title = "Schr{\"o}dinger's {C}at",
  author = {Erwin Schr\"{o}dinger and Smith, Jr., John},
  journal = jmlr,
  year = {1935},
  doi = {https://doi.org/10.1000/ABC.123},
  volume = {12}, number={3},
}

@article{broken,
  title = {Unclosed {brace,
  year = 2020,

@misc{arx,
  title={A preprint},
  author={Doe, Jane},
  eprint={2101.01234},
  archivePrefix={arXiv},
  note = {arXiv:2101.01234},
}
@book{nob, author={Knuth, Donald E.}, title={The {TeX}book}, publisher={Addison-Wesley}, year={1984}}
"""


@pytest.mark.parametrize("raw, expected", [
    ("article", "article"), ("INPROCEEDINGS", "inproceedings"), ("conference", "inproceedings"),
    ("articl", "article"), ("inproeedings", "inproceedings"), ("articlebazan", "article"),
    ("electronic", "online"), ("www", "online"), ("masterthesis", "mastersthesis"), ("data", "dataset"),
    ("std", "standard"), ("bookchapter", "incollection"), ("software", "software"),
    ("journal", "misc"), ("footnote", "misc"), ("unknown", "misc"), ("", "misc"),
])
def test_entry_type_normalization(raw, expected):
    from src.m2.bib_parse import ENTRY_TYPES, normalize_entry_type
    assert normalize_entry_type(raw) == expected
    assert normalize_entry_type(raw) in ENTRY_TYPES


def test_bib_entry_type_is_normalized():
    ents, _ = parse_bib("@Inproeedings{k, title={A title here}, year={2020}}\n@www{w, title={Site}}")
    assert [e.entry_type for e in ents] == ["inproceedings", "online"]


def test_bib_parse_fields_strings_concat():
    ents = {e.key: e for e in parse_bib(BIB)[0]}
    v = ents["vaswani2017"]
    assert v.entry_type == "inproceedings"
    assert v.title == "Attention is All you Need"
    assert v.authors == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    assert v.year == 2017
    assert v.journal == "Advances in Neural Information Processing Systems 30"  # booktitle fallback
    assert v.pages == "5998–6008"


def test_bib_parse_accents_doi_suffix_and_quotes():
    s = {e.key: e for e in parse_bib(BIB)[0]}["schr"]
    assert s.title == "Schrödinger's Cat"
    assert s.authors == ["Erwin Schrödinger", "John Smith Jr."]
    assert s.doi == "10.1000/abc.123"
    assert s.journal == "Journal of Machine Learning Research"


def test_malformed_entry_dropped_only():
    ents, stats = parse_bib(BIB)
    keys = {e.key for e in ents}
    assert "broken" not in keys
    assert {"vaswani2017", "schr", "arx", "nob"} <= keys
    assert "nope" not in keys
    assert stats["malformed"] >= 1


def test_eprint_and_year_from_arxiv_id():
    a = {e.key: e for e in parse_bib(BIB)[0]}["arx"]
    assert a.eprint == "2101.01234"
    assert a.year == 2021  # derived from YYMM when no year field


def test_book_entry():
    b = {e.key: e for e in parse_bib(BIB)[0]}["nob"]
    assert b.entry_type == "book" and b.title == "The TeXbook" and b.publisher == "Addison-Wesley"


# ---------------------------------------------------------------------------------------
# \bibitem -> BibTeX (real examples from the corpus)
# ---------------------------------------------------------------------------------------
REAL_BIBITEMS = [
    # (raw, title, first author "First Last", year, method)
    (r"""\bibitem{CHGNet2023} \bibinfo{author}{Deng, B.} \emph{et~al.} \newblock \bibinfo{title}{{CHGNet} as a pretrained universal neural network potential for charge-informed atomistic modelling}. \newblock \emph{\bibinfo{journal}{Nat. Mach. Intell.}} \textbf{\bibinfo{volume}{5}}, \bibinfo{pages}{1031--1041} (\bibinfo{year}{2023}). \newblock \urlprefix\url{https://doi.org/10.1038/s42256-023-00716-3}.""",
     "CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling", "B. Deng", 2023, "bibinfo"),
    (r"""\bibitem [{\citenamefont {Moore}\ and\ \citenamefont {Geraci}(2021)}]{Moore2021}\BibitemOpen \bibfield  {author} {\bibinfo {author} {\bibfnamefont {D.~C.}\ \bibnamefont {Moore}}\ and\ \bibinfo {author} {\bibfnamefont {A.~A.}\ \bibnamefont {Geraci}},\ }\bibfield  {title} {\bibinfo {title} {Searching for new physics using optically levitated sensors},\ }\href {https://doi.org/10.1088/2058-9565/abcf8a} {\bibfield  {journal} {\bibinfo  {journal} {Quantum Sci. Technol.}\ }\textbf {\bibinfo {volume} {6}},\ \bibinfo {pages} {014008} (\bibinfo {year} {2021})}\BibitemShut {NoStop}""",
     "Searching for new physics using optically levitated sensors", "D. C. Moore", 2021, "bibinfo"),
    (r"""\bibitem{su2021roformer} Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. \newblock RoFormer: Enhanced Transformer with Rotary Position Embedding. \newblock \emph{arXiv preprint arXiv:2104.09864}, 2021.""",
     "RoFormer: Enhanced Transformer with Rotary Position Embedding", "Jianlin Su", 2021, "newblock"),
    (r"""\bibitem[Ho et~al.(2020)]{ho2020ddpm} Jonathan Ho, Ajay Jain, and Pieter Abbeel. \newblock Denoising Diffusion Probabilistic Models. \newblock \emph{NeurIPS}, 2020.""",
     "Denoising Diffusion Probabilistic Models", "Jonathan Ho", 2020, "newblock"),
    (r"""\bibitem{Sio58} M.~Sion. \newblock On general minimax theorems. \newblock {\em Pacific Journal of Mathematics}, 8:171--176, 1958. \newblock \href {https://doi.org/10.2140/pjm.1958.8.171} {\path{doi:10.2140/pjm.1958.8.171}}.""",
     "On general minimax theorems", "M. Sion", 1958, "newblock"),
    (r"""\bibitem{Bar02} A.~Barvinok. \newblock {\em A Course in Convexity}, volume~54 of {\em Graduate Studies in Mathematics}. \newblock American Mathematical Society, 2002.""",
     "A Course in Convexity", "A. Barvinok", 2002, "newblock"),
    (r"""\bibitem{Doetsch1974} Gustav Doetsch, \newblock \emph{Introduction to the Theory and Application of the Laplace   Transformation}, \newblock Springer--Verlag, Berlin and Heidelberg, 1974.""",
     "Introduction to the Theory and Application of the Laplace Transformation", "Gustav Doetsch", 1974, "newblock"),
    (r"""\bibitem{Gubser:1998bc} S.~S.~Gubser, I.~R.~Klebanov and A.~M.~Polyakov, ``Gauge theory correlators from noncritical string theory,'' Phys. Lett. B \textbf{428} (1998), 105-114 doi:10.1016/S0370-2693(98)00377-3 [arXiv:hep-th/9802109 [hep-th]].""",
     "Gauge theory correlators from noncritical string theory", "S. S. Gubser", 1998, "regex"),
    (r"""\bibitem{MandB} M. Berciu and B. Janko, Phys. Rev. Lett. {\bf 90}, 246804 (2003).""",
     None, "M. Berciu", 2003, "regex"),
    (r"""\bibitem{rami2000linear-du5} M. A. Rami and X. Y. Zhou, "Linear matrix inequalities, riccati equations, and indefinite stochastic linear quadratic controls," \emph{IEEE Transactions on Automatic Control}, vol. 45, no. 6, pp. 1131--1143, 2000.""",
     "Linear matrix inequalities, riccati equations, and indefinite stochastic linear quadratic controls", "M. A. Rami", 2000, "regex"),
    (r"""\bibitem{Bessaga} C.~Bessaga, \emph{On the converse of the Banach fixed-point principle}, Colloq.\ Math.\ \textbf{7} (1959), 41--43.""",
     "On the converse of the Banach fixed-point principle", "C. Bessaga", 1959, "regex"),
    (r"""\bibitem{ref13} Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., \ldots{} \& Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proc. National Academy of Sciences, 114(13), 3521-3526. \url{https://doi.org/10.1073/pnas.1611835114}""",
     "Overcoming catastrophic forgetting in neural networks", "J. Kirkpatrick", 2017, "regex"),
    (r"""\bibitem{Hu25}Y. Hu and D. Wen, {\it Dynamical gravitational Casimir-Polder interaction}, \href{https://doi.org/10.1103/hcdb-7gwt}{Phys. Rev. D {\bf 112}, 026010 (2025)}.""",
     "Dynamical gravitational Casimir-Polder interaction", "Y. Hu", 2025, "regex"),
    (r"""\bibitem{BH} Berger, Annabell; Hochst\"attler, Winfried: Minconvex graph factors of prescribed size and a simpler reduction to weighted f-factors. Electronic Notes in Discrete Mathematics 28:69--76 (2007)""",
     "Minconvex graph factors of prescribed size and a simpler reduction to weighted f-factors", "Annabell Berger", 2007, "regex"),
    (r"""\bibitem{SiffertIVP} A.~Siffert, The initial value problem for harmonic maps of cohomogeneity one manifolds, preprint, arXiv:2601.15022 (2026).""",
     "The initial value problem for harmonic maps of cohomogeneity one manifolds", "A. Siffert", 2026, "regex"),
    (r"""\bibitem{milner1989} R. Milner, \emph{Communication and Concurrency}, Prentice Hall, 1989.""",
     "Communication and Concurrency", "R. Milner", 1989, "regex"),
    (r"""\bibitem{Datta:2018thy} S.~Datta and Y.~Jiang, ``$T\bar{T}$ deformed partition functions,'' JHEP \textbf{08} (2018), 106 doi:10.1007/JHEP08(2018)106 [arXiv:1806.07426 [hep-th]].""",
     "$T\\bar{T}$ deformed partition functions", "S. Datta", 2018, "regex"),
    # Vancouver-style "Surname Initials" without periods (2608.14007)
    (r"""\bibitem{AkiRichards} Aki K, Richards P G. Quantitative Seismology: Theory and Methods. W. H. Freeman and Company, 1980.""",
     "Quantitative Seismology: Theory and Methods", "K. Aki", 1980, "regex"),
    (r"""\bibitem{Arens1999} Arens T. The scattering of plane elastic waves by a one-dimensional periodic surface. Mathematical Methods in the Applied Sciences, 1999, 22: 55-72.""",
     "The scattering of plane elastic waves by a one-dimensional periodic surface", "T. Arens", 1999, "regex"),
    (r"""\bibitem{JLF2006} Joly P, Li J-R, Fliss S. Exact boundary conditions for periodic waveguides containing a local perturbation. Communications in Computational Physics, 2006, 1: 945-973.""",
     "Exact boundary conditions for periodic waveguides containing a local perturbation", "P. Joly", 2006, "regex"),
]


@pytest.mark.parametrize("raw, bad_title", [
    (r"\bibitem{freedman1975tail} David A. Freedman. On tail probabilities for martingales. The Annals of Probability, 3(1):100--118, 1975.", "Freedman"),
    (r"\bibitem{Bonnell1997} Bonnell I. A., Bate M. R., Clarke C. J., Pringle J. E., 1997, MNRAS, 285, 201", "A., Bate M"),
    (r"\bibitem{Lieb} Elliott H. Lieb, Robert Seiringer, Jan Philip Solovej and Jakob Yngvason, The Mathematics of the Bose Gas and its Condensation, Birkhauser Verlag, Basel, 2005.", None),
])
def test_first_name_initial_layouts_are_not_vancouver(raw, bad_title):
    e = parse_bibitems(raw)[0]
    assert e.title != bad_title
    assert not (e.title or "").startswith("Lieb, Robert")


@pytest.mark.parametrize("raw, title, first, year, method", REAL_BIBITEMS)
def test_real_bibitems(raw, title, first, year, method):
    ents = parse_bibitems(raw)
    assert len(ents) == 1
    e = ents[0]
    assert (e.title or None) == title
    assert e.authors and e.authors[0] == first
    assert e.year == year
    assert e.parse_method == method
    assert 0.0 <= e.confidence <= 1.0
    assert e.source == "bibitem"


def test_bibitem_doi_and_arxiv_extraction():
    e = parse_bibitems(REAL_BIBITEMS[7][0])[0]
    assert e.doi == "10.1016/s0370-2693(98)00377-3"
    assert e.eprint == "hep-th/9802109"
    assert e.journal.startswith("Phys. Lett. B")
    assert e.entry_type == "article"


def test_bibitem_split_and_ignores_bibitemstop():
    text = (r"\providecommand \bibitemStop [0]{} \bibitem{a} A. One, Title one, 2001. "
            r"\bibitem[B(2002)]{b} B. Two, Title two, 2002.")
    keys = [e.key for e in parse_bibitems(text)]
    assert keys == ["a", "b"]


def test_inproceedings_detection():
    e = parse_bibitems(r"\bibitem{x} A. Author, ``A title,'' in \emph{Proceedings of the Conference on X}, 2020, pp. 1--2.")[0]
    assert e.entry_type == "inproceedings"
    assert e.journal == "Proceedings of the Conference on X"


# ---------------------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------------------
def E(key, version, source="bib", **kw):
    e = BibEntry(key=key, entry_type=kw.pop("entry_type", "article"), source=source)
    for k, val in kw.items():
        setattr(e, k, val)
    e.version = version
    return e


def test_statement_example_vaswani_vit():
    t = "Attention is all you need"
    a = ["Ashish Vaswani", "Noam Shazeer"]
    v1 = [E("vaswani2017", 1, title=t, authors=a, year=2017)]
    v2 = [E("vit_transformer", 2, title=t, authors=a, year=2017, doi="10.5555/3295222")]
    v3 = [E("vit_transformer", 3, title=t, authors=a)]
    res = dedup_paper({1: v1, 2: v2, 3: v3}, cite_counts={})
    assert len(res.entries) == 1
    m = res.entries[0]
    assert m.key == "vit_transformer"
    assert m.title == t and m.authors == a and m.year == 2017 and m.doi == "10.5555/3295222"
    assert res.key_map[(1, "vaswani2017")] == "vit_transformer"
    assert res.key_map[(3, "vit_transformer")] == "vit_transformer"
    assert sorted(m.versions) == [1, 2, 3]
    assert set(m.merged_keys) == {"vaswani2017", "vit_transformer"}


def test_doi_veto():
    a = E("a", 1, title="Deep learning", authors=["Y. LeCun"], year=2015, doi="10.1038/nature14539")
    b = E("b", 1, title="Deep learning", authors=["Y. LeCun"], year=2015, doi="10.1000/other")
    assert not same_reference(a, b)


def test_arxiv_and_year_veto():
    a = E("a", 1, title="A survey of things", authors=["J. Doe"], year=2010, eprint="1001.00001")
    b = E("b", 1, title="A survey of things", authors=["J. Doe"], year=2010, eprint="1002.00002")
    assert not same_reference(a, b)
    c = E("c", 1, title="A survey of things", authors=["J. Doe"], year=2010)
    d = E("d", 1, title="A survey of things", authors=["J. Doe"], year=2015)
    assert not same_reference(c, d)


def test_same_doi_merges_despite_title_noise():
    a = E("a", 1, title="Neural ODEs", authors=["R. Chen"], year=2018, doi="10.1/x")
    b = E("b", 2, title="Neural ordinary differential equations", authors=["Ricky Chen"], year=2018, doi="10.1/X")
    assert same_reference(a, b)


def test_fuzzy_title_with_author_check():
    a = E("a", 1, title="Denoising diffusion probabilistic models", authors=["Jonathan Ho"], year=2020)
    b = E("b", 1, title="Denoising Diffusion Probabilistic Model", authors=["J. Ho"], year=2020)
    c = E("c", 1, title="Denoising diffusion probabilistic models", authors=["Someone Else"], year=2020)
    assert same_reference(a, b)
    assert not same_reference(a, c)


def test_within_version_duplicates_and_bib_outranks_bibitem():
    v1 = [E("ldm", 1, title="High-Resolution Image Synthesis with Latent Diffusion Models", authors=["Robin Rombach"], year=2022),
          E("rombach22", 1, source="bibitem", title="High-resolution image synthesis with latent diffusion models",
            authors=["R. Rombach"], year=2022, journal="CVPR wrong")]
    v1[0].journal = "CVPR"
    res = dedup_paper({1: v1}, cite_counts={"ldm": 3, "rombach22": 1})
    assert len(res.entries) == 1
    assert res.entries[0].key == "ldm"  # tie in version -> most cited
    assert res.entries[0].journal == "CVPR"


def test_same_key_conflict_in_one_version_keeps_bib_entry():
    # 2608.14102: a dead \bibitem{b11} (Dagan 2018) and the .bib b11 (Zhou 2025) share a key
    v1 = [E("b1", 1, title="The entropy of lies: playing twenty questions with a liar",
            authors=["Yuval Dagan"], year=2018),
          E("b11", 1, title="Twenty Questions with Random Error", authors=["Lin Zhou"], year=2025,
            doi="10.1561/0100000144"),
          E("b11", 1, source="bibitem", title="The entropy of lies: playing twenty questions with a liar",
            authors=["Y. Dagan", "Y. Filmus", "D. Kane", "S. Moran"], year=2018)]
    res = dedup_paper({1: v1})
    by = {e.key: e for e in res.entries}
    assert sorted(by) == ["b1", "b11"]
    assert by["b11"].title == "Twenty Questions with Random Error"
    assert by["b11"].authors == ["Lin Zhou"]
    assert by["b1"].doi == "" and by["b1"].merged_keys == ["b1"]
    assert res.key_map[(1, "b1")] == "b1" and res.key_map[(1, "b11")] == "b11"
    assert res.n_before == 3


def test_veto_blocks_transitive_merge():
    # a ~ c by title, b ~ c by DOI, but a and b have different DOIs: no group may hold both
    a = E("a", 1, title="Graph neural networks for physics", authors=["A. Smith"], year=2020, doi="10.1/a")
    b = E("b", 1, title="Something else entirely", authors=["B. Jones"], year=2020, doi="10.1/b")
    c = E("c", 2, title="Graph neural networks for physics", authors=["A. Smith"], year=2020)
    c2 = E("c2", 2, title="Unrelated", authors=["C. Doe"], year=2020, doi="10.1/b")
    res = dedup_paper({1: [a, b], 2: [c, c2]})
    groups = [set(e.merged_keys) for e in res.entries]
    assert not any({"a", "b"} <= g for g in groups)
    assert {"a", "c"} in groups and {"b", "c2"} in groups


def test_same_key_across_versions_still_merges():
    v1 = [E("k", 1, title="Some paper title here", authors=["A. B"], year=2021, journal="arXiv")]
    v2 = [E("k", 2, title="Some paper title here", authors=["A. B"], year=2024, journal="JMLR")]
    res = dedup_paper({1: v1, 2: v2})
    assert len(res.entries) == 1 and res.entries[0].journal == "JMLR"


def test_rename_cites():
    txt = "As in \\cite{vaswani2017, other} and \\cite{x}."
    assert rename_cites(txt, {"vaswani2017": "vit_transformer"}) == \
        "As in \\cite{vit_transformer,other} and \\cite{x}."
    assert rename_cites("\\cite{a,b}", {"a": "c", "b": "c"}) == "\\cite{c}"
