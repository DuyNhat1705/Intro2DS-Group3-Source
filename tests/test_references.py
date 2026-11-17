"""Chuẩn hóa trường reference: LaTeX sang Unicode, tên tác giả, arXiv ID và năm."""

import pytest

from src import bibtex
from src.m2_types import BibtexEntry

BS = "\\"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("{BoxE}: A box embedding model", "BoxE: A box embedding model"),
        ("{" + BS + ".I}statistik{" + BS + "c{c}}iler Dergisi", "İstatistikçiler Dergisi"),
        (BS + "textit{Nature}", "Nature"),
        ("{" + BS + "it Physical Review} {" + BS + "bf 12}", "Physical Review 12"),
        ("On $" + BS + "alpha$-stable {L}{" + BS + "'e}vy processes", "On $" + BS + "alpha$-stable Lévy processes"),
        (BS + "bibinfo{title}{Deep {L}earning}", "Deep Learning"),
        ("50" + BS + "% of {GPT}-4", "50% of GPT-4"),
        ("S" + BS + "'eminaire de Probabilit" + BS + "'es", "Séminaire de Probabilités"),
        ("Erd" + BS + "H{o}s and " + BS + "v Sev" + BS + "c{c}", "Erdős and Ševç"),
        (
            "Costs up to " + BS + "$8.6 Billion " + BS + "& more " + BS + BS + " here",
            "Costs up to $8.6 Billion & more here",
        ),
    ],
)
def test_standardize_text(raw, expected):
    assert bibtex._standardize_text(raw) == expected
    assert bibtex._standardize_text(expected) == expected  # lũy đẳng


def test_bibitem_umlaut_is_not_a_title_quote():
    entry = bibtex._parse_bibitem_block(
        "k", "U. K" + BS + "\"uhn, ``Analysis and PDE,'' Birkh" + BS + '"auser, 2019.', "2609.00001", enrich=False
    )
    assert entry.title == "Analysis and PDE" and entry.authors == ["U. Kühn"] and entry.year == 2019


def test_authors_unicode_and_first_last_order():
    raw = (
        "M{" + BS + '"u}ller, J{' + BS + "'e}r{" + BS + "^o}me and Erd" + BS + "H{o}s, Paul and "
        "{Google DeepMind} and Smith, Jr., John and {OpenAI, Inc.} and Xueyang Zhou and others"
    )
    assert bibtex._split_authors(raw) == [
        "Jérôme Müller",
        "Paul Erdős",
        "Google DeepMind",
        "John Smith Jr.",
        "OpenAI, Inc.",
        "Xueyang Zhou",
    ]


def test_bibitem_authors_with_tilde_are_separate_people():
    assert bibtex._parse_authors_from_text("T.~Matsusaka, W.~Chen and A.~Palmieri") == [
        "T. Matsusaka",
        "W. Chen",
        "A. Palmieri",
    ]
    assert bibtex._parse_authors_from_text("Smith, J., Doe, K.") == ["J. Smith", "K. Doe"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("arXiv preprint arXiv:2208.07628", "2208.07628"),
        ("http://arxiv.org/abs/2602.24235v2", "2602.24235v2"),
        ("doi:10.48550/arXiv.2301.00001", "2301.00001"),
        ("arXiv: hep-th/9901001", "hep-th/9901001"),
        ("Nature 2020", None),
    ],
)
def test_find_arxiv_id(text, expected):
    assert bibtex._find_arxiv_id(text) == expected


def test_bib_fields_give_arxiv_id_and_year():
    assert bibtex._bib_arxiv_id({"eprint": "2104.00001", "archiveprefix": "arXiv"}) == "2104.00001"
    assert bibtex._bib_arxiv_id({"eprint": "12345", "archiveprefix": "HAL"}) is None
    assert bibtex._bib_arxiv_id({"journal": "arXiv preprint arXiv:2404.16130"}) == "2404.16130"
    assert bibtex._bib_year({"year": "2017a"}) == 2017
    assert bibtex._bib_year({"date": "2020-05-01"}) == 2020
    assert bibtex._bib_year({"year": "in press"}) is None


def test_bib_file_entry_is_standardized(tmp_path):
    bib = tmp_path / "refs.bib"
    bib.write_text(
        "@article{k, title={{BoxE}: Box {E}mbeddings}, author={M{" + BS + '"u}ller, Jan},'
        " journal={arXiv preprint arXiv:2007.06267}, date={2020-07}}",
        encoding="utf-8",
    )
    entry = bibtex._parse_bib_file(bib, "2609.00001")[0]
    assert entry.title == "BoxE: Box Embeddings"
    assert entry.authors == ["Jan Müller"]
    assert entry.arxiv_id_hint == "2007.06267" and entry.year == 2020


def test_year_inferred_from_arxiv_id_only_when_missing():
    def preprint(key, year=None):
        return BibtexEntry(
            arxiv_id="2609.00001",
            citation_key=key,
            entry_type="misc",
            title="On the threshold nature of Dini continuity",
            authors=["W. Chen"],
            year=year,
            arxiv_id_hint="2306.11478",
        )

    inferred = bibtex.unify_publication("2609.00001", {1: [preprint("a")]}).entries[0]
    assert inferred.year == 2023 and inferred.to_debug_row()["inferred_fields"] == ["year"]
    stated = bibtex.unify_publication("2609.00001", {1: [preprint("a", year=2024)]}).entries[0]
    assert stated.year == 2024 and stated.to_debug_row()["inferred_fields"] == []
