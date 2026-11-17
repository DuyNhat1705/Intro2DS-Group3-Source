r"""Tách \bibitem theo khối \newblock và style tác giả–năm (chỉ dùng văn bản nguồn)."""

import pytest

from src import bibtex

BS = "\\"
NB = BS + "newblock "


def parse(raw):
    return bibtex._parse_bibitem_block("k", raw, "2609.00001", enrich=False)


@pytest.mark.parametrize(
    "raw, title, authors, journal",
    [
        # amsplain: khối tác giả kết thúc bằng dấu phẩy, tạp chí in nghiêng ở khối 3.
        (
            "M. Bonk and P. Koskela, "
            + NB
            + "Conformal metrics and size of the boundary, "
            + NB
            + BS
            + "emph{American Journal of Mathematics} "
            + BS
            + "textbf{124} (2002), 1247--1287.",
            "Conformal metrics and size of the boundary",
            ["M. Bonk", "P. Koskela"],
            "American Journal of Mathematics",
        ),
        # plain: ba khối trở lên trước phần in nghiêng.
        (
            "M.~Grobler, S.~Maaz, and S.~Siebertz. "
            + NB
            + "Solution discovery via reconfiguration for problems in P. "
            + NB
            + "In "
            + BS
            + "emph{51st International Colloquium on Automata, Languages, and Programming (ICALP 2024)}, "
            "volume 297 of " + BS + "emph{LIPIcs}, 2024.",
            "Solution discovery via reconfiguration for problems in P",
            ["M. Grobler", "S. Maaz", "S. Siebertz"],
            "51st International Colloquium on Automata, Languages, and Programming (ICALP 2024)",
        ),
        # Khối thứ hai là arXiv thì không phải tiêu đề.
        (
            "M. F. bin Zanal Abidin, "
            + NB
            + "Newton support functions and metric completion, "
            + NB
            + "arXiv:2608.17714 [math.DG] (2026).",
            "Newton support functions and metric completion",
            ["M. F. bin Zanal Abidin"],
            None,
        ),
    ],
)
def test_newblock_blocks_are_authors_title_venue(raw, title, authors, journal):
    entry = parse(raw)
    assert (entry.title, entry.authors, entry.journal) == (title, authors, journal)


def test_book_with_italic_title_block_keeps_title():
    entry = parse(
        "J. G. Kemeny and J. L. Snell. " + NB + BS + "emph{Finite Markov Chains}. " + NB + "Van Nostrand, 1960."
    )
    assert entry.title == "Finite Markov Chains" and entry.authors == ["J. G. Kemeny", "J. L. Snell"]


def test_author_year_style_drops_year_and_keeps_question_title():
    entry = parse(
        "Tan, M., M.~A. Merrill, and T. Hartvigsen. 2024. Are language models actually useful for "
        "time series forecasting? " + BS + "emph{Advances in Neural Information Processing Systems} 37."
    )
    assert entry.title == "Are language models actually useful for time series forecasting?"
    assert entry.authors == ["M. Tan", "M. A. Merrill", "T. Hartvigsen"]
    assert entry.journal == "Advances in Neural Information Processing Systems"
    assert parse(
        "Demšar, J. 2006. Statistical comparisons of classifiers over multiple data sets. " + BS + "emph{JMLR} 7."
    ).authors == ["J. Demšar"]


def test_plain_venue_block_is_trimmed_before_volume_and_year():
    entry = parse("Z. Author. " + NB + "A title of the paper. " + NB + "Annals of Probability, 48(2):963--1001, 2020.")
    assert entry.title == "A title of the paper" and entry.journal == "Annals of Probability"
    assert (
        parse("Z. Author. " + NB + "Another title here. " + NB + "URL: https://arxiv.org/abs/1412.1143").journal is None
    )


def test_group_italic_and_spaced_href_are_recognized():
    entry = parse(
        "B. Riemann, {" + BS + "it Ein Beitrag zur Elektrodynamik}, Ann. Phys. Chem. {" + BS + "bf 131}, 237 (1867)."
    )
    assert entry.title == "Ein Beitrag zur Elektrodynamik"
    entry = parse(
        "I.~Bouzenia and M.~Pradel. 2025. "
        + NB
        + BS
        + "href {https://doi.org/10.1109/ICSE55347.2025.00157} {{RepairAgent}: An autonomous agent}. "
        + NB
        + "In "
        + BS
        + "emph{Proc. ICSE}."
    )
    assert entry.title == "RepairAgent: An autonomous agent" and entry.journal == "Proc. ICSE"


def test_raw_text_is_kept_for_post_processing():
    raw = "A. Author. " + NB + "A title here. " + NB + "Somewhere, 2020."
    assert parse(raw).raw_text == raw
