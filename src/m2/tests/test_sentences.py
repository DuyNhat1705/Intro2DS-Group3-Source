r"""Tests of the rule-based sentence splitter (prompt §5.5). Inputs are *cleaned* text."""

from __future__ import annotations

import pytest

from src.m2.sentences import SentenceSplitter, split_sentences


@pytest.mark.parametrize("text, expected", [
    # statement cases: math, commands, abbreviations, URLs, decimals
    ("We set $x = 3.5. y$ here. Next one.", ["We set $x = 3.5. y$ here.", "Next one."]),
    ("As shown in \\cite{a.b, c.d}. Then we go.", ["As shown in \\cite{a.b, c.d}.", "Then we go."]),
    ("See \\ref{sec:a.b} for details. Ok.", ["See \\ref{sec:a.b} for details.", "Ok."]),
    ("This holds, i.e. the bound is tight. Good.", ["This holds, i.e. the bound is tight.", "Good."]),
    ("We use e.g. Adam. It is fast.", ["We use e.g. Adam.", "It is fast."]),
    ("Smith et al. proposed it. We extend it.", ["Smith et al. proposed it.", "We extend it."]),
    ("As in Fig. 1 we see. Also Fig. 2.", ["As in Fig. 1 we see.", "Also Fig. 2."]),
    ("As in Fig.~1 we see. Done.", ["As in Fig.~1 we see.", "Done."]),
    ("Code at https://github.com/x/y.git. Try it.", ["Code at https://github.com/x/y.git.", "Try it."]),
    ("Mail a.b@uni.edu. Thanks.", ["Mail a.b@uni.edu.", "Thanks."]),
    ("DOI 10.1000/xyz.123. End.", ["DOI 10.1000/xyz.123.", "End."]),
    ("The value is 3.14 and v2.1 is used. Yes.", ["The value is 3.14 and v2.1 is used.", "Yes."]),
    ("Accuracy rose to 95.5%. Then it fell.", ["Accuracy rose to 95.5%.", "Then it fell."]),
    # initials
    ("Proposed by J. Smith in 2020. Later work.", ["Proposed by J. Smith in 2020.", "Later work."]),
    ("By A. B. Cohen and X.~Y. Li. Second.", ["By A. B. Cohen and X.~Y. Li.", "Second."]),
    # etc. / al. genuinely ending a sentence
    ("We use cats, dogs, etc. The model is big.", ["We use cats, dogs, etc.", "The model is big."]),
    ("This is due to Smith et al. We follow them.", ["This is due to Smith et al.", "We follow them."]),
    ("We compare methods, etc. and more.", ["We compare methods, etc. and more."]),
    # cite after period attaches to the previous sentence
    ("Prior work exists. \\cite{x} We improve it.", ["Prior work exists. \\cite{x}", "We improve it."]),
    # other punctuation, quotes and brackets
    ("Is it true? Yes! It is.", ["Is it true?", "Yes!", "It is."]),
    ('He said "stop." Then left.', ['He said "stop."', "Then left."]),
    ("A result (see above.) Then B.", ["A result (see above.)", "Then B."]),
    ("First claim. $x$ is positive.", ["First claim.", "$x$ is positive."]),
    ("First claim. 2 cases remain.", ["First claim.", "2 cases remain."]),
    ("w.r.t. the loss we see. Ok.", ["w.r.t. the loss we see.", "Ok."]),
    ("Eqs. 3 and 4 hold. Sec. 2 shows it. Tab. 1 too.", ["Eqs. 3 and 4 hold.", "Sec. 2 shows it.", "Tab. 1 too."]),
    ("We wait... and then go. Done.", ["We wait... and then go.", "Done."]),
    ("vs. baseline we win. Ok.", ["vs. baseline we win.", "Ok."]),
    ("lowercase start. still same sentence.", ["lowercase start. still same sentence."]),
    # empty / punctuation-only pieces are dropped
    ("Real one. . ..", ["Real one."]),
    ("", []),
])
def test_split(text, expected):
    assert split_sentences(text) == expected


def test_cache_hits():
    sp = SentenceSplitter(max_bytes=10_000)
    a = sp.split("One. Two.")
    b = sp.split("One. Two.")
    assert a == b == ["One.", "Two."]
    assert sp.hits == 1 and sp.misses == 1


def test_cache_bounded():
    sp = SentenceSplitter(max_bytes=200)
    for i in range(100):
        sp.split(f"Sentence number {i}. Another one {i}.")
    assert sp.size_bytes <= 200
