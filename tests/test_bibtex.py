"""Hợp nhất reference giữa các phiên bản và cache enrichment."""

import io
import json
import warnings
from unittest.mock import patch

from src import bibtex
from src.m2_types import BibtexEntry


def entry(key, title=None, year=None, doi=None, authors=("Vaswani, Ashish",)):
    return BibtexEntry(
        arxiv_id="2609.00001",
        citation_key=key,
        entry_type="article",
        title=title,
        authors=list(authors),
        year=year,
        doi=doi,
    )


def final_key(result, version, key):
    """key_map chỉ chứa khóa bị đổi; khóa được chọn giữ nguyên tên."""
    return result.key_map.get(version, {}).get(key, key)


def test_spec_example_unions_fields_and_renames_keys():
    title = "Attention Is All You Need"
    result = bibtex.unify_publication(
        "2609.00001",
        {
            1: [entry("vaswani2017", title, year=2017)],
            2: [entry("vit_transformer", title, year=2017, doi="10.1000/attention")],
            3: [entry("vit_transformer", title, doi="10.1000/attention")],
        },
    )
    assert len(result.entries) == 1
    row = result.entries[0].to_row()
    assert row["year"] == 2017 and row["doi"] == "10.1000/attention"
    chosen = row["citation_key"]
    assert {
        final_key(result, 1, "vaswani2017"),
        final_key(result, 2, "vit_transformer"),
        final_key(result, 3, "vit_transformer"),
    } == {chosen}


def test_transitive_merges_across_stages_form_one_cluster():
    # a-b nối qua DOI, b-c qua tiêu đề, c-d qua key: cần đúng một cụm, các bài khác giữ riêng.
    title = "Deep Residual Learning for Image Recognition"
    result = bibtex.unify_publication(
        "2609.00001",
        {
            1: [entry("a", "Some Other Paper Title Here", doi="10.1/x"), entry("other", "Completely Unrelated Study")],
            2: [entry("b", title, doi="10.1/x"), entry("c", title, year=2016)],
            3: [entry("c", title, year=2016), entry("solo", "Yet Another Distinct Work")],
        },
    )
    assert len(result.entries) == 3
    merged = next(item for item in result.entries if item.citation_key == final_key(result, 1, "a"))
    assert sorted(merged.versions_present) == [1, 2, 3]
    assert {final_key(result, 2, "b"), final_key(result, 2, "c"), final_key(result, 3, "c")} == {merged.citation_key}
    assert final_key(result, 1, "other") != merged.citation_key


def _crossref_response(title, doi):
    body = json.dumps({"message": {"items": [{"title": [title], "DOI": doi, "author": [], "published": {}}]}})
    return io.BytesIO(body.encode())


def test_crossref_cache_calls_network_once_and_replays_warnings():
    title = "Attention Is All You Need"
    with patch("urllib.request.urlopen", side_effect=lambda *a, **k: _crossref_response(title, "10.1/A")) as urlopen:
        first = bibtex._enrich_with_crossref(title, None)
        first["title"] = "mutated by caller"
        second = bibtex._enrich_with_crossref(title, None)
    assert urlopen.call_count == 1
    assert second["title"] == title and second["doi"] == "10.1/a"

    with patch("urllib.request.urlopen", side_effect=OSError("offline")) as urlopen:
        for _ in range(2):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                assert bibtex._enrich_with_crossref("Unreachable Paper Title", None) is None
            assert any("CrossRef enrichment failed" in str(w.message) for w in caught)
    assert urlopen.call_count == 1
