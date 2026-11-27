"""Local regression checks for matching and the Parquet CLI."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from inputs import read_bibtex
from matching import arxiv_id, get_title_similarity, match, normalize

PAPER = "2609.17834"
OTHER = "2609.17835"
EMPTY = "2609.17836"
MANUAL = "2609.17837"
MISSING = "2609.17838"
TITLE = "Sparse estimation of high-dimensional random graphs"


class MatchingTests(unittest.TestCase):
    def test_normalizes_legacy_id(self):
        self.assertEqual(arxiv_id("arXiv:hep-th/9901001v3"), "hep-th/9901001")
        self.assertEqual(arxiv_id("https://arxiv.org/pdf/hep-th/9901001v3.pdf?download=1"), "hep-th/9901001")

    def test_title_is_the_only_evidence(self):
        row = {"arxiv_id": PAPER, "title": TITLE, "authors": ["A. Smith"],
               "year": 2012, "url": "https://arxiv.org/abs/2401.12345", "doi": "10.48550/arXiv.2401.12345"}
        candidates = {"2401.12345": ["unrelated words"], "2401.23456": [normalize(TITLE)]}
        result = match(row, candidates, .85, .06, title_similarity=get_title_similarity("sequence"))
        self.assertEqual(result["groundtruth_arxiv_id"], "2401.23456")
        self.assertEqual(result["match_score"], 1)

    def test_title_tie_is_no_match_even_with_zero_margin(self):
        title = normalize(TITLE)
        result = match({"title": TITLE}, {"2401.23456": [title], "2401.12345": [title]}, .85, 0,
                       title_similarity=get_title_similarity("sequence"))
        self.assertEqual(result["label_status"], "no_match")
        self.assertEqual(result["reason"], "ambiguous_candidates")
        self.assertIsNone(result["groundtruth_arxiv_id"])
        self.assertEqual(result["candidate_1_arxiv_id"], "2401.12345")
        self.assertEqual(result["candidate_1_score"], result["candidate_2_score"])

    @unittest.skipUnless(importlib.util.find_spec("rapidfuzz"), "requires rapidfuzz")
    def test_default_uses_rapidfuzz_title_score(self):
        row = dict(arxiv_id=PAPER, title="tide", authors=["Smith"])
        candidates = {"2401.12345": ["diet"]}
        result = match(row, candidates, .85, .06)
        self.assertEqual(result["label_status"], "no_match")
        self.assertEqual(result["match_score"], .5)
        self.assertEqual(get_title_similarity("sequence")("tide", "diet"), .25)

    def test_missing_titles_are_no_match(self):
        settings = {"title_similarity": get_title_similarity("sequence")}
        row = {"arxiv_id": PAPER, "url": "https://arxiv.org/abs/2401.12345"}
        self.assertEqual(match(row, {}, .85, .06, **settings)["label_status"], "no_match")
        result = match(row, {"2401.12345": [normalize(TITLE)]}, .85, .06, **settings)
        self.assertEqual(result["reason"], "missing_bibtex_title")
        self.assertEqual(result["label_status"], "no_match")
        result = match({"title": TITLE}, {"2401.12345": [""]}, .85, .06, **settings)
        self.assertEqual(result["label_status"], "no_match")
        self.assertEqual(result["reason"], "missing_candidate_titles")

    def test_keys_are_per_paper_and_conflicts_fail(self):
        first = {"arxiv_id": PAPER, "citation_key": "same", "title": TITLE}
        second = dict(first, arxiv_id=OTHER)
        with patch("inputs.parquet_rows", return_value=[(row, "shard") for row in (first, second, first)]):
            entries = read_bibtex([], {PAPER, OTHER})
            self.assertEqual(sum(len(group) for group in entries.values()), 2)
        conflict = dict(first, title="Different")
        with patch("inputs.parquet_rows", return_value=[(row, "shard") for row in (first, conflict)]):
            with self.assertRaisesRegex(ValueError, "Xung đột BibTeX"):
                read_bibtex([], {PAPER})


class PipelineTests(unittest.TestCase):
    def test_cli_distinguishes_empty_candidates_from_missing_exports(self):
        import auto_label

        range_one = "2609.27834"
        batch = lambda rows: [(row, "fixture") for row in rows]
        records = {
            "papers": batch({"arxiv_id": ident} for ident in (PAPER, EMPTY, MISSING, MANUAL, range_one)),
            "references": batch([dict(arxiv_id=PAPER, ref_arxiv_id="2401.12345", ref_title=TITLE)]),
            "bibtex": batch(dict(arxiv_id=ident, citation_key="same", title=TITLE,
                                url="https://arxiv.org/abs/2401.12345v2")
                            for ident in (PAPER, EMPTY, MISSING, MANUAL, range_one)),
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            refs_done = root / "refs_done.jsonl"
            refs_done.write_text("".join(
                json.dumps(dict(id=ident, status="no_refs")) + "\n" for ident in (EMPTY, range_one)))
            args = ["auto_label.py", "--papers", "papers.parquet", "--references", "references.parquet",
                    "--bibtex", "bibtex", "--refs-done", str(refs_done), "--manual-ids", MANUAL,
                    "--ranges", "1", "2", "--quota", "5", "--title-scorer", "sequence", "--out", str(root / "output")]
            with patch.object(sys, "argv", args), patch("inputs.parquet_rows") as source, \
                    patch("auto_label.write_outputs") as output:
                source.side_effect = lambda paths, table, *fields: iter(records[table])
                self.assertEqual(auto_label.main(), 0)
            _, labels, selected, summary = output.call_args.args
        self.assertEqual([p["arxiv_id"] for p in selected], [PAPER, EMPTY])
        self.assertEqual([r["label_status"] for r in labels], ["matched", "no_match"])
        self.assertFalse(summary["ranges"]["R1"]["m1_references_export_observed"])
        self.assertEqual(summary["ranges"]["R2"]["bibtex_without_confirmed_m1"], 1)
        self.assertEqual(summary["ranges"]["R2"]["selection_shortfall"], 3)
        self.assertFalse(summary["quota_met"])


@unittest.skipUnless(importlib.util.find_spec("pyarrow") and importlib.util.find_spec("rapidfuzz"),
                     "requires pyarrow and rapidfuzz")
class CliTests(unittest.TestCase):
    def test_parquet_round_trip_is_deterministic(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = [PAPER, OTHER, EMPTY, MANUAL, MISSING, "2609.27834"]
            pq.write_table(pa.Table.from_pylist([dict(arxiv_id=ident) for ident in ids]),
                           root / "papers.parquet")
            refs = [
                dict(arxiv_id=PAPER, ref_arxiv_id="2401.12345v1", ref_title=TITLE),
                dict(arxiv_id=PAPER, ref_arxiv_id="2401.12345v2", ref_title=TITLE),
                dict(arxiv_id=OTHER, ref_arxiv_id="2401.23456", ref_title=TITLE),
                dict(arxiv_id=MANUAL, ref_arxiv_id="2401.12345", ref_title=TITLE),
            ]
            pq.write_table(pa.Table.from_pylist(refs), root / "references.parquet")
            bib = [
                dict(arxiv_id=ident, citation_key="same", title=TITLE if ident in (PAPER, MANUAL) else None,
                     url="https://arxiv.org/abs/2401.12345v5" if ident == PAPER else None)
                for ident in ids
            ]
            bib.append(dict(bib[0], citation_key="explicit_elsewhere", title="Another paper",
                            url="https://arxiv.org/abs/2401.23456"))
            pq.write_table(pa.Table.from_pylist(bib), root / "part-c00000.parquet")
            (root / "refs_done.jsonl").write_text(
                json.dumps(dict(id=EMPTY, status="ok", n_refs=4)) + "\n"
                + json.dumps(dict(id=ids[-1], status="no_refs", n_refs=0)) + "\n")
            command = [sys.executable, str(Path(__file__).with_name("auto_label.py")),
                       "--papers", str(root / "papers.parquet"), "--references", str(root / "references.parquet"),
                       "--bibtex", str(root), "--refs-done", str(root / "refs_done.jsonl"), "--ranges", "1", "2"]
            sampled = [*command, "--manual-ids", MANUAL, "--quota", "5"]
            first, second = root / "first", root / "second"
            subprocess.run([*sampled, "--workers", "1", "--out", str(first)],
                           check=True, capture_output=True, text=True)
            subprocess.run([*sampled, "--out", str(second)],
                           check=True, capture_output=True, text=True)
            self.assertEqual((first / "selected_papers.csv").read_bytes(),
                             (second / "selected_papers.csv").read_bytes())
            serial_summary = json.loads((first / "summary.json").read_text())
            parallel_summary = json.loads((second / "summary.json").read_text())
            self.assertEqual(serial_summary["configuration"].pop("workers"), 1)
            self.assertEqual(parallel_summary["configuration"].pop("workers"), 4)
            self.assertEqual(serial_summary, parallel_summary)
            self.assertEqual(parallel_summary["configuration"]["title_scorer"], "rapidfuzz")
            table = pq.read_table(first / "auto_labels.parquet")
            self.assertEqual(table.schema.names, [
                "arxiv_id", "citation_key", "groundtruth_arxiv_id", "label_source", "label_status",
            ])
            self.assertEqual(table, pq.read_table(second / "auto_labels.parquet"))
            rows = table.to_pylist()
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["label_source"] == "automatic" for row in rows))
            states = {(r["arxiv_id"], r["citation_key"]): r["label_status"] for r in rows}
            self.assertEqual(states, {(PAPER, "same"): "matched", (OTHER, "same"): "no_match",
                                      (EMPTY, "same"): "no_match", (PAPER, "explicit_elsewhere"): "no_match"})
            with (first / "selected_papers.csv").open(newline="") as stream:
                self.assertEqual([r["arxiv_id"] for r in csv.DictReader(stream)], [PAPER, OTHER, EMPTY])
            summary = json.loads((first / "summary.json").read_text())
            self.assertFalse(summary["ranges"]["R1"]["m1_references_export_observed"])
            self.assertEqual(summary["ranges"]["R2"]["bibtex_without_confirmed_m1"], 1)
            self.assertEqual(summary["finalized_papers"], 3)
            self.assertFalse(summary["quota_met"])

            full = root / "full"
            subprocess.run([*command, "--full-corpus", "--out", str(full)],
                           check=True, capture_output=True, text=True)
            full_summary = json.loads((full / "summary.json").read_text())
            self.assertEqual(full_summary["mode"], "full_corpus")
            self.assertIsNone(full_summary["quota_met"])
            self.assertEqual(full_summary["selected_papers"], 4)
            self.assertEqual(full_summary["ranges"]["R2"]["available"], 4)
            self.assertEqual(full_summary["ranges"]["R2"]["bibtex_without_confirmed_m1"], 1)
            full_states = {(row["arxiv_id"], row["citation_key"]): row["label_status"]
                           for row in pq.read_table(full / "auto_labels.parquet").to_pylist()}
            self.assertEqual(full_states, {**states, (MANUAL, "same"): "matched"})


if __name__ == "__main__":
    unittest.main()
