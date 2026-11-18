# Intro2DS-Group3-Source

> **Course**: CSC14119 - Introduction to Data Science  
> **Group**: Group 3  

---

## Table of Contents

- [Project Overview](#i-project-overview)
- [Directory Structure](#ii-directory-structure)
- [Modules and Functionality](#iii-modules-and-functionality)
- [Installation and Setup](#iv-installation-and-setup)
- [Usage](#v-usage)
- [Dependencies](#vi-dependencies)

---

## I. Project Overview
### 1. Milestone 1 Overview

The goal of this project is to build an automated pipeline to crawl and collect scientific literature data from arXiv and academic APIs (such as Semantic Scholar). For each target paper, the pipeline collects:

1. **Metadata**: Core bibliographic details including paper title, authors, submission date, revision history, publication venue, categories, and abstract.
2. **Full LaTeX Source Archives**: Source files (TeX documents, bibliography files `.bib`, styles, classes, and figures) across revisions (`v1`, `v2`, etc.).
3. **Extracted References**: Citations and linked reference information (citing paper ID, reference arXiv ID, Semantic Scholar ID, title, authors, and publication date).

---

### 2. Milestone 2 Overview

Milestone 2 transforms the raw LaTeX sources collected in Milestone 1 into a cleaned, standardized hierarchical dataset for the reference-matching pipeline in Milestone 3.

For every publication and every version, the pipeline is designed to:

1. **Gather the compilation path**: Identify the main `.tex` file through its `\documentclass` declaration and follow active `\input` and `\include` commands transitively, ignoring commented commands and unused auxiliary files.
2. **Build the document hierarchy**: Organize documents, sections, subsections, paragraphs, and list items into a tree. Leaves represent sentences, block formulas, and figures (including tables). Sentence boundaries must preserve mathematics, commands, abbreviations, URLs, and decimal numbers. Include acknowledgements and appendices, but exclude reference sections from the hierarchy.
3. **Clean and normalize content**: Remove non-semantic formatting and placement commands, standardize inline mathematics to `$...$`, and standardize block mathematics to `\begin{equation}...\end{equation}`.
4. **Unify bibliography entries**: Extract entries from referenced `.bib` files and convert inline `\bibitem` references to BibTeX. Merge duplicates within and across versions, union their fields, choose canonical citation keys, and rewrite affected citation commands.
5. **Deduplicate full text across versions**: Store identical cleaned elements once with a shared identifier, while retaining each version's parent-child relationships. Record versions without TeX sources as empty for coverage reporting.

---

## II. Directory Structure

The source code is in `src/` and `data/` holds the local corpus collected in Milestone 1.

```text
Intro2DS-Group3-Source/
├── src/
|   ├── utils.py                     # Shared paths and GatherTex data structure
|   ├── get_metadata.py              # Milestone 1: bibliographic metadata
|   ├── get_full_source.py           # Milestone 1: versioned .tex/.bib downloads
|   ├── get_extracting_reference.py  # Milestone 1: citation API references
|   ├── scraper.ipynb                # Milestone 1: collection and reporting
|   ├── m2/                          # Milestone 2: parser package (see section VII)
|   ├── parser.ipynb                 # Milestone 2: Colab/local runner and benchmark notebook
|   ├── intro2ds-env.yml             # Conda environment (Python 3.12)
|   └── requirements.txt             # Pinned Python dependencies
├── data/                            # Local generated corpus
|   └── 2310-12345/                  # Example normalized arXiv ID
|       ├── metadata.json
|       ├── references.json
|       ├── v1/                      # Retained .tex/.bib files and subdirectories
|       └── v2/
├── .gitignore
└── README.md
```

---

## III. Modules and Functionality
### 1. Milestone 1
#### A. `src/utils.py`
Provides path resolution and workspace helper utilities:
- `DATA_DIR`: Resolves the absolute path to the root output directory (`data/`).
- `get_paper_dir(arxiv_id: str)`: Normalizes an arXiv identifier (converting `.` and `/` to `-`) and returns the designated output path within `data/`.

#### B. `src/get_metadata.py`
Fetches and stores bibliographic metadata:
- `get_entry_discovery(arXiv_id)`: Retrieves paper metadata including title, authors, submission date, revision dates, venue, categories, and abstract.
- `get_entry_discovery_batch(arXiv_ids)`: Same metadata for many papers with few requests (up to 100 IDs per arXiv API call); returns `None` for IDs arXiv does not have.
- `save_metadata(arXiv_id, metadata)`: Serializes and saves the metadata dictionary to `metadata.json` under the paper directory.

#### C. `src/get_full_source.py`
Downloads every version through `export.arxiv.org/src/<id>v<n>` and keeps only `.tex`/`.bib`:
- `Pacer`: at least 3 s between the *starts* of two arXiv requests (the download time counts towards the gap), shared by metadata and source calls; after HTTP 429/5xx it slows down (up to `MAX_REQUEST_DELAY`) and returns to 3 s once requests succeed again.
- `download_version(arxiv_id, version, pacer)`: streamed GET; a PDF-only version is recognised from `Content-Type` and its body is skipped.
- `extract_tex_bib(data, vdir)`: one in-memory pass that writes `.tex`/`.bib` (directory structure kept) and measures the whole archive ("size before removing figures"); archives and figures never touch the disk.
- `scrape_paper(arxiv_id, num_versions)`: every unfinished version, checkpointing the state file (`STATE_FILE`, default `data/source_state.json`) after each one, so an interrupted run resumes.

#### D. `src/get_extracting_reference.py`
Extracts citation network information:
- `get_extracting_references(arXiv_id)`: Queries academic citation APIs (such as Semantic Scholar) to retrieve referenced papers, mapping citing paper IDs to reference arXiv IDs, titles, authors, and publication dates.
- `save_extracting_references(arXiv_id, references)`: Writes the extracted reference mappings to `references.json`.

#### E. `src/scraper.ipynb`
End-to-end orchestration: entry discovery (batched arXiv API) → every version through `/src/` → Semantic Scholar references, plus time/RAM/disk measurement and the statistics exported for the Jupyter Book. The first cell picks which of the five ranges this machine crawls (`RUN_RANGES`), so each member runs their range on their own Colab instance; state/log folders named after the ranges let everyone's `data/` be merged without conflicts.

### 2. Milestone 2

The Milestone 2 parser is the package `src/m2/`, run by `src/parser.ipynb` (Colab or local) or by the
command line `python -m src.m2.run`. Per version it gathers the compilation path (root detection,
transitive includes, conditionals), builds the hierarchy (document → sections → … → sentences, block
formulas, figures, tables; lists and theorem-like blocks as containers), cleans and normalizes the
content, converts `\bibitem` entries to BibTeX, then deduplicates references (with `\cite` renaming
and field union) and elements across versions, and writes the `elements`, `hierarchy` and `bibtex`
tables. Modules, options, schemas and tests are documented in section VII.

---

## IV. Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Intro2DS-Group3-Source
```

### 2. Using VS Code - Set Up a Python Virtual Environment
Use Python 3.12+

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r src/requirements.txt
```

### 4. Using Conda

As an alternative to the virtual environment above, run these commands from the repository root in a terminal where Conda is available (for example, Anaconda Prompt):

```bash
conda env create -f src/intro2ds-env.yml
conda activate intro2ds
```

The environment file installs Python 3.12, pip, Notebook, ipykernel, and the pinned project dependencies. A separate `pip install -r src/requirements.txt` is not needed for this route.

To synchronize an existing environment after changes to the environment file:

```bash
conda env update -f src/intro2ds-env.yml --prune
```
---
> Note: You can run the jupyter notebook either by Conda or VS Code. We offer 2 options for you.
## V. Usage

### 1. Running on Google Colab (benchmark testbed)

1. Upload `scraper.ipynb` on Google Colab and select a **CPU** runtime.
2. Set `RUN_RANGES` to your assigned ranges (e.g. `[3]`); optionally set `SAMPLE_N = 20` for a trial run. Keep `RANGES` identical across group members.
3. Optionally add `S2_API_KEY` in Colab **Secrets** and enable notebook access.
4. Choose **Runtime > Run all**, authorize Google Drive access, and restart the session if prompted. Dependencies are installed automatically.
5. Data is saved to `MyDrive/Intro2DS-Group3/data` by default. Use a shared Drive folder or merge members' data afterward. Run all again in a new session to resume from saved checkpoints.
6. Once everyone's data is combined, set `REPORT_ALL_RANGES = True` and rerun to generate the group report statistics.

Outputs include paper metadata, references, versioned `.tex`/`.bib` sources, Parquet tables, checkpoints, run logs, and report statistics under `data/`. Additional settings are documented in the notebook's configuration cell.

### 2. Running locally using VS Code
In VS Code, open the notebook and use **Select Kernel** to choose the virtual environment has been created.

### 3. Running locally using Conda

From the repository root, activate the environment created during setup and open the collection notebook:

```bash
conda activate intro2ds
jupyter notebook src/scraper.ipynb
```

To open the Milestone 2 notebook in the same environment:

```bash
jupyter notebook src/parser.ipynb
```


When finished:

```bash
conda deactivate
```
---

> Note: You can run the jupyter notebook either by Conda or VS Code. We offer 2 options for you.
## VI. Dependencies

The project uses the following libraries specified in `src/requirements.txt`:
- [`arxiv`](https://pypi.org/project/arxiv/) (`==4.0.1`): Wrapper for the arXiv API.
- [`requests`](https://pypi.org/project/requests/) (`==2.34.0`): HTTP requests for scraping and API queries.
- [`psutil`](https://pypi.org/project/psutil/) (`==7.2.2`): Process and system resource monitoring.
- [`pandas`](https://pypi.org/project/pandas/) (`==3.1.0`) and [`pyarrow`](https://pypi.org/project/pyarrow/) (`==26.0.0`): Data tables and Parquet output.
- [`tqdm`](https://pypi.org/project/tqdm/) (`==4.70.1`): Progress bars.
- [`matplotlib`](https://pypi.org/project/matplotlib/) (`==3.11.2`): Quick sanity-check plots in the notebook.

---

## VII. Milestone 2 — Hierarchical Parsing and Standardization (`src/m2`)

The Milestone 2 parser lives in the package `src/m2/` (summarized in section III.2). It turns the Milestone 1 corpus (`data/<paper_key>/vN/*.tex|*.bib`, read-only) into the
three tables of the statement — `elements`, `hierarchy`, `bibtex` — plus internal working data for the
statistics. It never reads `references.json` (Milestone 3 must stay leakage free) and makes no network call.

### 1. Package layout

```text
src/m2/
├── config.py         # constants: command/environment/abbreviation lists, limits, PARSER_VERSION
├── io_utils.py       # decoding (utf-8 → cp1252 → latin-1), version folders, "(n)" Drive copies, hashes
├── discover.py       # papers and versions from data/, metadata.json and the M1 state (empty / missing)
├── gather.py         # root detection, transitive includes, conditionals, one string + source map
├── preprocess.py     # comments, \iffalse, verbatim masking, author macro expansion
├── scanner.py        # error-tolerant tokenizer (bounded brace matching, never hangs)
├── hierarchy.py      # rank-stack tree: sections, abstract, lists/items, blocks, leaves
├── clean.py          # LaTeX cleanup, accents, \cite / \ref normalization, footnotes
├── mathnorm.py       # inline → $...$, block → \begin{equation}...\end{equation} (aligned if multi-line)
├── sentences.py      # rule-based sentence splitter with masking and a byte-bounded cache
├── bib_parse.py      # tolerant .bib parser
├── bibitem.py        # \bibitem → BibTeX (\bibinfo, \newblock, regex) with confidence
├── bib_refine.py     # optional AnyStyle refinement (off by default)
├── bib_dedup.py      # reference dedup (union-find, vetoes, canonical key, field union), \cite renaming
├── dedup.py          # cross-version element dedup, element_id assignment
├── pipeline.py       # process_paper(job): fixed stage order, never raises
├── pool.py           # worker pool: watchdog, respawn, safe-mode retry, LPT, RSS recycling
├── storage.py        # Parquet shards, manifest, checkpoint signatures, compaction
├── run.py            # command-line entry point
├── validate.py       # structural checks + accuracy proxies
├── stats.py          # small CSV/JSON exports for the Jupyter Book
├── inspect_tree.py   # print one version's tree with file:line locations
├── profile_corpus.py # corpus profiling (Phase 1)
├── profile_run.py    # cProfile / pyinstrument profile of a sample
└── tests/            # pytest modules, gold sets (Python modules) and eval_gold.py
src/parser.ipynb      # Colab (and local) runner and benchmark notebook
```

### 2. Environment

Python 3.12 (tested with 3.12; the code needs ≥ 3.11 for possessive regex quantifiers).

```bash
# pip (venv)
pip install -r src/requirements.txt
# or conda
conda env create -f src/intro2ds-env.yml     # or: conda env update -f src/intro2ds-env.yml --prune
conda activate intro2ds
```

Runtime dependencies of the parser: `pyarrow`, `pandas`, `psutil`, `tqdm`, `rapidfuzz` (`plotly` for the
report). Development only: `pytest`, `pyinstrument`; `bibtexparser` and `wtpsplit` are used only for the
A/B comparisons.

### 3. Command line

Run from the repository root:

```bash
python -m src.m2.run --data-dir data --out data/tables --work-dir data/m2 --workers auto
```

| Option | Meaning |
|---|---|
| `--data-dir DIR` | Milestone 1 corpus root (read-only), default `data` |
| `--out DIR` | published tables (`elements/`, `hierarchy/`, `bibtex/`), default `data/tables` |
| `--work-dir DIR` | internal working data, default `data/m2` |
| `--workers auto\|N\|0` | `auto` = min(CPUs, free RAM / per-worker estimate, cgroup aware); `0` = in-process (debug) |
| `--limit K` / `--papers ID ...` / `--sample dev` | first K papers / given arXiv IDs or keys / the 200-paper dev sample |
| `--resume` | skip papers whose checkpoint signature (paths + sizes + contents + parser version) is unchanged |
| `--compact`, `--compact-only`, `--n-shards N` | merge shards into N deterministic files sorted by `arxiv_id` |
| `--timeout S` | per-paper deadline of the watchdog (default 60 s); a timed-out paper is retried once in safe mode |
| `--refine-bibitem` | optional AnyStyle refinement of low-confidence `\bibitem`s |

Then:

```bash
python -m src.m2.validate --out data/tables --work-dir data/m2 --data-dir data   # exit code 1 on violations
python -m src.m2.stats    --out data/tables --work-dir data/m2                   # report exports
python -m src.m2.inspect_tree 2608.04009 --version 1 --depth 3                   # debug one tree
```

### 4. Running on Google Colab (benchmark testbed)

1. Open `src/parser.ipynb` in Colab with a **CPU** runtime; set `REPO_URL`, `BRANCH` and the Drive
   paths in the first cell.
2. *Runtime → Run all*. The notebook installs the requirements, mounts Drive, packs `data/` once into a tar
   on Drive (without `references.json`) and extracts it to `/content` — this step is timed separately —
   then runs a 20-paper smoke test, the full run (`--resume`, so a new session continues where the previous
   one stopped), compaction, validation, statistics and prints the measurements (wall time, time per
   publication/version, peak RAM of parent + workers, output size, cache and reuse rates).
3. Results are copied back to `DRIVE_OUT`.

Locally the same notebook skips the Drive/clone steps and runs on a small sample of `data/`.

### 5. Optional AnyStyle refinement

Off by default; the parser works fully without Ruby. To enable:

```bash
sudo apt-get install ruby-full && gem install anystyle-cli
python -m src.m2.run ... --refine-bibitem
```

Only low-confidence `\bibitem` entries are sent; a returned field is accepted only if its text appears in
the original bibitem, and every change is logged.

### 6. Output schemas

Published tables (Parquet, zstd; `kind` and `entry_type` dictionary-encoded):

| Table | Columns |
|---|---|
| `elements` | `element_id` string, `arxiv_id` string, `kind` dictionary (`document, part, chapter, section, subsection, subsubsection, paragraph, subparagraph, abstract, list, item, block, sentence, formula, figure, table`), `content` string |
| `hierarchy` | `arxiv_id` string, `version` int32, `element_id` string, `parent_id` string — one row per node→parent edge, in pre-order |
| `bibtex` | `arxiv_id`, `citation_key`, `entry_type` dictionary, `title`, `authors` list<string>, `year` int32 nullable, `doi`, `journal`, `url` |

`element_id = <paper_key>#<prefix><n>` (`root<v>`, `h` headings, `a` abstract, `l` list, `i` item, `b` block,
`s` sentence, `f` formula, `fig`, `tab`). Identical cleaned content across versions shares one id; each
version remains a tree (identity = kind + content + occurrence index inside the version).

Internal working data in `--work-dir`: `parse_status/` (status `ok / root_fallback / partial / empty / missing /
failed`, reasons, file counts, per-stage times), `bibtex_full/` (all BibTeX fields, source, parse method,
confidence, versions, merged keys), `cite_key_map/`, `paper_stats/`, `manifest/` (checkpoints), `runs/`
(run summaries) and `report_exports/` (small CSV/JSON files for the book).

### 7. Tests and gold-set evaluation

```bash
python -m pytest src/m2/tests -q            # unit tests (inline LaTeX/BibTeX fixtures)
python -m src.m2.tests.eval_gold            # gold metrics: roots, headings, sentences, bibitems, ref-dedup
python -m src.m2.tests.eval_gold --sat      # + SaT sentence-splitting A/B (downloads the model once)
python -m src.m2.profile_run --sample dev   # profile the 200-paper dev sample
```
