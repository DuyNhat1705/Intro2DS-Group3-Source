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
- [Running the Tests](#vii-running-the-tests)

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
|   ├── gather_clean_tex.py          # Milestone 2: compilation path and LaTeX cleanup
|   ├── hierarchy.py                 # Milestone 2: document tree, sentences, element tables
|   ├── bibtex.py                    # Milestone 2: .bib/\bibitem extraction and unification
|   ├── m2_types.py                  # Milestone 2: BibtexEntry data class
|   ├── parser_pipeline.py           # Milestone 2: publication discovery and per-paper parsing
|   ├── parser_parallel.py           # Milestone 2: RAM-aware parallel workers
|   ├── parser_reporting.py          # Milestone 2: checkpoints, Parquet tables, statistics
|   ├── bibtex_refine.py             # Milestone 2: optional AnyStyle post-processing
|   ├── parser.ipynb                 # Milestone 2: parsing and reporting notebook
|   ├── intro2ds-env.yml             # Conda environment (Python 3.12)
|   └── requirements.txt             # Pinned Python dependencies
├── tests/                           # Milestone 2 automated tests (pytest)
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

Each paper is parsed from its own LaTeX sources only. The crawled reference metadata (`references.json`) is never read by the parser, because Milestone 3 matches the `bibtex` entries against it.

#### A. `src/gather_clean_tex.py`

- `gather_tex(version_dir)`: Finds the root file (the one declaring `\documentclass`; copies with the same content outside comments, such as the `main(1).tex` that Google Drive creates when a folder is uploaded twice, count as one candidate), expands active `\input`/`\include` commands transitively in compilation order, and lists the `.bib` files named by `\bibliography`. Returns a `GatherTex` with the expanded source, a source map, missing includes, and notes; returns `None` for a version without TeX files (an empty version).
- `cleanup_latex(source)`: Removes non-semantic formatting (`\centering`, float placement options, table rules, page headers, front-matter commands), expands simple user macros, and normalizes inline math to `$...$` and block math to `\begin{equation}...\end{equation}`.

#### B. `src/hierarchy.py`

- `construct_hierarchy(cleaned_tex)`: Builds one version's tree: the document root (holding the paper title), headings at their detected levels, lists and items, and leaves for sentences, block formulas, and figures/tables. Sentences are split with the SaT model (`sat-3l-sm`, CPU); mathematics, commands, abbreviations, URLs, and decimals are protected. The References section is excluded; acknowledgements and appendices are kept.
- `rewrite_citations(nodes, key_map)`: Renames `\cite` keys to the canonical keys chosen during bibliography unification.
- `build_element_tables(arxiv_id, nodes_by_version)`: Returns the `elements` rows (leaves with identical whitespace-normalized content share one ID across versions; containers are additionally matched by occurrence order, so two same-titled headings or two lists of one version never merge) and the per-version `hierarchy` edges.

#### C. `src/bibtex.py`

- `extract_bibtex(tex)`: Parses the referenced `.bib` files with `bibtexparser` and converts each `\bibitem` with rules for `\newblock` blocks, quoted/italic titles, and author–year styles. Titles, journals, and authors are converted from LaTeX to Unicode; authors are written as `First Last`; arXiv IDs and years are read from the fields.
- `unify_publication(arxiv_id, entries_by_version)`: Merges duplicates within and across versions (DOI/arXiv ID, normalized title, citation key, and fuzzy title with a DOI veto), unions their fields, chooses one citation key per entry, and returns the key map used by `rewrite_citations`.

#### D. `src/parser_pipeline.py`, `src/parser_parallel.py`, `src/parser_reporting.py`

- `discover_publications(data_dir, target_ids=None)`: Lists papers and versions from the Milestone 1 folders and download state; IDs without data are kept so coverage statistics stay complete.
- `parse_publication(job)`: Runs gathering, cleanup, hierarchy, bibliography extraction, unification, citation rewriting, and element tables for every version of one paper, recording per-stage timings, warnings, and statistics.
- `run_parser(jobs, output_dir, ...)`: Parses papers sequentially or with RAM-aware parallel workers (`iter_parallel_results`), skips papers whose checkpoint matches the source/code signature, writes per-paper partitions and the three combined Parquet tables, and exports the statistics (`per_paper.csv`, `per_version.csv`, `summary_stats.json`, ...).

#### E. `src/bibtex_refine.py`

- `refine_bibliography(output_dir, worker=...)`: Optional post-processing after `run_parser`. Sends only the `\bibitem` entries that look mis-split (no title or authors, a venue taken as the title, overly long author names) to AnyStyle in one batch and accepts a title only if it occurs in the original text. Results are cached; the unrefined table is kept as `bibtex_parsed.parquet` and every change is logged in `bibtex_refinement.jsonl`.

#### F. `src/parser.ipynb`

Configures the scope (ranges, sample size, worker name), mounts Drive, optionally extracts a data tarball to the local disk, runs a smoke test, runs the parser and the optional AnyStyle step, and renders the coverage, deduplication, and performance statistics used in the report.

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

**Milestone 2 (`parser.ipynb`) on Colab:**

1. Open `src/parser.ipynb` on Colab with a **CPU** runtime. `BRANCH` selects the branch that the notebook clones.
2. Set `RUN_RANGES` (or `"local"` to parse whatever is in the data folder) and optionally `SAMPLE_N` for a trial run.
3. Optional: put `data.tar.gz` (or `.tgz`/`.tar`) in `MyDrive/Intro2DS-Group3`; the notebook extracts it to the local disk and reads from there, which is much faster than reading many small files from Drive. The extraction time is recorded in `summary_stats.json` (`run.data_preparation`).
4. Choose **Runtime > Run all**. Outputs go to `MyDrive/Intro2DS-Group3/statistics_data/parser`. Running again resumes from checkpoints; `VERIFY_CHECKPOINTS = "full"` re-hashes every output file for a final check.
5. `REFINE_WITH_ANYSTYLE = True` installs Ruby and `anystyle-cli` on Colab (a few minutes) for the optional post-processing step; set it to `False` to skip.

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

## VI. Dependencies

The project uses the following libraries specified in `src/requirements.txt`:
- [`arxiv`](https://pypi.org/project/arxiv/) (`==4.0.1`): Wrapper for the arXiv API.
- [`requests`](https://pypi.org/project/requests/) (`==2.34.0`): HTTP requests for scraping and API queries.
- [`psutil`](https://pypi.org/project/psutil/) (`==7.2.2`): Process and system resource monitoring.
- [`pandas`](https://pypi.org/project/pandas/) (`==3.1.0`) and [`pyarrow`](https://pypi.org/project/pyarrow/) (`==26.0.0`): Data tables and Parquet output.
- [`tqdm`](https://pypi.org/project/tqdm/) (`==4.70.1`): Progress bars.
- [`bibtexparser`](https://pypi.org/project/bibtexparser/) (`==1.4.3`): `.bib` parsing and LaTeX-to-Unicode conversion.
- [`wtpsplit`](https://pypi.org/project/wtpsplit/) (`==2.2.2`) and [`onnxruntime`](https://pypi.org/project/onnxruntime/) (`==1.31.1`): SaT sentence segmentation on CPU.
- [`matplotlib`](https://pypi.org/project/matplotlib/) (`==3.11.2`): Report figures and sanity-check plots in the notebooks.

---

## VII. Running the Tests

The Milestone 2 parser has an automated test suite in `tests/` (pytest). `pytest` is not part of `src/requirements.txt`, so install it in the environment created above, then run the suite from the repository root:

```bash
pip install pytest
python -m pytest tests -q
```

The tests replace the SaT sentence model with a stub and disable the optional network lookups, so they run offline and do not download any model.