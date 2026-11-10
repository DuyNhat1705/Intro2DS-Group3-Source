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
|   ├── gather_clean_tex.py          # Milestone 2: gathering/cleanup interfaces
|   ├── hierarchy.py                 # Milestone 2: tree and element-table interfaces
|   ├── bibtex.py                    # Milestone 2: bibliography interfaces
|   ├── parser.ipynb                 # Milestone 2: notebook placeholder
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

The following interfaces describe the intended processing stages.

#### A. `src/gather_clean_tex.py`

- `gather_tex(version_dir, encoding="utf-8-sig")`: Intended to find the compilation root, expand reachable inputs in compilation order, and collect referenced bibliography files. Returns a `GatherTex` object; an existing directory without TeX files represents an empty version, distinct from a missing directory, ambiguous root, or unresolved include.
- `cleanup_latex(source)`: Intended to remove non-semantic formatting and normalize mathematics while preserving structural commands and citations.

#### B. `src/hierarchy.py`

- `Node`: Represents a node with a local identifier, optional parent identifier, kind, and content.
- `construct_hierarchy(cleaned_tex)`: Intended to build one version's tree, including a document root with no parent.
- `rewrite_citations(nodes, key_map)`: Intended to replace citation keys with canonical keys while preserving node identifiers and edges.
- `build_element_tables(arxiv_id, nodes_by_version)`: Intended to deduplicate cleaned content across versions and return shared `elements` rows and version-specific `hierarchy` rows. `ElementRow` and `HierarchyRow` define their schemas.

#### C. `src/bibtex.py`

- `BibEntry`: Holds a citation key, entry type, and bibliography fields.
- `extract_bibtex(tex)`: Intended to read referenced `.bib` files and convert reachable `\bibitem` entries.
- `merge_bibtex(entries_by_version)`: Intended to merge duplicate entries and union their fields, returning unified entries and a citation-key mapping for each version.
- `build_bibtex_rows(arxiv_id, entries)`: Intended to project merged entries into the final schema defined by `BibtexRow`.

#### D. `src/parser.ipynb`

Reserved for Milestone 2 orchestration. The intended order is gathering and cleanup per version, hierarchy and bibliography extraction, bibliography merging across versions, citation rewriting, then element deduplication and table assembly.

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