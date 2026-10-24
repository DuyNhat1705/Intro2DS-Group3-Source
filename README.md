# Intro2DS-Group3-Source

> **Course**: CSC14119 - Introduction to Data Science  
> **Group**: Group 3  

---

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Modules and Functionality](#modules-and-functionality)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Dependencies](#dependencies)

---

## Milestone 1 Overview

The goal of this project is to build an automated pipeline to crawl and collect scientific literature data from arXiv and academic APIs (such as Semantic Scholar). For each target paper, the pipeline collects:

1. **Metadata**: Core bibliographic details including paper title, authors, submission date, revision history, publication venue, categories, and abstract.
2. **Full LaTeX Source Archives**: Source files (TeX documents, bibliography files `.bib`, styles, classes, and figures) across revisions (`v1`, `v2`, etc.).
3. **Extracted References**: Citations and linked reference information (citing paper ID, reference arXiv ID, Semantic Scholar ID, title, authors, and publication date).

---

## Directory Structure

The repository is organized into source scripts (`src/`) and the generated paper repository (`data/`):

```
Intro2DS-Group3-Source/
├── src/
│   ├── utils.py                     # Common utilities (path resolver, paper directory mapper)
│   ├── get_metadata.py              # arXiv metadata collection and persistence
│   ├── get_full_source.py           # Multi-version LaTeX source archive download & extraction
│   ├── get_extracting_reference.py   # Reference & citation data extraction
│   ├── scraper.ipynb                # Orchestration Jupyter notebook
│   └── requirements.txt             # Python package dependencies
├── data/                            # Output directory for crawled paper datasets
│   └── 2310-12345/                  # Paper folder (arXiv ID with '.' and '/' replaced by '-')
│       ├── metadata.json            # Bibliographic metadata
│       ├── references.json          # Extracted reference information
│       │
│       ├── v1/                      # LaTeX source tree for version 1
│       │   ├── main.tex
│       │   ├── sections/
│       │   │   ├── introduction.tex
│       │   │   └── methodology.tex
│       │   ├── references.bib
│       │   └── figures/
│       │
│       ├── v2/                      # LaTeX source tree for version 2
│       │   ├── main.tex
│       │   ├── sections/
│       │   │   ├── introduction.tex
│       │   │   └── methodology.tex
│       │   └── references.bib
│       │
│       └── v3/                      # LaTeX source tree for version 3
│           ├── main.tex
│           └── sections/
│               └── introduction.tex
├── .gitignore
└── README.md
```

---

## Modules and Functionality

### 1. `src/utils.py`
Provides path resolution and workspace helper utilities:
- `DATA_DIR`: Resolves the absolute path to the root output directory (`data/`).
- `get_paper_dir(arxiv_id: str)`: Normalizes an arXiv identifier (converting `.` and `/` to `-`) and returns the designated output path within `data/`.

### 2. `src/get_metadata.py`
Fetches and stores bibliographic metadata:
- `get_entry_discovery(arXiv_id)`: Retrieves paper metadata including title, authors, submission date, revision dates, venue, categories, and abstract.
- `get_entry_discovery_batch(arXiv_ids)`: Same metadata for many papers with few requests (up to 100 IDs per arXiv API call); returns `None` for IDs arXiv does not have.
- `save_metadata(arXiv_id, metadata)`: Serializes and saves the metadata dictionary to `metadata.json` under the paper directory.

### 3. `src/get_full_source.py`
Handles raw LaTeX source archives:
- `get_source(arXiv_id, num_versions)`: Downloads the gzipped tar archive containing TeX sources, bibliography `.bib` files, style/class files, figures, and compiled assets.
- `save_source(arXiv_id, version, source)`: Extracts and stores source files into version-specific subdirectories (e.g., `v1/`, `v2/`).

### 4. `src/get_extracting_reference.py`
Extracts citation network information:
- `get_extracting_references(arXiv_id)`: Queries academic citation APIs (such as Semantic Scholar) to retrieve referenced papers, mapping citing paper IDs to reference arXiv IDs, titles, authors, and publication dates.
- `save_extracting_references(arXiv_id, references)`: Writes the extracted reference mappings to `references.json`.

### 5. `src/scraper.ipynb`
Interactive Jupyter Notebook that orchestrates the scraping workflow, tying together metadata retrieval, source extraction, and reference mining.

---

## Installation and Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Intro2DS-Group3-Source
```

### 2. Set Up a Python Virtual Environment
Recommended Python version: 3.9+

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

---

## Usage

### Running on Google Colab (benchmark testbed)
`src/scraper.ipynb` is the end-to-end entry point of the scraper. It crawls all 25,000 assigned papers on one Colab CPU instance across several sessions and resumes from on-disk state.

1. Open [`src/scraper.ipynb` in Colab](https://colab.research.google.com/github/DuyNhat1705/Intro2DS-Group3-Source/blob/main/src/scraper.ipynb) and choose *Runtime → Change runtime type → CPU*.
2. Recommended: add a Semantic Scholar API key under *Secrets* named `S2_API_KEY` (enable notebook access).
3. *Runtime → Run all*. The notebook clones this repository and installs `src/requirements.txt`; if it asks for a restart, use *Runtime → Restart session* and *Run all* again.
4. Google Drive is mounted and `data/` is stored at `MyDrive/Intro2DS-Group3/data`, so progress survives across sessions.
5. A session crawls until everything is done, until `MAX_RUN_HOURS` (11.5 h), or until a service refuses requests; the analysis part then rebuilds the tables and statistics. Keep the browser tab open.
6. In each new session, open the notebook and *Run all* again; finished work is never repeated.

All parameters are in the single configuration cell (§2): `RANGES_TO_RUN`, `SAMPLE_N` (e.g. `20` for a trial run), `METADATA_BATCH_SIZE`, `ARXIV_DELAY` (≥ 3 s), `COOLDOWN_SEC` / `MAX_COOLDOWNS` (pause on persistent HTTP 429), `MAX_RUN_HOURS`, and `PERSIST_TO_DRIVE`.

Outputs: one folder per paper in `data/` (`metadata.json`, `references.json`, `v1/`, `v2/`, … with only `.tex`/`.bib` files), the tables `data/tables/{papers,references,references_all}.parquet`, run logs in `data/run_logs/`, and small statistics files for the Jupyter Book report in `data/report_exports/`.

### Running locally
```bash
pip install -r src/requirements.txt
jupyter lab src/scraper.ipynb
```
Open the notebook from inside the repository; `data/` is created at the repository root.

### Using as Python Modules
Import functions into custom scraping scripts or batch pipelines:

```python
from src.utils import get_paper_dir
from src.get_metadata import get_entry_discovery, save_metadata
from src.get_full_source import get_source, save_source
from src.get_extracting_reference import get_extracting_references, save_extracting_references

arxiv_id = "2310.12345"

# Fetch and store metadata
metadata = get_entry_discovery(arxiv_id)
save_metadata(arxiv_id, metadata)

# Fetch and store references
references = get_extracting_references(arxiv_id)
save_extracting_references(arxiv_id, references)
```

---

## Dependencies

The project uses the following libraries specified in `src/requirements.txt`:
- [`arxiv`](https://pypi.org/project/arxiv/) (`==4.0.1`): Wrapper for the arXiv API.
- [`requests`](https://pypi.org/project/requests/) (`==2.34.0`): HTTP requests for scraping and API queries.
- [`psutil`](https://pypi.org/project/psutil/) (`==7.2.2`): Process and system resource monitoring.
- [`pandas`](https://pypi.org/project/pandas/) (`==3.1.0`) and [`pyarrow`](https://pypi.org/project/pyarrow/) (`==26.0.0`): Data tables and Parquet output.
- [`tqdm`](https://pypi.org/project/tqdm/) (`==4.70.1`): Progress bars.
- [`matplotlib`](https://pypi.org/project/matplotlib/) (`==3.11.2`): Quick sanity-check plots in the notebook.
