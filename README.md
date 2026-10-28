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
│   ├── get_full_source.py           # Every version through export.arxiv.org/src, in-memory .tex/.bib extraction
│   ├── get_extracting_reference.py   # Reference & citation data extraction
│   ├── scraper.ipynb                # Orchestration notebook (export only, one Colab per range)
│   ├── requirements.txt             # Python package dependencies
│   └── ExportS3/                    # Parked scraper v2: hybrid S3 bulk + export
│       ├── get_full_source_v2.py    # v2 export lane (shares helpers with get_full_source.py)
│       ├── s3_source.py             # Bulk source from arXiv's requester-pays S3 bucket
│       ├── scraper_v2.ipynb         # Orchestration notebook v2
│       └── requirements.txt         # Extra dependency of v2 (boto3)
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
Downloads every version through `export.arxiv.org/src/<id>v<n>` and keeps only `.tex`/`.bib`:
- `Pacer`: at least 3 s between the *starts* of two arXiv requests (the download time counts towards the gap), shared by metadata and source calls; after HTTP 429/5xx it slows down (up to `MAX_REQUEST_DELAY`) and returns to 3 s once requests succeed again.
- `download_version(arxiv_id, version, pacer)`: streamed GET; a PDF-only version is recognised from `Content-Type` and its body is skipped.
- `extract_tex_bib(data, vdir)`: one in-memory pass that writes `.tex`/`.bib` (directory structure kept) and measures the whole archive ("size before removing figures"); archives and figures never touch the disk.
- `scrape_paper(arxiv_id, num_versions)`: every unfinished version, checkpointing the state file (`STATE_FILE`, default `data/source_state.json`) after each one, so an interrupted run resumes.

### 4. `src/get_extracting_reference.py`
Extracts citation network information:
- `get_extracting_references(arXiv_id)`: Queries academic citation APIs (such as Semantic Scholar) to retrieve referenced papers, mapping citing paper IDs to reference arXiv IDs, titles, authors, and publication dates.
- `save_extracting_references(arXiv_id, references)`: Writes the extracted reference mappings to `references.json`.

### 5. `src/scraper.ipynb`
End-to-end orchestration: entry discovery (batched arXiv API) → every version through `/src/` → Semantic Scholar references, plus time/RAM/disk measurement and the statistics exported for the Jupyter Book. The first cell picks which of the five ranges this machine crawls (`RUN_RANGES`), so each member runs their range on their own Colab instance; state/log folders named after the ranges let everyone's `data/` be merged without conflicts.

### 6. `src/ExportS3/` (scraper v2, parked)
Hybrid variant that reads the v1 of single-version papers from the S3 bulk bucket and everything else through export. It is kept aside until the group has an AWS account that can read `s3://arxiv` (us-east-1).

`get_full_source_v2.py`: the export lane of v2 (same pacing and in-memory extraction as `get_full_source.py`, plus `via` per version and `store_fetched` for versions read from S3).

`s3_source.py`:
Reads source from the requester-pays bucket `s3://arxiv/src/`:
- `load_manifest` / `ChunkIndex`: which tar chunk holds which paper.
- `plan(...)`: splits papers between chunks and export (papers not on S3, `S3_MAX_GB` budget).
- `process_chunk(...)`: streams one chunk, stores the wanted papers, stops early once all were seen, and resumes a broken stream at the last member boundary (HTTP Range).

`scraper_v2.ipynb`: same measurement and analysis parts as `scraper.ipynb`; the source runs in two parallel lanes: S3 for papers that only have v1, export for the other versions and for anything S3 cannot serve. If S3 is unavailable (no key, no permission), every version goes through export.

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
`src/scraper.ipynb` is the end-to-end entry point of the scraper. The 25,000 assigned papers form five ranges; each member crawls the range assigned to them on their own Colab CPU instance (one connection and ≥ 3 s between arXiv requests per instance, about 5–6.5 h per range), and the notebook resumes from on-disk state across sessions.

1. Edit the first code cell (*Chọn range*): `RUN_RANGES` = the range numbers this machine crawls, e.g. `[3]`, `[1, 2]`, or `"all"` (one machine crawls every range, ~26 h over 3 sessions). `RANGES` maps the numbers 1–5 to `(YYMM, start, end)`; keep it identical on every member's copy.
2. Open [`src/scraper.ipynb` in Colab](https://colab.research.google.com/github/DuyNhat1705/Intro2DS-Group3-Source/blob/scraper/src/scraper.ipynb) and choose *Runtime → Change runtime type → CPU*.
3. Recommended: add a Semantic Scholar API key under *Secrets* named `S2_API_KEY` (enable notebook access). With a key the notebook paces Semantic Scholar at `S2_DELAY_WITH_KEY` (1 s) instead of the module's default 3 s; without a key nothing changes.
4. *Runtime → Run all*. The notebook clones (or pulls) this repository and installs `src/requirements.txt`; if it asks for a restart, use *Runtime → Restart session* and *Run all* again.
5. Google Drive is mounted and `data/` is stored at `DRIVE_DATA_DIR` (default `MyDrive/Intro2DS-Group3/data`). Recommended: one member creates `Intro2DS-Group3` on Drive and shares it (*Editor*) with the group, and everyone adds a shortcut to *My Drive*, so all data lands in one place. State and logs go to `data/.state/<WORKER>/` and `data/run_logs/<WORKER>/`, where `WORKER` is derived from the ranges (`R3`, `R1+R2`, `all`), and paper folders never overlap between ranges, so machines never overwrite each other's files (with separate Drives, simply copy the `data/` folders together).
6. A session crawls until everything is done, until `MAX_RUN_HOURS` (11.5 h), or until a service refuses requests; the analysis part then rebuilds the tables and statistics for this machine's ranges. Keep the browser tab open. In a new session, *Run all* again; finished work is never repeated.
7. Group report: once `data/` holds everyone's data, set `REPORT_ALL_RANGES = True` (§2) and run the notebook; Part IV then covers all five ranges and every machine's logs (`wall_time_sec` = longest machine, since they run in parallel; `machine_time_sec` = sum).

Other parameters are in the configuration cell (§2): `SAMPLE_N` (first cell, e.g. `20` for a trial run), `METADATA_BATCH_SIZE`, `S2_DELAY_WITH_KEY`, `ARXIV_DELAY` (≥ 3 s, measured between request starts), `ARXIV_MAX_DELAY` (adaptive slow-down after HTTP 429), `COOLDOWNS_SEC` (growing pauses, 5/10/15 min, on persistent HTTP 429), `SOURCE_WRITE_QUEUE` (versions waiting for the background extract-and-write thread), `MAX_RUN_HOURS`, `SOURCE_STATE_FLUSH_EVERY`, and `PERSIST_TO_DRIVE`.

Outputs: one folder per paper in `data/` (`metadata.json`, `references.json`, `v1/`, `v2/`, … with only `.tex`/`.bib` files), the tables `data/tables/{papers,references,references_all}.parquet`, per-machine state and run logs in `data/.state/<WORKER>/` and `data/run_logs/<WORKER>/`, and small statistics files for the Jupyter Book report in `data/report_exports/`.

### Running scraper v2 (hybrid S3 + export, parked)
`src/ExportS3/scraper_v2.ipynb` crawls all ranges on one instance (no range selection) and needs `pip install -r src/requirements.txt -r src/ExportS3/requirements.txt`, plus two more Colab Secrets:

1. `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` of an IAM user allowed to read S3 (e.g. `AmazonS3ReadOnlyAccess`). The bucket is *requester pays*: the group's AWS account pays the egress (~0.09 USD/GB). The `plan` line printed at the start of a run shows how many GB will be read and the estimated cost; `S3_MAX_GB` caps the total.
2. Optional: before setting `S3_LATEST_FOR_MULTI = True`, run `scraper_v2_prep/scripts/07_verify_s3_latest.py` and enable it only if every checked paper is `s3_is_latest`.

Without AWS keys the notebook still completes, with every version downloaded through export. Locally, set the two variables in the environment (or `~/.aws/credentials`) before starting Jupyter.

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
from src.get_full_source import scrape_paper
from src.get_extracting_reference import get_extracting_references, save_extracting_references

arxiv_id = "2310.12345"

# Fetch and store metadata
metadata = get_entry_discovery(arxiv_id)
save_metadata(arxiv_id, metadata)

# Fetch every version and keep only .tex/.bib (data/2310-12345/v1/, v2/, ...)
scrape_paper(arxiv_id, metadata["num_versions"])

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

Scraper v2 additionally needs [`boto3`](https://pypi.org/project/boto3/) (`==1.43.119`, in `src/ExportS3/requirements.txt`) for the S3 bulk source.
