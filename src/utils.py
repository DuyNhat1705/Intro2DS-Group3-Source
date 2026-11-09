from pathlib import Path
from dataclasses import dataclass


@dataclass
class GatherTex:
    """The reachable LaTeX and bibliography files of one paper version."""

    root: Path
    source: str
    tex_files: list[Path]
    bib_files: list[Path]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_paper_dir(arxiv_id: str) -> Path:
    return DATA_DIR / arxiv_id.replace(".", "-").replace("/", "-")