"""Multi-file LaTeX gathering and cleanup interface for Milestone 2."""
import re
import warnings
from pathlib import Path
from utils import GatherTex

_COMMENT = re.compile()
_DOCUMENT_CLASS = re.compile()
_INCLUDE = re.compile()
_BIBLIOGRAPHY = re.compile()


def _strip_comments(source: str) -> str:
    pass


def _find_asset(
    name: str, extension: str, root_dir: Path, source_dir: Path, version_dir: Path
) -> Path | None:
    pass


def gather_tex(version_dir: Path, *, encoding: str = "utf-8-sig") -> GatherTex | None:
    """Expand static inputs in compilation order and list referenced .bib files.

    An existing directory without .tex files is an empty version. A missing
    directory, ambiguous main file, or unresolved include is a separate error.
    """
    pass


def cleanup_latex(source: str) -> str:
    """Normalize formatting and math while preserving structural commands and cites."""
    pass