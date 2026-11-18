import os
from pathlib import Path
from dataclasses import dataclass, field


def walk_files(root, *, ignore_missing=False):
    """Yield ``(path, stat)`` for every regular file under ``root``.

    One scandir pass; directory symlinks are not followed, file symlinks are.
    Each file is stat'ed for real (``entry.stat()`` follows the link and, on
    Linux, is not cached from the listing): Google Drive may list entries that
    cannot be opened, and those are skipped, as are unreadable subdirectories.
    Only an unreadable ``root`` raises, unless ``ignore_missing`` and the root
    simply does not exist.
    """
    root = Path(root)
    pending = [root]
    while pending:
        current = pending.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                            continue
                        if not entry.is_file():
                            continue
                        info = entry.stat()
                    except OSError:
                        # Vanished during the scan, or listed but unopenable (Drive).
                        continue
                    yield Path(entry.path), info
        except FileNotFoundError:
            if current == root and not ignore_missing:
                raise
        except OSError:
            if current == root:
                raise


def file_inventory(directory):
    """One directory scan, reusable by input inspection and TeX gathering."""
    return {path: {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            for path, info in walk_files(Path(directory).resolve())}


def list_files(directory) -> list[Path]:
    """Every regular file under ``directory``, sorted; same rules as ``walk_files``."""
    return sorted(path for path, _ in walk_files(directory))


@dataclass
class GatherTex:
    """The reachable LaTeX and bibliography files of one paper version.

    ``source`` is the whole compilation path expanded in order, with comments
    and inactive conditional branches removed but otherwise untouched (the
    BibTeX stage reads ``\\bibitem`` from it; the hierarchy stage reads
    ``cleanup_latex(source)``).
    """

    root: Path
    source: str
    tex_files: list[Path]
    bib_files: list[Path]
    # Includes and .bib files referenced by the sources but absent from the
    # version directory, as written in the command (e.g. "figs/plot.tikz").
    missing: list[str] = field(default_factory=list)
    # How the root file was chosen, for the parsing statistics.
    root_reason: str = ""
    # Recoverable problems: include cycles, encoding fallbacks, ...
    notes: list[str] = field(default_factory=list)
    # Source map: origins[i] is (file relative to the version directory,
    # 1-based line in that file) for line i of ``source``.
    # gather_clean_tex.locate(tex, offset) answers "where does this come from?".
    origins: list[tuple[str, int]] = field(default_factory=list)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_paper_dir(arxiv_id: str) -> Path:
    return DATA_DIR / arxiv_id.replace(".", "-").replace("/", "-")
