from pathlib import Path
import os


def walk_files(root, *, ignore_missing=False):
    """Yield paths and cached scandir stats without following directory symlinks."""
    pending = [Path(root)]
    while pending:
        try:
            with os.scandir(pending.pop()) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            pending.append(Path(entry.path))
                        elif entry.is_file():
                            yield Path(entry.path), entry.stat()
                    except FileNotFoundError:
                        if not ignore_missing:
                            raise
        except FileNotFoundError:
            if not ignore_missing:
                raise


def file_inventory(directory):
    """One directory scan, reusable by input inspection and TeX gathering."""
    return {path: {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
            for path, info in walk_files(Path(directory).resolve())}


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def get_paper_dir(arxiv_id: str) -> Path:
    return DATA_DIR / arxiv_id.replace(".", "-").replace("/", "-")
