import os
from pathlib import Path
from dataclasses import dataclass, field


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


def list_files(directory) -> list[Path]:
    """Mọi tệp thường dưới ``directory``, sắp xếp theo đường dẫn.

    Tương đương ``sorted(p for p in directory.rglob("*") if p.is_file())``
    nhưng chỉ duyệt cây một lần bằng scandir: loại mục lấy sẵn từ lúc liệt kê
    nên không cần stat từng tệp, tiết kiệm nhiều trên ổ mạng như Google Drive.
    Không đi theo liên kết tượng trưng tới thư mục; liên kết tới tệp vẫn tính.
    """
    directory = Path(directory)
    files, stack = [], [directory]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(current / entry.name)
                        elif entry.is_file():
                            files.append(current / entry.name)
                    except OSError:
                        continue
        except OSError:
            if current == directory:
                raise
    return sorted(files)
