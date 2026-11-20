r"""Low-level I/O helpers: tolerant bytes->str decoding (utf-8, then cp1252/latin-1),
version-directory scanning, handling of Google-Drive ``(n)`` duplicate copies and content hashes.

Nothing in this module writes to the corpus: every function only reads.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

TEX_EXTS = (".tex",)
BIB_EXTS = (".bib",)

# "main(1).tex" -> stem "main", copy 1 ; "v1(2)" -> "v1", copy 2. A space before "(" is allowed.
_DUP_RE = re.compile(r"^(?P<stem>.+?) ?\((?P<n>\d{1,3})\)(?P<ext>\.[A-Za-z0-9]+)?$")
_VERSION_DIR_RE = re.compile(r"^v(?P<v>\d{1,3})(?: ?\((?P<k>\d{1,3})\))?$")


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode raw file bytes. Returns ``(text, encoding_used)``.

    UTF-8 (with optional BOM) is tried first; then cp1252, which is what most legacy
    Western sources use; latin-1 never fails and is the last resort.
    """
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return data.decode("cp1252"), "cp1252"
    except UnicodeDecodeError:
        return data.decode("latin-1"), "latin-1"


def read_text(path: Path) -> tuple[str, str]:
    """Read a file once and decode it with :func:`decode_bytes`."""
    with open(path, "rb") as fh:
        return decode_bytes(fh.read())


def content_hash(text: str | bytes) -> str:
    """Short, stable content hash (blake2b, 16 hex chars)."""
    if isinstance(text, str):
        text = text.encode("utf-8", "surrogatepass")
    return hashlib.blake2b(text, digest_size=8).hexdigest()


def split_dup_suffix(name: str) -> tuple[str, int]:
    """``"main(1).tex"`` -> ``("main.tex", 1)``; names without a copy suffix return copy 0."""
    m = _DUP_RE.match(name)
    if not m:
        return name, 0
    return m.group("stem") + (m.group("ext") or ""), int(m.group("n"))


def canonical_relpath(rel: str) -> tuple[str, bool]:
    """Strip ``(n)`` copy suffixes from every component of a relative POSIX path."""
    parts = rel.split("/")
    out, changed = [], False
    for p in parts:
        c, n = split_dup_suffix(p)
        changed |= n > 0
        out.append(c)
    return "/".join(out), changed


def parse_version_dir(name: str) -> tuple[int, int] | None:
    """``"v2"`` -> ``(2, 0)``, ``"v2(1)"`` -> ``(2, 1)``; anything else -> ``None``."""
    m = _VERSION_DIR_RE.match(name)
    if not m:
        return None
    return int(m.group("v")), int(m.group("k") or 0)


@dataclass
class VersionFiles:
    """Files of one version, keyed by relative POSIX path (as found on disk)."""

    files: dict[str, Path] = field(default_factory=dict)
    sizes: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    other_files: int = 0  # files that are neither .tex nor .bib

    @property
    def tex(self) -> list[str]:
        return sorted(r for r in self.files if r.lower().endswith(TEX_EXTS))

    @property
    def bib(self) -> list[str]:
        return sorted(r for r in self.files if r.lower().endswith(BIB_EXTS))


def _walk(root: Path) -> list[tuple[str, Path, int]]:
    out = []
    root_s = str(root)
    for dirpath, _dirnames, filenames in os.walk(root_s):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root_s).replace(os.sep, "/")
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            out.append((rel, Path(full), size))
    out.sort()
    return out


def scan_version_dirs(dirs: list[Path]) -> VersionFiles:
    """List the ``.tex``/``.bib`` files of a version.

    ``dirs[0]`` is the primary folder (``vN``) and the rest are Drive double uploads
    (``vN(k)``). A file of a secondary folder is added only when its relative path is not
    already present; collisions keep the primary copy. Every merge is noted.
    """
    vf = VersionFiles()
    for i, d in enumerate(dirs):
        if not d.is_dir():
            continue
        added = 0
        for rel, full, size in _walk(d):
            low = rel.lower()
            if not low.endswith(TEX_EXTS + BIB_EXTS):
                vf.other_files += 1
                continue
            if rel in vf.files:
                if i > 0:
                    vf.notes.append(f"dup_dir_collision:{d.name}/{rel}")
                continue
            vf.files[rel] = full
            vf.sizes[rel] = size
            added += i > 0
        if i > 0:
            vf.notes.append(f"merged_dir:{d.name}:+{added}")
    return vf


def signature_of(vf: VersionFiles) -> str:
    """Cheap signature of a version: relative paths + sizes + content hashes."""
    h = hashlib.blake2b(digest_size=12)
    for rel in sorted(vf.files):
        h.update(rel.encode("utf-8", "surrogatepass"))
        h.update(str(vf.sizes.get(rel, 0)).encode())
        try:
            with open(vf.files[rel], "rb") as fh:
                h.update(hashlib.blake2b(fh.read(), digest_size=8).digest())
        except OSError:
            h.update(b"?")
    return h.hexdigest()
