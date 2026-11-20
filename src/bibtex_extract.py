"""Compatibility import for the canonical extraction implementation in bibtex.

Both module names reference the same module object so legacy monkeypatches,
including no-network worker patches, affect the functions actually executed.
"""
import sys

if __package__:
    from . import bibtex as _implementation
else:
    import bibtex as _implementation

sys.modules[__name__] = _implementation
