"""Shared data model for the Conquer3D documentation generator.

Every extractor -- the Python ``ast`` reader in :mod:`pyapi` and the Doxygen XML
reader in :mod:`nativeapi` -- normalises its findings into the structures defined
here. The renderer then consumes only this model, which is what allows a CUDA
``__global__`` kernel and a pure-Python autograd wrapper to be presented with the
same visual grammar.

Tiers follow the audit layering used throughout the site:

==== ===========================================================
T1   Python API (``conquer3d.*``)
T2   Native symbols exposed through pybind11 (``conquer3d._C``)
T3   Host dispatchers declared in ``csrc/**/*.h``
T4   CUDA ``__global__`` kernels
T5   CUDA ``__device__`` helpers
T6   Inline math primitives in ``csrc/maths/``
T7   ``__constant__`` lookup tables in ``csrc/ops/*_data.h``
==== ===========================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# --------------------------------------------------------------------------- #
# Tier metadata
# --------------------------------------------------------------------------- #

TIERS: Dict[str, Dict[str, str]] = {
    "T1": {
        "name": "Python API",
        "slug": "python",
        "blurb": "The public surface you import: operators, data structures, primitives, I/O.",
    },
    "T2": {
        "name": "Native Bindings",
        "slug": "bindings",
        "blurb": "Classes and functions pybind11 exposes on conquer3d._C.",
    },
    "T3": {
        "name": "Host Dispatchers",
        "slug": "dispatchers",
        "blurb": "C++ launch entry points that validate tensors and configure grids.",
    },
    "T4": {
        "name": "CUDA Kernels",
        "slug": "kernels",
        "blurb": "__global__ kernels -- the parallel work itself.",
    },
    "T5": {
        "name": "Device Helpers",
        "slug": "device",
        "blurb": "__device__ functions called from inside kernels.",
    },
    "T6": {
        "name": "Math Primitives",
        "slug": "maths",
        "blurb": "Inline vector, matrix and QEF routines shared across kernels.",
    },
    "T7": {
        "name": "Lookup Tables",
        "slug": "tables",
        "blurb": "__constant__ topology tables driving isosurface extraction.",
    },
}

TIER_ORDER: List[str] = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]


# --------------------------------------------------------------------------- #
# Docstring pieces
# --------------------------------------------------------------------------- #


@dataclass
class Param:
    """A single documented parameter."""

    name: str
    type: str = ""
    default: str = ""
    description: str = ""
    direction: str = ""  # Doxygen [in] / [out] / [in,out]


@dataclass
class ReturnItem:
    """One element of a structured (tuple/dict) return value."""

    name: str = ""
    type: str = ""
    description: str = ""


@dataclass
class Doc:
    """A parsed docstring, language-agnostic."""

    summary: str = ""
    description: str = ""
    params: List[Param] = field(default_factory=list)
    returns_type: str = ""
    returns_description: str = ""
    returns_items: List[ReturnItem] = field(default_factory=list)
    raises: List[Param] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    attributes: List[Param] = field(default_factory=list)
    references: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.summary or self.description or self.params or self.returns_description)


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #


@dataclass
class Symbol:
    """One documented entity, at any tier and in any language."""

    name: str
    kind: str  # function | class | method | property | module | kernel | device
    #        | inline | struct | table | macro | constant | typedef
    tier: str
    qualname: str = ""
    signature: str = ""
    doc: Doc = field(default_factory=Doc)
    source_file: str = ""
    source_line: int = 0
    language: str = "python"  # python | cuda | cpp
    bases: List[str] = field(default_factory=list)
    flags: List[str] = field(default_factory=list)  # __global__, static, inline, autograd...
    children: List["Symbol"] = field(default_factory=list)
    anchor_override: str = ""

    @property
    def anchor(self) -> str:
        if self.anchor_override:
            return self.anchor_override
        base = self.qualname or self.name
        return slugify(base)

    @property
    def documented(self) -> bool:
        """Whether this symbol carries real prose, not just a signature."""
        return bool(self.doc.summary or self.doc.description)

    def walk(self):
        """Yield this symbol and every descendant, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Group:
    """A rendered page: a Python module, or a native file/namespace."""

    title: str
    slug: str
    tier: str
    symbols: List[Symbol] = field(default_factory=list)
    doc: Doc = field(default_factory=Doc)
    source_file: str = ""
    subtitle: str = ""
    #: Sibling pages belonging to the same section, as (title, href, count).
    related: List[tuple] = field(default_factory=list)

    @property
    def all_symbols(self) -> List[Symbol]:
        out: List[Symbol] = []
        for sym in self.symbols:
            out.extend(sym.walk())
        return out

    @property
    def counts(self) -> tuple:
        """(documented, total) across every documentable symbol on the page.

        Module/file band headers are structural, not API surface, so counting
        them would understate real coverage.
        """
        # Stale stub entries describe symbols the sources no longer bind, so
        # they are reported on the page but excluded from the coverage ratio.
        syms = [
            s for s in self.all_symbols
            if s.kind != "module" and "stale stub entry" not in s.flags
        ]
        return sum(1 for s in syms if s.documented), len(syms)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_SLUG_STRIP = re.compile(r"[^a-zA-Z0-9._-]+")
_SLUG_DASH = re.compile(r"-{2,}")


def slugify(text: str) -> str:
    """Convert a qualified name into a stable, URL-safe anchor."""
    out = _SLUG_STRIP.sub("-", text.strip())
    out = _SLUG_DASH.sub("-", out).strip("-.")
    return out.lower() or "item"
