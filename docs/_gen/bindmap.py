"""Recover the real structure of ``conquer3d._C`` from the pybind11 sources.

``_C.pyi`` is a flat namespace -- ``pybind.cpp`` hands every ``bind_*`` function
the same root module -- so reading the stub alone makes 29 unrelated operators
look like one undifferentiated pile of free functions. They are not: each is
defined in exactly one translation unit under ``csrc/binds/<category>/<file>.cpp``,
and that directory layout is the authoritative grouping.

This module parses those ``m.def(...)`` and ``py::class_<...>(m, "...")`` calls so
the API reference can present the native surface with the same sectioning the
source uses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

_DEF = re.compile(r'm\.def\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_CLASS = re.compile(r'py::class_\s*<[^>]*>\s*\(\s*m\s*,\s*"([A-Za-z_][A-Za-z0-9_]*)"')

#: Directory under ``csrc/binds`` -> (page slug, title, subtitle).
CATEGORIES: Dict[str, tuple] = {
    "ops": (
        "ops",
        "_C · Operators",
        "Isosurface extraction, flood fill, distance and volume-integral entry points.",
    ),
    "data_structure": (
        "data_structure",
        "_C · Data Structures",
        "Voxel grid construction, Z-curve ordering, and the native acceleration classes.",
    ),
    "primitive": (
        "primitive",
        "_C · Primitives",
        "Gaussian splatting operators and the geometric primitive types.",
    ),
    "creation": (
        "creation",
        "_C · Creation",
        "Procedural mesh generators bound from csrc/creation.",
    ),
}

CATEGORY_ORDER = ["ops", "data_structure", "primitive", "creation"]


_PBDOC = re.compile(r'R"pbdoc\((.*?)\)pbdoc"', re.DOTALL)
_PYARG = re.compile(r'py::arg\(\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)(?:\s*=\s*([^,\n]+))?')
_MEMBER = re.compile(r'\.(def|def_property_readonly|def_static|def_readonly)\(\s*"([A-Za-z_][A-Za-z0-9_]*)"')
#: A trailing plain-string docstring, e.g. .def("x", &X::x, "Does the thing.")
_TRAILING_STR = re.compile(r',\s*"((?:[^"\\]|\\.){8,})"\s*\)\s*$', re.DOTALL)
_INIT = re.compile(r"\.def\(\s*py::init")


def _class_block(text: str, start: int) -> str:
    """Source of a ``py::class_<...>(...)`` statement, up to its closing ``;``."""
    depth = 0
    i = text.index("(", start)
    n = len(text)
    while i < n:
        if text.startswith('R"pbdoc(', i):
            end = text.find(')pbdoc"', i)
            i = (end + len(')pbdoc"')) if end >= 0 else n
            continue
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == ";" and depth == 0:
            return text[start : i + 1]
        i += 1
    return text[start:]


@dataclass
class Binding:
    """Where a Python-visible native symbol is actually defined."""

    name: str
    category: str  # ops | data_structure | primitive | creation
    source_file: str  # repo-relative path to the binding .cpp
    source_line: int
    is_class: bool
    docstring: str = ""
    #: (arg name, default) pairs declared via ``py::arg``.
    args: tuple = ()


def _call_text(text: str, start: int) -> str:
    """Return the source of the ``m.def(...)`` call beginning at ``start``.

    Parentheses are balanced manually because the docstring is a raw string
    literal that routinely contains unbalanced ones -- ``(N, 3)`` and the like --
    so it is consumed atomically rather than scanned character by character.
    """
    i = text.index("(", start)
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith('R"pbdoc(', i):
            end = text.find(')pbdoc"', i)
            if end < 0:
                return text[start:]
            i = end + len(')pbdoc"')
            continue
        ch = text[i]
        if ch == '"':  # ordinary string literal
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
        i += 1
    return text[start:]


def scan(repo_root: Path) -> Dict[str, Binding]:
    """Map every ``_C`` symbol name to its binding translation unit.

    The binding sources -- not ``_C.pyi`` -- are the source of truth. The stub is
    regenerated at build time and can lag the C++ (``compute_flood_fill_cf`` is
    bound but missing from the committed stub), so anything found here that the
    stub lacks is still real API and must still be documented.
    """
    binds_dir = repo_root / "conquer3d" / "csrc" / "binds"
    out: Dict[str, Binding] = {}
    if not binds_dir.is_dir():
        return out

    for path in sorted(binds_dir.rglob("*.cpp")):
        category = path.parent.name
        if category not in CATEGORIES:
            category = "ops"
        rel = str(path.relative_to(repo_root))
        text = path.read_text(encoding="utf-8", errors="replace")

        for match in _DEF.finditer(text):
            name = match.group(1)
            if name in out:
                continue
            call = _call_text(text, match.start())
            doc_match = _PBDOC.search(call)
            args = tuple(
                (arg, (default or "").strip().rstrip(")").strip())
                for arg, default in _PYARG.findall(call)
            )
            out[name] = Binding(
                name=name,
                category=category,
                source_file=rel,
                source_line=text.count("\n", 0, match.start()) + 1,
                is_class=False,
                docstring=doc_match.group(1) if doc_match else "",
                args=args,
            )

        for match in _CLASS.finditer(text):
            name = match.group(1)
            class_call = _call_text(text, match.start())
            class_doc = _PBDOC.search(class_call)
            out[name] = Binding(
                name=name,
                category=category,
                source_file=rel,
                source_line=text.count("\n", 0, match.start()) + 1,
                is_class=True,
                # py::class_ carries its own pbdoc; the stub does not always keep it.
                docstring=class_doc.group(1) if class_doc else "",
            )
            # Members are keyed "Class.method" so a stub entry with no docstring
            # can fall back to the pbdoc actually written in the bindings.
            block = _class_block(text, match.start())

            # py::init carries no name string, but it is what becomes __init__.
            init = _INIT.search(block)
            if init:
                call = _call_text(block, init.start())
                doc_match = _PBDOC.search(call)
                out[f"{name}.__init__"] = Binding(
                    name="__init__",
                    category=category,
                    source_file=rel,
                    source_line=text.count("\n", 0, match.start() + init.start()) + 1,
                    is_class=False,
                    docstring=doc_match.group(1) if doc_match else "",
                    args=tuple(
                        (arg, (default or "").strip().rstrip(")").strip())
                        for arg, default in _PYARG.findall(call)
                    ),
                )

            for member in _MEMBER.finditer(block):
                member_name = member.group(2)
                call = _call_text(block, member.start())
                doc_match = _PBDOC.search(call)
                if not doc_match:
                    # A short string literal is the other documented form.
                    literal = _TRAILING_STR.search(call)
                    doc_text = literal.group(1) if literal else ""
                else:
                    doc_text = doc_match.group(1)
                out[f"{name}.{member_name}"] = Binding(
                    name=member_name,
                    category=category,
                    source_file=rel,
                    source_line=text.count("\n", 0, match.start() + member.start()) + 1,
                    is_class=False,
                    docstring=doc_text,
                    args=tuple(
                        (arg, (default or "").strip().rstrip(")").strip())
                        for arg, default in _PYARG.findall(call)
                    ),
                )

    return out


def category_of(name: str, bindings: Dict[str, Binding]) -> Optional[str]:
    binding = bindings.get(name)
    return binding.category if binding else None
