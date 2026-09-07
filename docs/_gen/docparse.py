"""Google-style docstring parser.

Conquer3D writes Google style everywhere -- in Python modules and, via
``R"pbdoc(...)pbdoc"``, in the C++ bindings too. The pybind flavour reaches us
through ``_C.pyi`` carrying the original C++ indentation (a blank first line then
8-12 leading spaces), so normalisation has to happen before section splitting;
:func:`inspect.cleandoc` handles exactly that asymmetry between the first line
and the rest.

Sections understood: Args/Arguments/Parameters, Returns/Return, Yields, Raises,
Example(s), Note(s), Warning(s), Attributes, References.
"""

from __future__ import annotations

import inspect
import re
from typing import List, Optional, Tuple

from model import Doc, Param, ReturnItem

# --------------------------------------------------------------------------- #
# Section handling
# --------------------------------------------------------------------------- #

_SECTION_ALIASES = {
    "args": "args",
    "arguments": "args",
    "parameters": "args",
    "params": "args",
    "keyword args": "args",
    "keyword arguments": "args",
    "returns": "returns",
    "return": "returns",
    "yields": "returns",
    "yield": "returns",
    "raises": "raises",
    "raise": "raises",
    "except": "raises",
    "exceptions": "raises",
    "example": "examples",
    "examples": "examples",
    "usage": "examples",
    "note": "notes",
    "notes": "notes",
    "warning": "warnings",
    "warnings": "warnings",
    "warns": "warnings",
    "attributes": "attributes",
    "members": "attributes",
    "references": "references",
    "reference": "references",
    "see also": "references",
}

# A section header is a bare "Word:" on its own line at column zero.
_HEADER_RE = re.compile(r"^([A-Za-z][A-Za-z ]{0,24}):\s*$")

# "name (type, optional): description"  /  "name: description"
_PARAM_RE = re.compile(
    r"^(?P<name>\*{0,2}[A-Za-z_][A-Za-z0-9_]*)\s*"
    r"(?:\((?P<type>[^)]*)\))?\s*:\s*(?P<desc>.*)$"
)

# "- name (type): description"  /  "- name: description"
_ITEM_RE = re.compile(
    r"^[-*+]\s+(?:\[?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\]?)?\s*"
    r"(?:\((?P<type>[^)]*)\))?\s*(?::\s*(?P<desc>.*))?$"
)

# "Defaults to X." trailing clause -> surfaced as the default column.
_DEFAULT_RE = re.compile(r"Defaults?\s+to\s+([^.]+)\.?\s*$", re.IGNORECASE)


def _split_sections(text: str) -> Tuple[str, List[Tuple[str, List[str]]]]:
    """Split a cleaned docstring into (preamble, [(section_key, body_lines)])."""
    lines = text.splitlines()
    preamble: List[str] = []
    sections: List[Tuple[str, List[str]]] = []
    current: Optional[List[str]] = None

    for line in lines:
        match = _HEADER_RE.match(line)
        key = _SECTION_ALIASES.get(match.group(1).strip().lower()) if match else None
        if key:
            current = []
            sections.append((key, current))
            continue
        (preamble if current is None else current).append(line)

    return "\n".join(preamble).strip(), sections


def _dedent_block(lines: List[str]) -> List[str]:
    """Strip the common leading indentation from a section body."""
    meaningful = [ln for ln in lines if ln.strip()]
    if not meaningful:
        return []
    indent = min(len(ln) - len(ln.lstrip()) for ln in meaningful)
    return [ln[indent:] if len(ln) >= indent else ln.strip() for ln in lines]


def _group_entries(lines: List[str]) -> List[List[str]]:
    """Group a parameter-style body into per-entry line runs.

    A new entry starts at indentation zero; deeper lines continue the previous
    entry, which is how Google style wraps long descriptions.
    """
    body = _dedent_block(lines)
    entries: List[List[str]] = []
    for line in body:
        if not line.strip():
            continue
        if line[:1].isspace() and entries:
            entries[-1].append(line.strip())
        else:
            entries.append([line.strip()])
    return entries


def _parse_params(lines: List[str]) -> List[Param]:
    out: List[Param] = []
    for entry in _group_entries(lines):
        head, *rest = entry
        match = _PARAM_RE.match(head)
        if not match:
            # Unparseable line: keep it as prose against the previous entry so
            # nothing silently disappears from the rendered page.
            if out:
                out[-1].description = f"{out[-1].description} {head}".strip()
            continue
        desc = " ".join([match.group("desc").strip(), *rest]).strip()
        raw_type = (match.group("type") or "").strip()

        optional = False
        if raw_type.endswith(", optional"):
            raw_type = raw_type[: -len(", optional")].strip()
            optional = True

        default = ""
        found = _DEFAULT_RE.search(desc)
        if found:
            default = found.group(1).strip().strip("`")
        elif optional:
            default = "optional"

        out.append(
            Param(
                name=match.group("name"),
                type=raw_type,
                default=default,
                description=desc,
            )
        )
    return out


def _parse_returns(lines: List[str]) -> Tuple[str, str, List[ReturnItem]]:
    """Parse a Returns body into (type, description, structured items)."""
    body = [ln for ln in _dedent_block(lines)]
    items: List[ReturnItem] = []
    rtype = ""
    desc_parts: List[str] = []

    # Bullet sub-items describe tuple/dict elements.
    bullet_idx = [i for i, ln in enumerate(body) if _ITEM_RE.match(ln.strip()) and ln.strip()[:1] in "-*+"]

    head_lines = body[: bullet_idx[0]] if bullet_idx else body
    head = " ".join(ln.strip() for ln in head_lines if ln.strip()).strip()

    if head:
        # "Tuple[a, b]: description" -- split on the first colon that is not
        # inside brackets, so subscripted generics survive intact.
        depth = 0
        cut = -1
        for i, ch in enumerate(head):
            if ch in "[(":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif ch == ":" and depth == 0:
                cut = i
                break
        if cut >= 0:
            rtype, tail = head[:cut].strip(), head[cut + 1 :].strip()
            if tail:
                desc_parts.append(tail)
        else:
            # Some entries use "type - description" instead of a colon.
            if " - " in head:
                rtype, tail = head.split(" - ", 1)
                rtype, tail = rtype.strip(), tail.strip()
                if tail:
                    desc_parts.append(tail)
            else:
                rtype = head

    if bullet_idx:
        for entry in _group_entries(body[bullet_idx[0] :]):
            text = " ".join(entry)
            match = _ITEM_RE.match(text)
            if match:
                items.append(
                    ReturnItem(
                        name=(match.group("name") or "").strip(),
                        type=(match.group("type") or "").strip(),
                        description=(match.group("desc") or "").strip(),
                    )
                )
            else:
                desc_parts.append(text)

    return rtype, " ".join(desc_parts).strip(), items


def _parse_block(lines: List[str]) -> str:
    return "\n".join(_dedent_block(lines)).strip()


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse(docstring: Optional[str]) -> Doc:
    """Parse a Google-style docstring into a :class:`~model.Doc`.

    Args:
        docstring: Raw docstring text, or None.

    Returns:
        Doc: Structured result. Unrecognised prose is preserved in
        ``description`` rather than dropped.
    """
    if not docstring or not docstring.strip():
        return Doc()

    text = inspect.cleandoc(docstring)
    preamble, sections = _split_sections(text)

    # First paragraph (up to a blank line) is the summary; the rest is detail.
    summary, description = "", ""
    if preamble:
        parts = preamble.split("\n\n", 1)
        summary = " ".join(parts[0].split())
        if len(parts) > 1:
            description = parts[1].strip()

    doc = Doc(summary=summary, description=description)

    for key, body in sections:
        if key == "args":
            doc.params.extend(_parse_params(body))
        elif key == "returns":
            rtype, rdesc, items = _parse_returns(body)
            # A symbol may document both Returns and Yields; keep the richer one.
            if rtype or rdesc or items:
                doc.returns_type = doc.returns_type or rtype
                doc.returns_description = (doc.returns_description + " " + rdesc).strip()
                doc.returns_items.extend(items)
        elif key == "raises":
            doc.raises.extend(_parse_params(body))
        elif key == "attributes":
            doc.attributes.extend(_parse_params(body))
        elif key == "examples":
            block = _parse_block(body)
            if block:
                doc.examples.append(block)
        elif key == "notes":
            block = _parse_block(body)
            if block:
                doc.notes.append(block)
        elif key == "warnings":
            block = _parse_block(body)
            if block:
                doc.warnings.append(block)
        elif key == "references":
            block = _parse_block(body)
            if block:
                doc.references.append(block)

    return doc
