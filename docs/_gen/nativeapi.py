"""Native (CUDA/C++) extraction via Doxygen XML.

Doxygen does the parsing rather than a hand-rolled regex pass: ``csrc/`` contains
templates, lambdas, anonymous namespaces, macro-qualified declarations and an
80-method class, and the headers are already ~90% annotated with Doxygen syntax,
so its XML is both the correct and the cheapest front-end.

The XML is normalised into the same :mod:`model` structures the Python extractor
produces, which is what lets a ``__global__`` kernel render with exactly the same
grammar as a Python function.

Tier assignment:

* **T7** -- ``csrc/ops/*_data.h`` (``__constant__`` topology tables)
* **T6** -- ``csrc/maths/*.h`` (inline vector/matrix/QEF routines)
* **T4** -- any member whose declaration carries ``__global__``
* **T5** -- any remaining member carrying ``__device__``
* **T3** -- everything else: host dispatchers, classes and structs
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

from model import Doc, Group, Param, Symbol, slugify

DOXYFILE = """
PROJECT_NAME           = Conquer3D
INPUT                  = {input_dir}
OUTPUT_DIRECTORY       = {output_dir}
RECURSIVE              = YES
FILE_PATTERNS          = *.cu *.cuh *.cpp *.h *.hpp *.c
EXTENSION_MAPPING      = cu=C++ cuh=C++
GENERATE_XML           = YES
GENERATE_HTML          = NO
GENERATE_LATEX         = NO
XML_PROGRAMLISTING     = NO
EXTRACT_ALL            = YES
EXTRACT_STATIC         = YES
EXTRACT_PRIVATE        = NO
EXTRACT_LOCAL_CLASSES  = YES
EXTRACT_ANON_NSPACES   = YES
HIDE_UNDOC_MEMBERS     = NO
HIDE_UNDOC_CLASSES     = NO
MACRO_EXPANSION        = YES
EXPAND_ONLY_PREDEF     = YES
ENABLE_PREPROCESSING   = YES
PREDEFINED             = __restrict__= \\
                         __CUDACC__ \\
                         WITH_CUDA
OPTIMIZE_OUTPUT_FOR_C  = NO
QUIET                  = YES
WARNINGS               = NO
WARN_IF_UNDOCUMENTED   = NO
WARN_IF_DOC_ERROR      = NO
WARN_NO_PARAMDOC       = NO
SORT_MEMBER_DOCS       = NO
SHOW_INCLUDE_FILES     = NO
CASE_SENSE_NAMES       = YES
# Include guards and function-local helper macros are implementation
# scaffolding rather than API surface, so they are not counted or rendered.
EXCLUDE_SYMBOLS        = *_H *_CUH *_HPP TEST_AXIS
"""


# --------------------------------------------------------------------------- #
# Doxygen invocation
# --------------------------------------------------------------------------- #


def run_doxygen(repo_root: Path, docs_dir: Path,
                scratch: Optional[Path] = None) -> Optional[Path]:
    """Run Doxygen over ``csrc/`` and return the XML output directory.

    Args:
        repo_root: Source tree to document -- the working tree, or a checkout
            of a tag.
        docs_dir: Site directory, used only to site the default scratch dir.
        scratch: Where to put the Doxyfile and XML. Each source tree needs its
            own: Doxygen does not remove XML for files that have disappeared,
            and the caller globs the directory, so a shared scratch folds one
            tree's deleted symbols into the next tree's page set.
    """
    doxygen = shutil.which("doxygen")
    if not doxygen:
        # Conda installs it outside the active environment's bin, so an activated
        # project env can hide it from PATH entirely.
        for candidate in (
            Path.home() / "anaconda3" / "bin" / "doxygen",
            Path.home() / "miniconda3" / "bin" / "doxygen",
            Path("/usr/bin/doxygen"),
        ):
            if candidate.exists():
                doxygen = str(candidate)
                break
    if not doxygen:
        raise RuntimeError(
            "doxygen not found -- install it (conda install -c conda-forge doxygen) "
            "to build tiers T3-T7"
        )

    build_dir = scratch if scratch is not None else docs_dir / "_build" / "doxygen"
    # Cleared rather than merged, for the reason given above.
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    config = DOXYFILE.format(
        input_dir=repo_root / "conquer3d" / "csrc",
        output_dir=build_dir,
    )
    config_path = build_dir / "Doxyfile"
    config_path.write_text(config, encoding="utf-8")

    result = subprocess.run(
        [doxygen, str(config_path)],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    xml_dir = build_dir / "xml"
    if not (xml_dir / "index.xml").exists():
        raise RuntimeError(f"doxygen produced no XML: {result.stderr[:400]}")
    return xml_dir


# --------------------------------------------------------------------------- #
# XML helpers
# --------------------------------------------------------------------------- #


def _text(node: Optional[ET.Element]) -> str:
    """Flatten a Doxygen description node to plain text.

    Doxygen nests prose in <para>, <ref>, <computeroutput> and friends; the
    markers below are re-read by the site's inline markdown pass, so formulas
    and identifiers survive into the rendered page.
    """
    if node is None:
        return ""
    parts: List[str] = []

    def walk(el: ET.Element) -> None:
        tag = el.tag
        if tag == "computeroutput":
            inner = "".join(el.itertext()).strip()
            parts.append(f"`{inner}`")
            if el.tail:
                parts.append(el.tail)
            return
        if tag in ("formula",):
            inner = "".join(el.itertext()).strip()
            parts.append(inner if inner.startswith("$") else f"${inner}$")
            if el.tail:
                parts.append(el.tail)
            return
        if tag in ("parameterlist", "simplesect", "xrefsect"):
            return  # handled separately
        if tag == "para":
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n\n")
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
        if el.tail:
            parts.append(el.tail)

    walk(node)
    text = "".join(parts)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def _simplesects(detail: Optional[ET.Element], kind: str) -> List[str]:
    if detail is None:
        return []
    out = []
    for sect in detail.iter("simplesect"):
        if sect.get("kind") == kind:
            body = "".join(_text(p) for p in sect.findall("para")) or _text(sect)
            if body.strip():
                out.append(body.strip())
    return out


def _doc_params(detail: Optional[ET.Element]) -> List[Param]:
    if detail is None:
        return []
    out: List[Param] = []
    for plist in detail.iter("parameterlist"):
        if plist.get("kind") != "param":
            continue
        for item in plist.findall("parameteritem"):
            names = item.findall("./parameternamelist/parametername")
            desc = _text(item.find("parameterdescription"))
            for name in names:
                direction = name.get("direction", "") or ""
                out.append(
                    Param(
                        name="".join(name.itertext()).strip(),
                        description=desc,
                        direction=f"[{direction}]" if direction else "",
                    )
                )
    return out


def _raises(detail: Optional[ET.Element]) -> List[Param]:
    if detail is None:
        return []
    out = []
    for plist in detail.iter("parameterlist"):
        if plist.get("kind") not in ("exception", "retval"):
            continue
        for item in plist.findall("parameteritem"):
            for name in item.findall("./parameternamelist/parametername"):
                out.append(
                    Param(
                        name="".join(name.itertext()).strip(),
                        description=_text(item.find("parameterdescription")),
                    )
                )
    return out


def build_doc(member: ET.Element) -> Doc:
    """Assemble a :class:`~model.Doc` from a Doxygen memberdef/compounddef."""
    brief = _text(member.find("briefdescription"))
    detail_node = member.find("detaileddescription")
    detail = _text(detail_node)

    doc = Doc(summary=brief, description=detail)
    doc.params = _doc_params(detail_node)
    doc.raises = _raises(detail_node)
    doc.notes = _simplesects(detail_node, "note")
    doc.warnings = _simplesects(detail_node, "warning")
    doc.references = _simplesects(detail_node, "see")

    returns = _simplesects(detail_node, "return")
    if returns:
        doc.returns_description = "\n\n".join(returns)
        rtype = member.findtext("type", default="").strip()
        if rtype and rtype not in ("void",):
            doc.returns_type = rtype

    # A brief-only symbol reads better with the brief promoted to summary and
    # nothing repeated underneath it.
    if doc.description == doc.summary:
        doc.description = ""
    return doc


# --------------------------------------------------------------------------- #
# Tier classification
# --------------------------------------------------------------------------- #


def classify(location: str, definition: str, argsstring: str, kind: str) -> tuple:
    """Return (tier, symbol_kind) for a native member."""
    decl = f"{definition} {argsstring}"
    name = Path(location).name

    if name.endswith("_data.h"):
        return "T7", "table"
    if "/maths/" in location:
        return "T6", "inline"
    if "__global__" in decl:
        return "T4", "kernel"
    if "__device__" in decl:
        return "T5", "device"
    if kind == "variable":
        return ("T7", "table") if "__constant__" in decl else ("T3", "constant")
    if kind == "define":
        return "T3", "macro"
    if kind == "typedef":
        return "T3", "typedef"
    return "T3", "function"


# Page grouping: which file a symbol's page is keyed on.
_TIER_TITLES = {
    "T3": "Native declarations",
    "T4": "CUDA kernels",
    "T5": "Device helpers",
    "T6": "Math primitives",
    "T7": "Lookup tables",
}


def _page_key(tier: str, location: str) -> tuple:
    """(slug, title, subtitle) for the page a symbol belongs to."""
    path = Path(location)
    rel = location

    if tier == "T7":
        return ("tables", "Constant Tables", "Topology lookup tables driving extraction.")
    if tier == "T6":
        return (
            slugify(path.stem),
            f"maths/{path.name}",
            "Inline host/device math routines.",
        )
    if tier == "T3":
        parent = path.parent.name or "csrc"
        return (
            slugify(parent),
            f"csrc/{parent}",
            f"C++ declarations, dispatchers and classes in csrc/{parent}.",
        )
    # T4 / T5 are keyed on the implementation file they live in.
    return (
        slugify(path.stem),
        path.name,
        f"{_TIER_TITLES[tier]} implemented in {rel}.",
    )


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def _rel(location: str, repo_root: Path) -> str:
    try:
        return str(Path(location).resolve().relative_to(repo_root.resolve()))
    except Exception:
        idx = location.find("conquer3d/csrc")
        return location[idx:] if idx >= 0 else location


def _member_symbol(
    member: ET.Element, repo_root: Path, parent: str = ""
) -> Optional[Symbol]:
    kind = member.get("kind", "")
    if kind not in ("function", "variable", "define", "typedef", "enum"):
        return None

    name = (member.findtext("name") or "").strip()
    if not name or name.startswith("@"):
        return None

    loc_node = member.find("location")
    location = loc_node.get("file", "") if loc_node is not None else ""
    line = int(loc_node.get("line", 0) or 0) if loc_node is not None else 0

    definition = (member.findtext("definition") or "").strip()
    argsstring = (member.findtext("argsstring") or "").strip()
    rtype = (member.findtext("type") or "").strip()

    tier, sym_kind = classify(location, f"{rtype} {definition}", argsstring, kind)

    flags = []
    for marker in ("__global__", "__device__", "__host__", "__constant__", "__forceinline__"):
        if marker in f"{rtype} {definition}":
            flags.append(marker)
    if member.get("static") == "yes":
        flags.append("static")
    if member.get("inline") == "yes" and "__forceinline__" not in flags:
        flags.append("inline")

    clean_type = rtype
    for marker in ("__global__", "__device__", "__host__", "__forceinline__", "static", "inline"):
        clean_type = clean_type.replace(marker, "")
    clean_type = " ".join(clean_type.split())

    signature = f"{argsstring}" if kind == "function" else ""
    if clean_type:
        signature = f"{signature} -> {clean_type}" if signature else f": {clean_type}"

    qualname = f"{parent}::{name}" if parent else name

    return Symbol(
        name=name,
        kind=sym_kind,
        tier=tier,
        qualname=qualname,
        signature=signature,
        doc=build_doc(member),
        source_file=_rel(location, repo_root),
        source_line=line,
        language="cuda" if location.endswith((".cu", ".cuh")) else "cpp",
        flags=flags,
        anchor_override=slugify(f"{qualname}-{line}"),
    )


def collect_native(repo_root: Path, docs_dir: Path,
                   scratch: Optional[Path] = None) -> List[Group]:
    """Parse Doxygen XML into T3-T7 page groups."""
    xml_dir = run_doxygen(repo_root, docs_dir, scratch)

    pages: Dict[tuple, Group] = {}

    def page_for(tier: str, location: str) -> Group:
        slug, title, subtitle = _page_key(tier, location)
        key = (tier, slug)
        if key not in pages:
            pages[key] = Group(
                title=title,
                slug=slug,
                tier=tier,
                subtitle=subtitle,
                source_file=_rel(location, repo_root),
            )
        return pages[key]

    for xml_file in sorted(xml_dir.glob("*.xml")):
        if xml_file.name in ("index.xml", "Doxyfile.xml"):
            continue
        try:
            root = ET.parse(xml_file).getroot()
        except ET.ParseError:
            continue

        for compound in root.findall("compounddef"):
            ckind = compound.get("kind", "")
            if ckind not in ("file", "namespace", "class", "struct"):
                continue

            cname = (compound.findtext("compoundname") or "").strip()
            loc_node = compound.find("location")
            clocation = loc_node.get("file", "") if loc_node is not None else ""

            # Classes and structs render as a parent card with their members
            # nested inside, mirroring how Python classes are presented.
            if ckind in ("class", "struct") and clocation:
                cline = int(loc_node.get("line", 0) or 0)
                parent_symbol = Symbol(
                    name=cname,
                    kind="struct" if ckind == "struct" else "class",
                    tier="T3",
                    qualname=cname,
                    doc=build_doc(compound),
                    source_file=_rel(clocation, repo_root),
                    source_line=cline,
                    language="cuda" if clocation.endswith((".cu", ".cuh")) else "cpp",
                    bases=[b.text or "" for b in compound.findall("basecompoundref")],
                    anchor_override=slugify(f"{cname}-{cline}"),
                )
                for section in compound.findall("sectiondef"):
                    if section.get("kind", "").startswith("private"):
                        continue
                    for member in section.findall("memberdef"):
                        child = _member_symbol(member, repo_root, parent=cname)
                        if child is not None:
                            child.tier = "T3"
                            parent_symbol.children.append(child)
                page_for("T3", clocation).symbols.append(parent_symbol)
                continue

            for section in compound.findall("sectiondef"):
                if section.get("kind", "").startswith("private"):
                    continue
                for member in section.findall("memberdef"):
                    symbol = _member_symbol(member, repo_root)
                    if symbol is None:
                        continue
                    member_loc = member.find("location")
                    where = member_loc.get("file", clocation) if member_loc is not None else clocation
                    page_for(symbol.tier, where).symbols.append(symbol)

    # Drop empty pages, de-duplicate, and order deterministically.
    groups = [g for g in pages.values() if g.symbols]
    for group in groups:
        seen = set()
        unique = []
        for symbol in group.symbols:
            key = (symbol.qualname, symbol.source_file, symbol.source_line)
            if key in seen:
                continue
            seen.add(key)
            unique.append(symbol)
        group.symbols = sorted(unique, key=lambda s: (s.source_line, s.name))

    groups.sort(key=lambda g: (g.tier, -len(g.all_symbols), g.slug))
    return groups
