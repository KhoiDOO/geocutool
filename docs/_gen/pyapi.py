"""Static extraction of the Python-facing API via :mod:`ast`.

Two sources feed this module, and neither is imported:

* ``conquer3d/**/*.py`` -- tier T1, the public Python surface.
* ``conquer3d/_C.pyi``  -- tier T2, the pybind11 surface. ``pybind11-stubgen``
  copies each ``R"pbdoc(...)pbdoc"`` block into the stub verbatim, so the native
  API arrives fully documented without a CUDA build.

Importing is deliberately avoided: ``conquer3d/__init__.py`` pulls in the
compiled ``_C`` extension at module scope, so ``import conquer3d`` cannot succeed
without a torch-matched CUDA build.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, List, Optional, Set

import bindmap
import docparse
from model import Doc, Group, Symbol

# Subpackages that become their own T1 page, in nav order.
PY_PAGES = [
    ("conquer3d", "Package Root", "Top-level exports and version metadata."),
    ("ops", "Operators", "Isosurface extraction, distances, and volume integrals."),
    ("data_structure", "Data Structures", "BVHs, KD-trees, meshes, and voxel grids."),
    ("primitive", "Primitives", "Rays, triangles, and Gaussian splatting primitives."),
    ("conversion", "Conversion", "Mesh, voxel, and sparse representation conversion."),
    ("creation", "Creation", "Procedural mesh generators."),
    ("io", "File I/O", "OBJ, PLY, and OFF readers and writers."),
    ("data", "Datasets", "Benchmark assets, datasets, transforms, and collation."),
]

# Names leaked into conquer3d.data.assets by star-imports without __all__.
_LEAKED = {
    "os", "sys", "torch", "np", "numpy", "urllib", "zipfile", "tarfile", "shutil",
    "gdown", "trimesh", "tqdm", "Tuple", "Optional", "List", "Dict", "Union", "Any",
    "annotations", "read_obj", "read_off", "read_ply", "Path",
}


# --------------------------------------------------------------------------- #
# Signature rendering
# --------------------------------------------------------------------------- #


def _annotation(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _default(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def format_signature(node, *, drop_self: bool = False) -> str:
    """Render a def/class argument list back into readable source form."""
    args = node.args
    parts: List[str] = []

    positional = list(args.posonlyargs) + list(args.args)
    defaults: List[Optional[ast.AST]] = [None] * (
        len(positional) - len(args.defaults)
    ) + list(args.defaults)

    if drop_self and positional and positional[0].arg in ("self", "cls"):
        positional = positional[1:]
        defaults = defaults[1:]

    for index, arg in enumerate(positional):
        text = arg.arg
        ann = _annotation(arg.annotation)
        if ann:
            text += f": {ann}"
        if defaults[index] is not None:
            text += f" = {_default(defaults[index])}" if ann else f"={_default(defaults[index])}"
        parts.append(text)
        if args.posonlyargs and index == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        ann = _annotation(args.vararg.annotation)
        parts.append(f"*{args.vararg.arg}" + (f": {ann}" if ann else ""))
    elif args.kwonlyargs:
        parts.append("*")

    for arg, kwdefault in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        ann = _annotation(arg.annotation)
        if ann:
            text += f": {ann}"
        if kwdefault is not None:
            text += f" = {_default(kwdefault)}" if ann else f"={_default(kwdefault)}"
        parts.append(text)

    if args.kwarg:
        ann = _annotation(args.kwarg.annotation)
        parts.append(f"**{args.kwarg.arg}" + (f": {ann}" if ann else ""))

    returns = _annotation(getattr(node, "returns", None))
    suffix = f" -> {returns}" if returns else ""
    return f"({', '.join(parts)}){suffix}"


# --------------------------------------------------------------------------- #
# Module reading
# --------------------------------------------------------------------------- #


def _decorator_names(node) -> List[str]:
    out = []
    for dec in getattr(node, "decorator_list", []):
        try:
            out.append(ast.unparse(dec))
        except Exception:
            pass
    return out


def _base_names(node: ast.ClassDef) -> List[str]:
    out = []
    for base in node.bases:
        try:
            out.append(ast.unparse(base))
        except Exception:
            pass
    return out


def _read_all(tree: ast.Module) -> Optional[Set[str]]:
    """Return the module's ``__all__`` contents if it declares one."""
    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                value = node.value
                if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
                    names = set()
                    for element in value.elts:
                        if isinstance(element, ast.Constant) and isinstance(element.value, str):
                            names.add(element.value)
                    return names
                # `__all__` built by concatenating category lists (data_structure).
                if isinstance(value, ast.BinOp):
                    return None
    return None


def _method_kind(node, decorators: List[str]) -> str:
    if any(d.endswith("property") or d.endswith("cached_property") for d in decorators):
        return "property"
    return "method"


def _class_symbol(node: ast.ClassDef, module: str, rel_path: str, tier: str) -> Symbol:
    qualname = f"{module}.{node.name}" if module else node.name
    bases = _base_names(node)

    flags: List[str] = []
    joined = " ".join(bases)
    if "Function" in joined:
        flags.append("autograd.Function")
    if "Module" in joined:
        flags.append("nn.Module")
    if "Dataset" in joined:
        flags.append("Dataset")

    symbol = Symbol(
        name=node.name,
        kind="class",
        tier=tier,
        qualname=qualname,
        signature="",
        doc=docparse.parse(ast.get_docstring(node)),
        source_file=rel_path,
        source_line=node.lineno,
        language="python",
        bases=bases,
        flags=flags,
    )

    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name.startswith("_") and child.name != "__init__":
                continue
            decorators = _decorator_names(child)
            kind = _method_kind(child, decorators)
            symbol.children.append(
                Symbol(
                    name=child.name,
                    kind=kind,
                    tier=tier,
                    qualname=f"{qualname}.{child.name}",
                    signature=(
                        "" if kind == "property" else format_signature(child, drop_self=True)
                    ),
                    doc=docparse.parse(ast.get_docstring(child)),
                    source_file=rel_path,
                    source_line=child.lineno,
                    language="python",
                    flags=[d for d in decorators if "property" not in d],
                )
            )
    return symbol


def read_module(path: Path, module: str, rel_path: str, tier: str) -> tuple:
    """Parse one file into (module Doc, [Symbol]).

    Only module-level definitions are collected; nested helpers stay private.
    """
    source = path.read_text(encoding="utf-8", errors="replace")
    tree = ast.parse(source, filename=str(path))
    module_doc = docparse.parse(ast.get_docstring(tree))
    declared = _read_all(tree)

    symbols: List[Symbol] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            if declared is not None and node.name not in declared and module.endswith("_C"):
                continue
            symbols.append(_class_symbol(node, module, rel_path, tier))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            if node.name in _LEAKED:
                continue
            symbols.append(
                Symbol(
                    name=node.name,
                    kind="function",
                    tier=tier,
                    qualname=f"{module}.{node.name}" if module else node.name,
                    signature=format_signature(node),
                    doc=docparse.parse(ast.get_docstring(node)),
                    source_file=rel_path,
                    source_line=node.lineno,
                    language="python",
                    flags=_decorator_names(node),
                )
            )

    return module_doc, symbols


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


def _iter_package_files(pkg_root: Path, subpackage: str) -> List[Path]:
    if subpackage == "conquer3d":
        return [pkg_root / "__init__.py"]
    base = pkg_root / subpackage
    if not base.is_dir():
        return []
    return sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)


def collect_python(repo_root: Path) -> List[Group]:
    """Build the T1 page set from ``conquer3d/**/*.py``."""
    pkg_root = repo_root / "conquer3d"
    groups: List[Group] = []

    for subpackage, title, subtitle in PY_PAGES:
        files = _iter_package_files(pkg_root, subpackage)
        if not files:
            continue

        group = Group(
            title=title if subpackage == "conquer3d" else f"conquer3d.{subpackage}",
            slug=subpackage,
            tier="T1",
            subtitle=subtitle,
            source_file=f"conquer3d/{'' if subpackage == 'conquer3d' else subpackage}",
        )

        for path in files:
            rel_path = str(path.relative_to(repo_root))
            parts = path.relative_to(pkg_root).with_suffix("").parts
            parts = parts[:-1] if parts and parts[-1] == "__init__" else parts
            module = ".".join(("conquer3d",) + parts)

            try:
                module_doc, symbols = read_module(path, module, rel_path, "T1")
            except SyntaxError as exc:
                print(f"  ! skipped {rel_path}: {exc}")
                continue

            if path.name == "__init__.py" and not group.doc.summary:
                group.doc = module_doc

            if symbols:
                header = Symbol(
                    name=module,
                    kind="module",
                    tier="T1",
                    qualname=module,
                    doc=module_doc,
                    source_file=rel_path,
                    source_line=1,
                    language="python",
                )
                header.children = symbols
                group.symbols.append(header)

        # The package root defines no symbols of its own -- it only re-exports --
        # but its module docstring is still worth a page.
        if group.symbols or not group.doc.is_empty:
            groups.append(group)

    return groups


# Per-class subtitles for the native types that warrant a page of their own.
_C_CLASS_BLURB = {
    "TriangleMesh": "Half-edge connectivity, curvature, and flood-fill queries.",
    "MeshBVH": "Ray, point, and voxel queries against triangle meshes.",
    "BVH": "Karras linear bounding volume hierarchy.",
    "GSBVH": "Hierarchy specialised for 3D Gaussian splats.",
    "PGSBVH": "Hierarchy for periodic Gaussian splats.",
    "KDTree": "Parallel nearest-neighbour search.",
    "Triangle": "Intersection, projection, and quality predicates.",
    "Ray": "Parametric ray with slab-test AABB intersection.",
}


def _classes_from_bindings(bindings: Dict[str, bindmap.Binding]) -> List[Symbol]:
    """Reconstruct the T2 class symbols from the binding sources alone.

    ``_C.pyi`` is produced by ``pybind11-stubgen`` after a build and is matched
    by ``*.pyi`` in .gitignore, so it exists in a working tree and in no tagged
    checkout. The bindings carry the same classes, their members, their pbdoc
    blocks and their ``py::arg`` lists, so a stub-free tree loses the stub's
    type annotations but nothing structural.
    """
    out: List[Symbol] = []
    for name, binding in sorted(bindings.items()):
        if not binding.is_class:
            continue
        klass = Symbol(
            name=name,
            kind="class",
            tier="T2",
            qualname=f"conquer3d._C.{name}",
            signature="",
            doc=docparse.parse(binding.docstring),
            source_file=binding.source_file,
            source_line=binding.source_line,
            language="python",
        )
        prefix = f"{name}."
        members = [b for key, b in bindings.items() if key.startswith(prefix)]
        for member in sorted(members, key=lambda b: b.source_line):
            params = ", ".join(
                arg if not default else f"{arg}={default}" for arg, default in member.args
            )
            klass.children.append(
                Symbol(
                    name=member.name,
                    kind="method",
                    tier="T2",
                    qualname=f"conquer3d._C.{name}.{member.name}",
                    signature=f"({params})",
                    doc=docparse.parse(member.docstring),
                    source_file=member.source_file,
                    source_line=member.source_line,
                    language="python",
                )
            )
        out.append(klass)
    return out


def collect_bindings(repo_root: Path) -> List[Group]:
    """Build the T2 page set.

    Signatures and docstrings come from ``_C.pyi``, but the *structure* comes
    from ``csrc/binds/`` -- the stub is a flat namespace only because
    ``pybind.cpp`` passes every ``bind_*`` call the same root module, and
    presenting it that way would misrepresent the source. Each symbol is filed
    under the category directory its binding lives in, and points at that
    translation unit rather than at the generated stub.
    """
    bindings = bindmap.scan(repo_root)
    if not bindings:
        print("  ! csrc/binds not readable -- T2 will be empty")
        return []

    stub = repo_root / "conquer3d" / "_C.pyi"
    if stub.exists():
        rel_path = str(stub.relative_to(repo_root))
        module_doc, symbols = read_module(stub, "conquer3d._C", rel_path, "T2")
    else:
        # Expected for any tagged checkout: the stub is generated and gitignored.
        # The bindings are the authoritative source in either case -- the stub
        # only ever contributed type annotations on top of them.
        print("  . conquer3d/_C.pyi absent -- T2 built from csrc/binds alone")
        rel_path = "conquer3d/csrc/binds"
        module_doc, symbols = Doc(), _classes_from_bindings(bindings)

    class_groups: Dict[str, List[Group]] = {}
    # category -> binding file -> [Symbol]
    buckets: Dict[str, Dict[str, List[Symbol]]] = {}

    for symbol in symbols:
        binding = bindings.get(symbol.name)
        category = binding.category if binding else "ops"

        if binding:
            # Point at the binding source; that is where the docstring is
            # authored and where a reader would go to change it.
            symbol.source_file = binding.source_file
            symbol.source_line = binding.source_line
            # The stub can lose a pbdoc that the bindings do carry, so fall back
            # to the source for functions just as for class members.
            if symbol.doc.is_empty and binding.docstring:
                symbol.doc = docparse.parse(binding.docstring)

        if symbol.kind == "class":
            if symbol.doc.is_empty and binding is not None and binding.docstring:
                symbol.doc = docparse.parse(binding.docstring)
            # The stub leaves Triangle's and Ray's methods as bare `...`; the
            # docstrings that do exist live in the bindings, so prefer those.
            bound_members = any(
                key.startswith(f"{symbol.name}.") for key in bindings
            )
            for child in symbol.children:
                member = bindings.get(f"{symbol.name}.{child.name}")
                if member is None:
                    # Present in the generated stub but bound nowhere in the
                    # current sources -- the stub predates a binding removal.
                    if bound_members:
                        child.flags.append("stale stub entry")
                    continue
                child.source_file = member.source_file
                child.source_line = member.source_line
                if child.doc.is_empty and member.docstring:
                    child.doc = docparse.parse(member.docstring)

            group = Group(
                title=f"_C.{symbol.name}",
                slug=symbol.name.lower(),
                tier="T2",
                subtitle=_C_CLASS_BLURB.get(symbol.name, f"Native {symbol.name} bindings."),
                source_file=symbol.source_file or rel_path,
                symbols=[symbol],
                doc=symbol.doc,
            )
            class_groups.setdefault(category, []).append(group)
        else:
            key = binding.source_file if binding else rel_path
            buckets.setdefault(category, {}).setdefault(key, []).append(symbol)

    # Anything bound in C++ but absent from the stub is still real API -- the
    # stub is regenerated at build time and lags the sources. Recover those
    # directly from their pbdoc block and py::arg list.
    stub_names = {s.name for s in symbols}
    for name, binding in sorted(bindings.items()):
        # "Class.method" keys are class members, already rendered under their
        # class -- only module-level bindings belong here.
        if binding.is_class or "." in name or name in stub_names:
            continue
        params = ", ".join(
            arg if not default else f"{arg}={default}" for arg, default in binding.args
        )
        recovered = Symbol(
            name=name,
            kind="function",
            tier="T2",
            qualname=f"conquer3d._C.{name}",
            signature=f"({params})",
            doc=docparse.parse(binding.docstring),
            source_file=binding.source_file,
            source_line=binding.source_line,
            language="python",
            flags=["missing from _C.pyi"],
        )
        buckets.setdefault(binding.category, {}).setdefault(
            binding.source_file, []
        ).append(recovered)

    groups: List[Group] = []

    for category in bindmap.CATEGORY_ORDER:
        files = buckets.get(category)
        if not files:
            continue
        slug, title, subtitle = bindmap.CATEGORIES[category]

        band_symbols: List[Symbol] = []
        for source_file in sorted(files):
            members = sorted(files[source_file], key=lambda s: s.source_line)
            header = Symbol(
                name=source_file.replace("conquer3d/csrc/binds/", ""),
                kind="module",
                tier="T2",
                qualname=source_file,
                source_file=source_file,
                source_line=0,
                language="cpp",
            )
            header.children = members
            band_symbols.append(header)

        groups.append(
            Group(
                title=title,
                slug=slug,
                tier="T2",
                subtitle=subtitle,
                source_file=f"conquer3d/csrc/binds/{category}",
                symbols=band_symbols,
                doc=module_doc if category == "ops" else Doc(),
            )
        )

    # Class pages follow their category, largest first.
    for category in bindmap.CATEGORY_ORDER:
        for group in sorted(class_groups.get(category, []), key=lambda g: -len(g.all_symbols)):
            groups.append(group)

    # Cross-link each category page to the classes bound in the same directory.
    for category in bindmap.CATEGORY_ORDER:
        slug = bindmap.CATEGORIES[category][0]
        page = next((g for g in groups if g.tier == "T2" and g.slug == slug), None)
        if page is None:
            continue
        for klass in class_groups.get(category, []):
            page.related.append(
                (klass.title, f"bindings-{klass.slug}.html", len(klass.all_symbols))
            )

    return groups
