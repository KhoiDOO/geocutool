"""Narrative pages: showcase, documentation, and about.

Content is derived from the repository's own sources -- ``README.md`` for the
feature taxonomy, benchmarks and quickstarts, and ``acknowledgement/*.md`` for
the bibliography -- so the site stays truthful to what the project actually
claims about itself. The BibTeX and link lists are parsed rather than retyped,
which keeps the References section in step with the repo.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import render
from render import code_block, esc, md_inline

# --------------------------------------------------------------------------- #
# Bibliography parsing
# --------------------------------------------------------------------------- #

_FIELD = re.compile(r"(\w+)\s*=\s*[{\"]?(.*?)[}\"]?\s*,?\s*$")
_ENTRY = re.compile(r"@(\w+)\s*\{\s*([^,]*),(.*?)\n\}", re.DOTALL)
_LINK_ITEM = re.compile(r"^-\s*\[([^\]]+)\]\s*\(?\s*(https?://[^)\s]+)\)?")


def _clean(value: str) -> str:
    value = value.strip().strip(",").strip()
    value = value.replace("{", "").replace("}", "").replace("\\", "")
    return re.sub(r"\s+", " ", value).strip()


def parse_bibtex(text: str) -> List[Dict[str, str]]:
    """Extract BibTeX entries into flat dicts."""
    entries: List[Dict[str, str]] = []
    for kind, key, body in _ENTRY.findall(text):
        fields: Dict[str, str] = {"_type": kind.lower(), "_key": key.strip()}
        for line in body.split("\n"):
            match = _FIELD.match(line.strip())
            if match:
                fields[match.group(1).lower()] = _clean(match.group(2))
        if fields.get("title"):
            entries.append(fields)
    return entries


def parse_links(text: str) -> List[Dict[str, str]]:
    out = []
    for line in text.split("\n"):
        match = _LINK_ITEM.match(line.strip())
        if match:
            out.append({"title": match.group(1).strip(), "url": match.group(2).strip()})
    return out


#: Some BibTeX entries in acknowledgement/ record only the short title. Completing
#: them here keeps the rendered bibliography correct without editing the source
#: files, which are the author's own notes.
TITLE_COMPLETIONS = {
    "Wenger2013": "Isosurfaces: Geometry, Topology, and Algorithms",
}


def _citation(entry: Dict[str, str]) -> str:
    authors = entry.get("author", "")
    if authors:
        names = [a.strip() for a in re.split(r"\s+and\s+", authors) if a.strip()]
        if len(names) > 3:
            authors = f"{names[0]} et al."
        else:
            authors = ", ".join(names)

    venue = (
        entry.get("booktitle")
        or entry.get("journal")
        or entry.get("publisher")
        or entry.get("series")
        or ""
    )
    year = entry.get("year", "")
    url = entry.get("url") or (f"https://doi.org/{entry['doi']}" if entry.get("doi") else "")

    title = esc(TITLE_COMPLETIONS.get(entry.get("_key", ""), entry["title"]))
    if url:
        title = f'<a href="{esc(url)}">{title}</a>'

    meta = " · ".join(p for p in (esc(authors), esc(venue), esc(year)) if p)
    kind = "Book" if entry.get("_type") == "book" else "Paper"
    return (
        f'<div class="card"><span class="idx">{kind}</span>'
        f"<h3>{title}</h3><p>{meta}</p></div>"
    )


# --------------------------------------------------------------------------- #
# Showcase
# --------------------------------------------------------------------------- #

_HERO_SNIPPET = """import torch
from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import dmc

grid_vertices, voxels, _ = create_voxel_grid(
    grid_min=[-1.0] * 3, grid_max=[1.0] * 3,
    res=[64, 64, 64], device="cuda",
)

sdf = (torch.norm(grid_vertices, dim=-1) - 0.6).requires_grad_(True)
verts, faces = dmc(grid_vertices, voxels, sdf, iso=0.0)

verts.sum().backward()          # gradients flow into the field
"""

_SNIPPET_1 = "docker pull kohido/conquer3d:latest\\ndocker run --rm --gpus all -it kohido/conquer3d:latest bash"
_SNIPPET_2 = "git clone https://github.com/KhoiDOO/conquer3d.git\\ncd conquer3d\\npip install -e . --no-build-isolation"

FEATURES = [
    (
        "01",
        "Surfaces from fields",
        "Turn any scalar field — an analytic SDF, a network output, distances queried from a "
        "mesh — into a watertight, 2-manifold surface. Sharp CAD creases survive when you want "
        "them to, and ambiguous cells resolve without pinch points.",
    ),
    (
        "02",
        "Gradients that flow through",
        "Extraction is part of the training step, not a preprocessing stage around it. Where an "
        "analytical backward exists, gradients propagate from the mesh back into the field and "
        "the colours defined on it.",
    ),
    (
        "03",
        "Built for the memory you have",
        "Narrow-band voxelisation never allocates the dense volume, so a $1024^3$ extraction fits "
        "on a single consumer GPU. Spatial hierarchies keep queries cache-coherent instead of "
        "chasing pointers.",
    ),
    (
        "04",
        "Everything stays on device",
        "Every operator consumes and produces PyTorch tensors in place. No host round-trip, no "
        "format conversion, no copying a mesh back to the CPU just to measure or voxelise it.",
    ),
]

BENCHMARKS = [
    ("Dual Marching Cubes", "Triangles", "1,716,386", "3,432,768", "3.44", "~1.0B faces/s", True),
    ("Dual Marching Cubes", "Pure Quads", "1,716,386", "1,716,384", "3.22", "533M quads/s", True),
    ("Dual Contouring + Normals", "Triangles", "1,716,386", "3,432,768", "4.79", "717M faces/s", True),
    ("Marching Cubes Asymptotic", "Triangles", "1,716,384", "3,432,764", "16.42", "209M faces/s", True),
]


# --------------------------------------------------------------------------- #
# Qualitative results gallery
# --------------------------------------------------------------------------- #

#: (file, title, kicker, prose, chips, wide). Every image is real library output,
#: regenerated by docs/_figures/make_figures.py.
FIGURES = [
    (
        "fig-algorithms", "Four extractors, one field", "Comparison",
        "The source mesh leads, then the same signed distance field on a 64³ grid meshed "
        "four ways. Marching Cubes and its asymptotic variant round the crease into a "
        "staircase; Dual Contouring's QEF solve snaps the vertex onto the feature; Dual "
        "Marching Cubes holds manifold topology throughout. The close crop is placed "
        "automatically at the point where MC and DC disagree most, measured with the "
        "library's own Chamfer distance.",
        ["Fandisk", "ground truth first", "64³ grid", "auto-located crop"], True,
    ),
    (
        "fig-resolution", "Detail is a resolution dial", "Scaling",
        "Dual Marching Cubes on the Armadillo across five grid resolutions, against the "
        "source mesh. Face count grows from 12,076 to 3,317,336 while the extraction code "
        "and the field are unchanged — the armour plating only resolves once the grid is "
        "fine enough to carry it, and by 1024³ the extraction is visually indistinguishable "
        "from the 99,976-face original.",
        ["Armadillo", "ground truth first", "64³ → 1024³", "12k → 3.3M faces"], True,
    ),
    (
        "fig-sign-modes", "Six ways to decide inside", "Robustness",
        "One axial slice through the Bimba bust, signed by each of the six available modes "
        "and contoured at zero. Ray parity misclassifies a scatter of points in the upper "
        "left where rays graze the surface; pseudonormals and winding numbers are clean; the "
        "flood-fill modes quantise the boundary to their occupancy grid. Most libraries give "
        "you exactly one of these.",
        ["Bimba", "sign_mode 0–5", "320² slice"], True,
    ),
    (
        "fig-curvature", "Differential geometry, on device", "Analysis",
        "Per-vertex curvature over 134,345 vertices of the Igea bust, computed with the "
        "cotangent Laplace–Beltrami operator and mixed Voronoi areas, shown beside the "
        "unshaded source. Mean curvature and the first principal curvature pick out the "
        "eyelids, lips and braid detail that Gaussian curvature alone flattens.",
        ["Igea", "ground truth first", "134,345 vertices", "cotangent operator"], False,
    ),
    (
        "fig-normal-modes", "The field Dual Contouring solves against", "Normals",
        "Dual Contouring is only as sharp as the normals it is given. Against the source "
        "mesh: exact CAD face normals produce a clean surface; smooth vertex normals soften "
        "it; deriving normals from the SDF gradient introduces the visible speckle that a "
        "QEF then faithfully reproduces.",
        ["RockerArm", "ground truth first", "normal_mode 0/1/2", "224³ grid"], False,
    ),
    (
        "fig-sdf-slices", "The field behind the surface", "Fields",
        "Three axial slices of the signed distance field around Spot, with the zero level set "
        "drawn in green. This is the quantity every extractor in the library consumes — the "
        "mesh is just where this field crosses zero.",
        ["Spot", "3 slices", "zero level set"], False,
    ),
    (
        "fig-memory", "Why the grid stays sparse", "Memory",
        "Measured grid storage against resolution. Narrow-band voxelisation visits only the "
        "cells the surface actually crosses, so cost tracks surface area rather than volume: "
        "at 512³ that is 58.6 MB instead of 5.6 GB, and it is the reason high-resolution "
        "extraction fits on a single consumer GPU at all.",
        ["Fandisk", "32³ → 512³", "96× smaller at 512³"], False,
    ),
]


def _figure_card(entry) -> str:
    name, title, kicker, prose, chips, _ = entry
    chip_html = "".join(f"<span>{esc(c)}</span>" for c in chips)
    return (
        f'<figure class="fig">'
        f'<div class="fig-media">'
        f'<img src="assets/img/{name}.webp" alt="{esc(title)}" loading="lazy" decoding="async">'
        f"</div>"
        f'<figcaption class="fig-body">'
        f'<span class="fig-kicker">{esc(kicker)}</span>'
        f"<h3>{esc(title)}</h3><p>{md_inline(prose)}</p>"
        f'<div class="fig-meta">{chip_html}</div>'
        f"</figcaption></figure>"
    )


def gallery() -> str:
    """The qualitative results section of the showcase."""
    wide = "".join(_figure_card(f) for f in FIGURES if f[5])
    narrow = "".join(_figure_card(f) for f in FIGURES if not f[5])
    return f"""
<section class="section wrap">
  <div class="section-head">
    <span class="kicker">Qualitative results</span>
    <h2>What the operators actually produce</h2>
    <p>Every figure below is real output, generated from the bundled benchmark assets by
    <code>docs/_figures/make_figures.py</code> and regenerated whenever the library changes.
    Click any figure to enlarge.</p>
  </div>
  <div class="fig-grid" style="grid-template-columns:minmax(0,1fr)">{wide}</div>
  <div class="fig-grid" style="margin-top:18px">{narrow}</div>
</section>
"""


def showcase(version: str, stats: Dict[str, int]) -> str:
    """The landing page: what the library is for, and how to start.

    Deliberately light on internals -- throughput tables and the tier breakdown
    live on the Benchmarks page. This is the slot qualitative results and
    rendered examples will occupy.
    """
    cards = "".join(
        f'<div class="card"><span class="idx">{idx}</span><h3>{esc(title)}</h3>'
        f"<p>{md_inline(body)}</p></div>"
        for idx, title, body in FEATURES
    )

    return f"""
<section class="section wrap">
  <div class="section-head">
    <span class="kicker">What it is</span>
    <h2>A GPU-native geometry toolbox, built for gradients</h2>
    <p>Conquer3D implements computational geometry directly in CUDA and exposes it through
    PyTorch tensors — so meshing, querying and voxelising happen where your data already
    lives.</p>
  </div>
  <div class="grid grid-2">{cards}</div>
</section>

{gallery()}

<section class="section wrap" style="border-bottom:none">
  <div class="section-head">
    <span class="kicker">Get started</span>
    <h2>Install and extract a surface</h2>
  </div>
  <div class="grid grid-2">
    <div>{code_block("pip install -U conquer3d", "bash")}
      <p style="color:var(--fg-mid);font-size:.9rem">Prebuilt CUDA wheels are published per
      release for Python 3.10–3.14 against PyTorch 2.8 and 2.11. Building from source needs
      CUDA $\\ge$ 12.0.</p>
      <p><a class="btn" href="documentation.html">Read the guide →</a>
         <a class="btn" href="benchmarks.html">See the numbers →</a></p>
    </div>
    <div>{code_block(_HERO_SNIPPET, "python")}</div>
  </div>
</section>
"""


def benchmarks(version: str, stats: Dict[str, int]) -> str:
    """Measured extraction throughput."""
    rows = "".join(
        f"<tr><td><strong>{esc(algo)}</strong></td><td>{esc(kind)}</td>"
        f'<td class="num">{esc(verts)}</td><td class="num">{esc(faces)}</td>'
        f'<td class="num hl">{esc(ms)} ms</td><td class="num">{esc(rate)}</td>'
        f'<td>{"Yes" if tight else "—"}</td></tr>'
        for algo, kind, verts, faces, ms, rate, tight in BENCHMARKS
    )

    return f"""
<div class="wrap" style="padding-top:52px">
<div class="page-head">
  <div class="crumbs"><a href="index.html">Home</a> › Benchmarks</div>
  <h1 style="font-family:var(--sans)">Benchmarks</h1>
  <p class="sub">Measured extraction throughput on real geometry.</p>
</div>
</div>

<section class="section wrap" style="padding-top:8px;border-bottom:none">
  <div class="section-head">
    <span class="kicker">Throughput</span>
    <h2>Isosurface extraction</h2>
    <p>Measured on an NVIDIA GeForce RTX 4090 over the Fandisk CAD model at $1024^3$
    resolution — 6.86M sparse vertices across 5.15M active voxels.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Algorithm</th><th>Output</th><th>Vertices</th><th>Faces</th>
    <th>Latency</th><th>Throughput</th><th>Watertight</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  <div class="adm adm-note" style="margin-top:20px"><div class="adm-title">Reading these numbers</div>
  <p>Latency covers extraction only, on a grid that has already been built and populated.
  Dual Marching Cubes leads because it emits multiple dual vertices per cell without a
  per-cell solve; Dual Contouring pays for a $3\\times3$ Jacobi SVD per voxel and buys sharp
  creases with it.</p></div>
</section>

"""


def hero(version: str, stats: Dict[str, int]) -> str:
    return f"""
<section class="hero">
  <canvas id="hero-canvas"></canvas>
  <div class="hero-fade"></div>
  <div class="hero-inner">
    <div class="hero-copy">
      <span class="eyebrow"><span class="pulse"></span> v{esc(version)} · CUDA 12 · PyTorch 2.x</span>
      <h1>Differentiable geometry<br><span class="grad">at kernel speed</span></h1>
      <p class="lede">A GPU-native toolbox for isosurface extraction, spatial acceleration, and
      volumetric conversion — written in CUDA, exposed as PyTorch tensors, and documented down
      to the individual kernel.</p>
      <div class="btn-row">
        <a class="btn btn-primary" href="documentation.html">Get started</a>
        <a class="btn" href="api/index.html">API Reference</a>
        <a class="btn" href="{render.REPO}">GitHub</a>
      </div>
    </div>
    <div class="hero-card">
      <div class="hero-card-bar"><span class="dot"></span><span class="dot"></span><span class="dot"></span>
        &nbsp;dual_marching_cubes.py</div>
      {code_block(_HERO_SNIPPET, "python")}
    </div>
  </div>
</section>
"""


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #

_QS_DC = """import torch
from conquer3d.data.assets import Fandisk
from conquer3d.data_structure import TriangleMesh, create_voxel_grid_from_tmesh
from conquer3d.ops import dc

v, f, _ = Fandisk().get()
tmesh = TriangleMesh(v.cuda(), f.cuda().int())

# Sparse narrow-band grid plus exact CAD face normals (normal_mode=0)
grid_vertices, voxels, _, grid_normals = create_voxel_grid_from_tmesh(
    grid_min=[-1.0] * 3, grid_max=[1.0] * 3, res=[256] * 3,
    tmesh=tmesh, pad=1, return_normals=True, normal_mode=0,
)

# Signed distances via GPU flood fill
tmesh.build_flood_fill_data([-1.0] * 3, [1.0] * 3, [256] * 3)
_, _, _, sdfs = tmesh.query_points(grid_vertices, return_sdf=True, sign_mode=3)

# Jacobi-SVD QEF solve preserves sharp creases
verts, faces = dc(grid_vertices, voxels, sdfs, grid_normals=grid_normals, iso=0.0)
"""

_QS_QUERY = """query_pts = torch.randn((100_000, 3), device="cuda")

query_ids, closest_tri_ids, projected_pts, sdfs = tmesh.query_points(
    query_pts, return_sdf=True, return_prj_pts=True, sign_mode=0,
)
"""



def documentation(version: str, stats: Dict[str, int], tiers: List[tuple]) -> str:
    tier_rows = "".join(
        f'<tr><td><span class="badge badge-{tier.lower()}">{esc(tier)}</span></td>'
        f"<td><strong>{esc(name)}</strong></td>"
        f'<td class="num">{total}</td>'
        f'<td class="num">{documented}</td>'
        f'<td style="min-width:150px"><div class="cov"><span class="cov-bar">'
        f'<span class="cov-fill" style="width:{pct}%"></span></span>'
        f"<span>{pct}%</span></div></td></tr>"
        for tier, name, documented, total, pct in tiers
    )

    return f"""
<div class="wrap" style="padding-top:52px">
<div class="page-head">
  <div class="crumbs"><a href="index.html">Home</a> › Documentation</div>
  <h1 style="font-family:var(--sans)">Documentation</h1>
  <p class="sub">Installing Conquer3D, the concepts it is built on, and worked examples.</p>
</div>
</div>

<section class="section wrap" style="padding-top:8px">
  <div class="section-head"><span class="kicker">Installation</span><h2>Three ways in</h2></div>
  <div class="grid grid-3">
    <div class="card"><span class="idx">Recommended</span><h3>PyPI</h3>
      {code_block("pip install -U conquer3d", "bash")}
      <p>Prebuilt CUDA wheels are attached to each GitHub release for Python 3.10–3.14
      against PyTorch 2.8 and 2.11 (CUDA 12.8).</p></div>
    <div class="card"><span class="idx">Zero setup</span><h3>Docker</h3>
      {code_block(_SNIPPET_1, "bash")}
      <p>Ships a complete CUDA and PyTorch environment; nothing to compile.</p></div>
    <div class="card"><span class="idx">Development</span><h3>From source</h3>
      {code_block(_SNIPPET_2, "bash")}
      <p>Needs CUDA $\\ge$ 12.0, a matching PyTorch, and <code>pybind11-stubgen</code>
      (the build regenerates <code>_C.pyi</code>).</p></div>
  </div>

  <div class="adm adm-note" style="margin-top:26px"><div class="adm-title">Requires a GPU</div>
  <p>Importing <code>conquer3d</code> loads the compiled <code>_C</code> extension at module
  scope, so a CUDA-capable device and a matching PyTorch build are required. The one exception
  is <code>conquer3d.ops.dpsr</code>, which is pure PyTorch and will also run on CPU.</p></div>
</section>

<section class="section wrap">
  <div class="section-head"><span class="kicker">Concepts</span>
    <h2>How the pieces fit</h2>
    <p>Most pipelines follow the same three steps: build a grid, evaluate a field on it,
    then extract a surface from the field.</p></div>
  <div class="grid grid-3">
    <div class="card"><span class="idx">Step 1</span><h3>Build a grid</h3>
      <p>A grid is a vertex array <code>(V, 3)</code> plus a voxel array <code>(N, 8)</code> of
      corner indices. <code>create_voxel_grid</code> makes a dense one;
      <code>create_voxel_grid_from_tmesh</code> makes a narrow band hugging a surface, which is
      what makes $1024^3$ tractable.</p></div>
    <div class="card"><span class="idx">Step 2</span><h3>Evaluate a field</h3>
      <p>Any <code>(V,)</code> scalar per grid vertex works — an analytic SDF, a network output,
      or distances queried from a mesh via <code>TriangleMesh.query_points</code>. The
      <code>sign_mode</code> argument selects how inside/outside is decided (ray parity, winding
      number, flood fill).</p></div>
    <div class="card"><span class="idx">Step 3</span><h3>Extract a surface</h3>
      <p>Pick the extractor that matches the geometry: <code>dmc</code> for throughput and
      manifold output, <code>dc</code> for sharp CAD creases, <code>mca</code> when face
      ambiguities matter, <code>marching_tetrahedra</code> for unstructured domains.</p></div>
  </div>
</section>

<section class="section wrap">
  <div class="section-head">
    <span class="kicker">Architecture</span>
    <h2>Seven tiers, {stats.get('total', 0)} documented symbols</h2>
    <p>The library is a stack, and the reference mirrors it exactly. It does not stop at the
    Python surface: it descends through the pybind11 bindings and host dispatchers into the
    CUDA kernels, the device helpers they call, and the inline math those are built from.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Tier</th><th>Layer</th><th>Symbols</th><th>Documented</th><th>Coverage</th></tr></thead>
    <tbody>{tier_rows}</tbody>
  </table></div>
  <p style="margin-top:18px"><a class="btn" href="api/index.html">Browse the API Reference →</a></p>
</section>

<section class="section wrap">
  <div class="section-head"><span class="kicker">Examples</span><h2>Worked pipelines</h2></div>

  <h3 style="margin-bottom:12px">Sharp CAD extraction with Dual Contouring</h3>
  <p class="prose" style="margin-bottom:14px">Dual Contouring places one vertex per voxel by
  minimising a quadratic error function over the surface normals, which is what preserves
  mechanical edges that Marching Cubes rounds off.</p>
  {code_block(_QS_DC, "python")}

  <h3 style="margin:34px 0 12px">Spatial queries against a mesh</h3>
  {code_block(_QS_QUERY, "python")}


  <div class="adm adm-warn" style="margin-top:26px"><div class="adm-title">Differentiability</div>
  <p>Gradient support is per-operator. <code>diff_marching_cubes</code>,
  <code>marching_tetrahedra</code>, <code>diff_marching_tetrahedra_grid</code> and
  <code>dpsr</code> propagate gradients today. The autograd wrappers for <code>dmc</code>,
  <code>dc</code> and <code>mca</code> are present in the source but currently commented out,
  so those three are forward-only — check the operator's API page before relying on
  <code>.backward()</code>.</p></div>
</section>

<section class="section wrap" style="border-bottom:none">
  <div class="section-head"><span class="kicker">Next</span><h2>Where to go</h2></div>
  <div class="grid grid-3">
    <div class="card"><h3><a href="api/index.html">API Reference →</a></h3>
      <p>Every symbol, from the Python surface down to individual CUDA kernels.</p></div>
    <div class="card"><h3><a href="about.html">About &amp; References →</a></h3>
      <p>The papers, books, and implementations this library is built on.</p></div>
    <div class="card"><h3><a href="{render.REPO}/issues">Issues →</a></h3>
      <p>Bug reports and feature requests on GitHub.</p></div>
  </div>
</section>
"""


# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #


def about(version: str, ack_dir: Path, stats: Dict[str, int]) -> str:
    def read(name: str) -> str:
        path = ack_dir / name
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    papers = parse_bibtex(read("REFERENCE.md"))
    books = parse_bibtex(read("BOOK.md"))
    repos = parse_links(read("REPOSITORY.md"))
    posts = parse_links(read("BLOG_POST.md"))

    def link_cards(items: List[Dict[str, str]], kind: str) -> str:
        return "".join(
            f'<div class="card"><span class="idx">{kind}</span>'
            f'<h3><a href="{esc(i["url"])}">{esc(i["title"])}</a></h3>'
            f'<p style="font-size:.8rem;word-break:break-all">{esc(i["url"])}</p></div>'
            for i in items
        )

    papers_html = "".join(_citation(p) for p in papers) or "<p>No entries found.</p>"
    books_html = "".join(_citation(b) for b in books) or "<p>No entries found.</p>"

    return f"""
<div class="wrap" style="padding-top:52px">
<div class="page-head">
  <div class="crumbs"><a href="index.html">Home</a> › About</div>
  <h1 style="font-family:var(--sans)">About Conquer3D</h1>
  <p class="sub">Why it exists, how it is built, and what it stands on.</p>
</div>
</div>

<section class="section wrap" style="padding-top:8px">
  <div class="section-head"><span class="kicker">Motivation</span>
  <h2>A PhD side project, built to learn by building</h2></div>
  <div class="prose" style="font-size:1rem;max-width:70ch">
    <p>Conquer3D is written during my PhD in Vision and Graphics. It exists to learn 3D
    differentiable geometry and graphics from the ground up — by implementing the
    algorithms rather than calling them.</p>
    <p>Almost every component runs on the GPU: isosurface extraction, spatial acceleration
    structures, volumetric conversion, and the geometric primitives underneath them are
    written as CUDA kernels and exposed as PyTorch tensors.</p>
  </div>
</section>

<section class="section wrap">
  <div class="section-head"><span class="kicker">Bibliography</span><h2>Research papers</h2>
  <p>The algorithms implemented here come from published work. These are the primary sources.</p></div>
  <div class="grid grid-2">{papers_html}</div>
</section>

<section class="section wrap">
  <div class="section-head"><span class="kicker">Bibliography</span><h2>Books</h2></div>
  <div class="grid grid-2">{books_html}</div>
</section>

<section class="section wrap">
  <div class="section-head"><span class="kicker">Bibliography</span><h2>Blog posts &amp; repositories</h2>
  <p>Practical GPU traversal and construction writing, and the open-source implementations this
  project learned from.</p></div>
  <div class="grid grid-3">{link_cards(posts, "Blog")}{link_cards(repos, "Repository")}</div>
</section>

<section class="section wrap" style="border-bottom:none">
  <div class="section-head"><span class="kicker">Citation</span><h2>Using Conquer3D in your work</h2></div>
  {code_block('''@software{conquer3d,
  title  = {Conquer3D: GPU-Accelerated Differentiable Geometry and Spatial Computing},
  author = {Do, Hoang Khoi},
  year   = {2025},
  url    = {https://github.com/KhoiDOO/conquer3d},
  version = {%s}
}''' % version, "bash")}
  <p class="prose">Released under the MIT License.</p>
</section>
"""
