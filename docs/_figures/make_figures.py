#!/usr/bin/env python
"""Generate the qualitative result figures for the Conquer3D showcase.

Run with the project's conda environment::

    ~/anaconda3/envs/geocutool/bin/python docs/_figures/make_figures.py

Every figure is produced from the bundled benchmark assets and real library
output -- nothing here is mocked or hand-drawn -- so the gallery can be
regenerated after any change rather than going stale. Figures are written to
``docs/assets/img/`` as transparent PNGs that suit both site themes.

Each figure is independent and failures are reported rather than fatal, so a
missing asset or an unavailable sign mode cannot take the whole run down.
"""

from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compose  # noqa: E402
import render as R  # noqa: E402

OUT = HERE.parent / "assets" / "img"
DEV = "cuda"

BOUNDS_MIN = [-1.0, -1.0, -1.0]
BOUNDS_MAX = [1.0, 1.0, 1.0]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def load_mesh(asset_cls):
    """Fetch a benchmark asset and wrap it in a CUDA TriangleMesh."""
    from conquer3d.data_structure import TriangleMesh

    v, f, _ = asset_cls().get()
    v = v.cuda().float()
    # Fit the asset into the extraction bounds with a margin, so the surface
    # never touches the domain boundary (which would let the flood fill leak).
    lo, hi = v.min(0).values, v.max(0).values
    v = (v - 0.5 * (lo + hi)) / (hi - lo).max() * 1.6
    return TriangleMesh(v.contiguous(), f.cuda().int().contiguous())


def build_grid(tmesh, res: int, normal_mode: int = 0, pad: int = 1):
    """Narrow-band sparse grid plus signed distances for an asset."""
    from conquer3d.data_structure import create_voxel_grid_from_tmesh

    out = create_voxel_grid_from_tmesh(
        grid_min=BOUNDS_MIN, grid_max=BOUNDS_MAX, res=[res] * 3,
        tmesh=tmesh, pad=pad, return_normals=True, normal_mode=normal_mode,
    )
    grid_vertices, voxels, normals = out[0], out[1], out[-1]

    # The single-level fill allocates a dense grid (1536³ int32 is 13.5 GiB and
    # will not fit). The coarse-fine structure stays under ~10 MB at 1024³, so
    # anything large is signed with that instead.
    if res >= 1024:
        tmesh.build_flood_fill_cf_data(BOUNDS_MIN, BOUNDS_MAX, [res] * 3)
        sign_mode = 5
    else:
        tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [res] * 3)
        sign_mode = 3
    sdf = tmesh.query_points(grid_vertices, return_sdf=True, sign_mode=sign_mode)[-1]
    return grid_vertices, voxels, sdf.contiguous(), normals


def first_two(result):
    """Extractors return 2- to 5-tuples; the mesh is always the first two."""
    return result[0].contiguous(), result[1].contiguous().int()


def render_mesh(rnd, verts, faces, **kw):
    v = R.normalize_mesh(verts)
    return rnd.render(v, faces, **kw)


GT_TINT = (0.80, 0.83, 0.88)


def gt_mesh(tmesh):
    """The source asset, normalised the same way as every extraction.

    Shown first in every comparison so the reader has the reference in view
    rather than having to remember what the object is meant to look like.
    """
    return R.normalize_mesh(tmesh.vertices), tmesh.triangles.int().contiguous()


SURFACE_SAMPLES = 200_000


def surface_metrics(verts, faces, ref_points):
    """Chamfer and Hausdorff distance of a mesh against reference surface samples.

    Points are drawn area-weighted from the extracted surface rather than using
    its vertices, so the metric measures the surfaces themselves and is not
    biased by how densely each extractor happens to place vertices.

    Returns:
        (chamfer, hausdorff) as Euclidean distances in grid units.
    """
    from conquer3d.data_structure import TriangleMesh
    from conquer3d.ops import chamfer_distance, hausdorff_distance

    pts = TriangleMesh(verts.contiguous(), faces.int().contiguous()).sample_points(
        SURFACE_SAMPLES)[0].contiguous()
    scalar = lambda r: float(r[0] if isinstance(r, tuple) else r)
    cd = scalar(chamfer_distance(pts, ref_points, squared=False))
    hd = scalar(hausdorff_distance(pts, ref_points, squared=False))
    del pts
    return cd, hd


def divergence_point(verts_a, verts_b):
    """The point where two extractions disagree most.

    Found with the library's own one-sided Chamfer distance, so the zoom lands
    on real algorithmic disagreement instead of a hand-picked crop.
    """
    from conquer3d.ops import one_sided_chamfer_distance

    d = one_sided_chamfer_distance(verts_a, verts_b)
    d = d[0] if isinstance(d, tuple) else d
    d = d.reshape(-1)
    # The single worst vertex is often an isolated spike. Take the centroid of
    # the worst 1% instead, which lands on the feature the methods disagree
    # about rather than on an outlier.
    # A specific high-divergence vertex, not the centroid of many -- averaging
    # across separate creases lands in empty space between them.
    k = max(1, int(0.002 * d.numel()))
    idx = torch.topk(d, k).indices
    return verts_a[idx[len(idx) // 2]]


def style_axes(ax, fig):
    """Match matplotlib output to the site's restrained, theme-neutral look."""
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    for spine in ax.spines.values():
        spine.set_color("#8b93a7")
        spine.set_linewidth(0.8)
    ax.tick_params(colors="#8b93a7", labelsize=9)
    ax.xaxis.label.set_color("#7b8496")
    ax.yaxis.label.set_color("#7b8496")
    ax.title.set_color("#7b8496")


# --------------------------------------------------------------------------- #
# Figure 1 -- extraction algorithms compared
# --------------------------------------------------------------------------- #


RES_ALGO = 64
# Front-facing view of the Fandisk part.
AZ_ALGO = 218


def fig_algorithms(rnd):
    from conquer3d.ops import compute_hermite_from_mesh, dc, dmc, marching_cubes, mca

    tmesh = load_mesh(_asset("Fandisk"))
    tmesh.compute_triangle_normals()
    gv, vox, sdf, nrm = build_grid(tmesh, RES_ALGO, normal_mode=0)

    # The dual methods are shown at their best: exact Hermite data rather than
    # normals interpolated across the crease from the grid corners.
    ep, en = compute_hermite_from_mesh(tmesh, gv, vox, sdf)

    runs = {
        "Marching Cubes": first_two(marching_cubes(gv, vox, sdf, iso=0.0)),
        "MC Asymptotic": first_two(mca(gv, vox, sdf, iso=0.0)),
        "Dual Contouring": first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0,
                                        edge_points=ep, edge_normals=en)),
        "Dual Marching Cubes": first_two(dmc(gv, vox, sdf, iso=0.0,
                                             edge_points=ep, edge_normals=en)),
    }

    tints = [R.CYAN, R.GREEN, R.VIOLET, R.AMBER]
    accents = [(34, 211, 238), (118, 185, 0), (167, 139, 250), (251, 191, 36)]

    # Zoom where Marching Cubes and Dual Contouring disagree most: the crease.
    # Located from the plain Dual Contouring run, not the Hermite one, so that
    # supplying Hermite data changes the surfaces on show without also moving
    # the camera to a different part of the model.
    mc_v = R.normalize_mesh(runs["Marching Cubes"][0])
    dc_v = R.normalize_mesh(first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0))[0])
    target = divergence_point(mc_v, dc_v).tolist()

    # Ground truth leads, so every extraction is read against the original.
    gv_gt, gf_gt = gt_mesh(tmesh)
    wide = [rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=True,
                       azimuth=AZ_ALGO, elevation=20)]
    close = []  # filled once the crease target is known
    labels = ["Ground truth"]
    subs = [f"{gf_gt.shape[0]:,} faces"]
    accents = [(150, 158, 176)] + accents

    for (name, (v, f)), tint in zip(runs.items(), tints):
        nv = R.normalize_mesh(v)
        wide.append(rnd.render(nv, f, base=tint, flat=True, azimuth=AZ_ALGO, elevation=20))
        close.append(
            rnd.render(nv, f, base=tint, flat=True, azimuth=AZ_ALGO, elevation=20,
                       target=target, fit_radius=0.13, wireframe=0.004)
        )
        labels.append(name)
        subs.append(f"{v.shape[0]:,} verts · {f.shape[0]:,} faces")
        print(f"    {name:22} {v.shape[0]:>9,} verts  {f.shape[0]:>9,} faces")

    close.insert(0, rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=True, azimuth=AZ_ALGO,
                               elevation=20, target=target, fit_radius=0.13,
                               wireframe=0.004))
    wide = compose.trim(wide)
    close = compose.trim(close)
    top = compose.grid(wide, labels, sublabels=subs, accents=accents)
    bot = compose.grid(close, ["Crease detail"] * len(close),
                       sublabels=["reference edge"] + [f"{RES_ALGO}³ grid"] * (len(close) - 1),
                       accents=accents)
    compose.save(compose.stack([top, bot], pad=16), OUT / "fig-algorithms.png")


# --------------------------------------------------------------------------- #
# Figure 2 -- normal field modes
# --------------------------------------------------------------------------- #


def fig_normal_modes(rnd):
    from conquer3d.ops import dc

    tmesh = load_mesh(_asset("RockerArm"))
    panels, labels, subs = [], [], []
    names = ["Exact face normals", "Smooth vertex normals", "SDF gradient"]
    tints = [R.GREEN, R.CYAN, R.VIOLET]
    accents = [(118, 185, 0), (34, 211, 238), (167, 139, 250)]

    gv_gt, gf_gt = gt_mesh(tmesh)
    panels.append(rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=True,
                             azimuth=48, elevation=24))
    labels.append("Ground truth")
    subs.append(f"{gf_gt.shape[0]:,} faces")
    accents = [(150, 158, 176)] + accents

    for mode, name, tint in zip((0, 1, 2), names, tints):
        gv, vox, sdf, nrm = build_grid(tmesh, 224, normal_mode=mode)
        v, f = first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0))
        panels.append(render_mesh(rnd, v, f, base=tint, flat=True,
                                  azimuth=48, elevation=24))
        labels.append(name)
        subs.append(f"normal_mode={mode}")

    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs, accents=accents),
                 OUT / "fig-normal-modes.png")


# --------------------------------------------------------------------------- #
# Figure 3 -- sign determination modes
# --------------------------------------------------------------------------- #


def fig_sign_modes(rnd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmesh = load_mesh(_asset("Bimba"))
    res = 320
    tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [128] * 3)

    # A single axial slice through the model, queried under each sign mode.
    lin = torch.linspace(-1.0, 1.0, res, device=DEV)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1),
                       torch.zeros(res * res, device=DEV)], dim=-1).contiguous()

    names = {0: "Ray parity", 1: "Pseudonormal", 2: "Winding number",
             3: "Flood fill", 4: "Hybrid consensus", 5: "Coarse-fine fill"}
    fields = {}
    for mode, name in names.items():
        try:
            sdf = tmesh.query_points(pts, return_sdf=True, sign_mode=mode)[-1]
            fields[mode] = sdf.reshape(res, res).detach().cpu().numpy()
        except Exception as exc:
            print(f"    sign_mode={mode} unavailable: {str(exc)[:70]}")

    # Lead with the mesh being sliced, so the contours have a referent.
    gv_gt, gf_gt = gt_mesh(tmesh)
    shot = compose.trim([rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=False,
                                    azimuth=90, elevation=6, rim_strength=0.18)])[0]

    n = len(fields) + 1
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.5), dpi=170)
    axes = np.atleast_1d(axes)

    axes[0].imshow(shot, origin="upper")
    axes[0].set_title("Ground truth\nsliced at z = 0", fontsize=12, pad=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    style_axes(axes[0], fig)

    for ax, (mode, field) in zip(axes[1:], sorted(fields.items())):
        lim = float(np.percentile(np.abs(field), 97)) or 1.0
        ax.imshow(field, cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower",
                  extent=[-1, 1, -1, 1])
        ax.contour(field, levels=[0.0], colors=["#76b900"], linewidths=1.6,
                   extent=[-1, 1, -1, 1], origin="lower")
        ax.set_title(f"{names[mode]}\nsign_mode={mode}", fontsize=12, pad=9)
        ax.set_xticks([]); ax.set_yticks([])
        style_axes(ax, fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig-sign-modes.png", transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    to_webp(OUT / "fig-sign-modes.png")
    print(f"    wrote fig-sign-modes  ({n} modes)")


# --------------------------------------------------------------------------- #
# Figure 4 -- differential geometry on the GPU
# --------------------------------------------------------------------------- #


def fig_curvature(rnd):
    from conquer3d.ops import dmc

    # Igea is upright in its source frame and richly detailed -- ideal for
    # showing a curvature field.
    tmesh = load_mesh(_asset("Igea"))
    verts = tmesh.vertices
    faces = tmesh.triangles.int()

    panels, labels, subs = [], [], []
    panels.append(render_mesh(rnd, verts, faces, base=GT_TINT, flat=False,
                              azimuth=20, elevation=8, rim_strength=0.18))
    labels.append("Ground truth")
    subs.append(f"{verts.shape[0]:,} vertices")

    fields = [
        ("get_gaussian_curvature", "Gaussian curvature", "turbo", None),
        ("get_mean_curvature", "Mean curvature", "magma", None),
        # get_principal_curvatures returns (kappa1, kappa2) per vertex.
        ("get_principal_curvatures", "Principal curvature κ₁", "viridis", 0),
    ]
    for getter, name, cmap, column in fields:
        try:
            k = getattr(tmesh, getter)()
            k = k[:, column] if column is not None else k
            k = k.detach().cpu().numpy()
        except Exception as exc:
            print(f"    {getter} unavailable: {str(exc)[:70]}")
            continue
        cols = torch.tensor(compose.colormap(k, cmap, robust=6.0),
                            dtype=torch.float32, device=DEV)
        panels.append(render_mesh(rnd, verts, faces, colors=cols, flat=False,
                                  azimuth=20, elevation=8, rim_strength=0.18))
        labels.append(name)
        subs.append(f"{len(k):,} vertices")

    accents = [(150, 158, 176), (255, 137, 60), (200, 80, 160),
               (70, 190, 130)][: len(panels)]
    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs, accents=accents),
                 OUT / "fig-curvature.png")


# --------------------------------------------------------------------------- #
# Figure 5 -- the field behind the surface
# --------------------------------------------------------------------------- #


def fig_sdf_slices(rnd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmesh = load_mesh(_asset("Spot"))
    res = 340
    tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [128] * 3)

    lin = torch.linspace(-1.0, 1.0, res, device=DEV)
    slices = [-0.25, 0.0, 0.25]
    gv_gt, gf_gt = gt_mesh(tmesh)
    shot = compose.trim([rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=False,
                                    azimuth=200, elevation=12, rim_strength=0.18)])[0]

    fig, axes = plt.subplots(1, len(slices) + 1,
                             figsize=(3.4 * (len(slices) + 1), 3.6), dpi=170)
    axes = np.atleast_1d(axes)
    axes[0].imshow(shot, origin="upper")
    axes[0].set_title("Ground truth", fontsize=13, pad=9)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_visible(False)
    style_axes(axes[0], fig)

    for ax, z in zip(axes[1:], slices):
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        pts = torch.stack([xx.reshape(-1), yy.reshape(-1),
                           torch.full((res * res,), z, device=DEV)], dim=-1).contiguous()
        sdf = tmesh.query_points(pts, return_sdf=True, sign_mode=3)[-1]
        field = sdf.reshape(res, res).detach().cpu().numpy()

        lim = float(np.percentile(np.abs(field), 98)) or 1.0
        ax.imshow(field, cmap="Spectral", vmin=-lim, vmax=lim, origin="lower",
                  extent=[-1, 1, -1, 1])
        ax.contour(field, levels=np.linspace(-lim, lim, 13), colors=["#ffffff"],
                   linewidths=0.35, alpha=0.35, extent=[-1, 1, -1, 1], origin="lower")
        ax.contour(field, levels=[0.0], colors=["#76b900"], linewidths=2.0,
                   extent=[-1, 1, -1, 1], origin="lower")
        ax.set_title(f"z = {z:+.2f}", fontsize=13, pad=9)
        ax.set_xticks([]); ax.set_yticks([])
        style_axes(ax, fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig-sdf-slices.png", transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    to_webp(OUT / "fig-sdf-slices.png")


# --------------------------------------------------------------------------- #
# Figure 6 -- resolution ladder
# --------------------------------------------------------------------------- #


# Front-facing view of the Armadillo.
AZ_RES = 205
RES_LADDER = (64, 128, 256, 512, 1024, 1536, 2048)


def fig_resolution(rnd):
    from conquer3d.ops import dmc

    tmesh = load_mesh(_asset("Armadillo"))

    gv_gt, gf_gt = gt_mesh(tmesh)
    panels = [rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=True,
                         azimuth=AZ_RES, elevation=10)]
    labels = ["Ground truth"]
    subs = [f"{gf_gt.shape[0]:,} faces"]
    tints = [R.CYAN, R.GREEN, R.VIOLET, R.AMBER, R.ROSE,
             (0.45, 0.78, 0.62), (0.95, 0.60, 0.35)]
    accents = [(150, 158, 176), (34, 211, 238), (118, 185, 0), (167, 139, 250),
               (251, 191, 36), (251, 113, 133), (115, 199, 158), (242, 153, 89)]

    # Reference samples for the accuracy metrics, drawn once from the source.
    ref = tmesh.sample_points(SURFACE_SAMPLES)[0].contiguous()
    metrics = []

    for res, tint in zip(RES_LADDER, tints):
        gv, vox, sdf, _ = build_grid(tmesh, res)
        v, f = first_two(dmc(gv, vox, sdf, iso=0.0))
        panels.append(render_mesh(rnd, v, f, base=tint, flat=True,
                                  azimuth=AZ_RES, elevation=10))
        cd, hd = surface_metrics(v, f, ref)
        metrics.append((res, int(f.shape[0]), cd, hd))
        labels.append(f"{res}³")
        subs.append(f"{f.shape[0]:,} faces\nCD {cd * 1e3:.2f}e-3 · HD {hd * 1e3:.1f}e-3")
        print(f"    {res:>5}³  {f.shape[0]:>10,} faces   "
              f"chamfer {cd:.3e}   hausdorff {hd:.3e}")
        del gv, vox, sdf, v, f
        torch.cuda.empty_cache()

    Path(OUT.parent.parent / "_figures" / "resolution_metrics.json").write_text(
        __import__("json").dumps(
            [{"res": r, "faces": n, "chamfer": c, "hausdorff": h}
             for r, n, c, h in metrics], indent=2))

    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs, accents=accents),
                 OUT / "fig-resolution.png")


# --------------------------------------------------------------------------- #
# Figure 7 -- why narrow-band sparsity matters
# --------------------------------------------------------------------------- #


def fig_memory():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmesh = load_mesh(_asset("Fandisk"))
    resolutions = [32, 64, 128, 256, 512]
    dense, sparse = [], []

    for res in resolutions:
        gv, vox, _, _ = build_grid(tmesh, res)
        # 3 float32 per grid vertex, 8 int32 per voxel.
        sparse.append((gv.shape[0] * 3 * 4 + vox.shape[0] * 8 * 4) / 2**20)
        dense.append(((res + 1) ** 3 * 3 * 4 + res**3 * 8 * 4) / 2**20)
        print(f"      {res}³: sparse {sparse[-1]:8.1f} MB   dense {dense[-1]:9.1f} MB")
        del gv, vox
        torch.cuda.empty_cache()

    fig, ax = plt.subplots(figsize=(7.4, 4.4), dpi=170)
    ax.plot(resolutions, dense, "o--", color="#fb7185", lw=2, ms=6,
            label="Dense grid")
    ax.plot(resolutions, sparse, "o-", color="#76b900", lw=2.4, ms=6,
            label="Narrow-band sparse")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(resolutions)
    ax.set_xticklabels([f"{r}³" for r in resolutions])
    ax.set_xlabel("Grid resolution")
    ax.set_ylabel("Grid storage (MB)")
    ax.grid(alpha=0.15, color="#8b93a7", ls=":")
    leg = ax.legend(frameon=False)
    for text in leg.get_texts():
        text.set_color("#7b8496")
    factor = dense[-1] / sparse[-1]
    ax.annotate(f"{factor:.0f}× smaller at {resolutions[-1]}³",
                xy=(resolutions[-1], sparse[-1]), xytext=(-140, 34),
                textcoords="offset points", color="#76b900", fontsize=11,
                arrowprops=dict(arrowstyle="->", color="#76b900", lw=1.3))
    style_axes(ax, fig)
    fig.tight_layout()
    fig.savefig(OUT / "fig-memory.png", transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    to_webp(OUT / "fig-memory.png")
    print(f"    wrote fig-memory  ({factor:.0f}x at {resolutions[-1]}^3)")


# --------------------------------------------------------------------------- #


def to_webp(png_path: Path) -> None:
    """Convert a matplotlib PNG to WebP and drop the original."""
    from PIL import Image

    img = Image.open(png_path).convert("RGBA")
    out = png_path.with_suffix(".webp")
    # Lossless: these are line art and small text on transparency, which lossy
    # WebP smears into unreadable blocks. Flat regions keep the file small.
    img.save(out, "WEBP", lossless=True, method=6)
    png_path.unlink()
    print(f"    -> {out.name}  {out.stat().st_size / 1024:.0f} KB")


# --------------------------------------------------------------------------- #
# Figure 8 -- the extraction pipeline, step by step
# --------------------------------------------------------------------------- #

# Coarse enough that individual cells are legible as cubes.
RES_PIPE = 40
AZ_PIPE = 210


def fig_pipeline(rnd):
    """Walk one extraction end to end: mesh, cells, field, active cells, surface."""
    from conquer3d.ops import dmc

    tmesh = load_mesh(_asset("Spot"))
    gv, vox, sdf, _ = build_grid(tmesh, RES_PIPE)
    idx = vox.long()

    # One shared normalisation for every panel, so the cubes sit exactly where
    # the surface does instead of each panel being framed independently.
    src = tmesh.vertices.float()
    lo, hi = src.min(0).values, src.max(0).values
    centre = 0.5 * (lo + hi)
    radius = (src - centre).norm(dim=-1).max().clamp(min=1e-8)
    fit = lambda p: (p - centre) / radius

    cell_world = (BOUNDS_MAX[0] - BOUNDS_MIN[0]) / RES_PIPE
    cell = cell_world / float(radius)

    centres = fit(gv[idx].mean(1))
    corner_sdf = sdf[idx]                      # (M, 8)
    centre_sdf = corner_sdf.mean(1)
    # A cell contributes geometry only when its corners straddle the isolevel.
    bipolar = (corner_sdf.min(1).values < 0.0) & (corner_sdf.max(1).values >= 0.0)

    print(f"    {centres.shape[0]:,} narrow-band cells, "
          f"{int(bipolar.sum()):,} bipolar ({100 * bipolar.float().mean():.0f}%)")

    shot = lambda v, f, **kw: rnd.render(v, f, flat=True, azimuth=AZ_PIPE,
                                         elevation=16, **kw)
    panels, labels, subs = [], [], []

    # 1 -- the input.
    panels.append(shot(fit(src), tmesh.triangles.int(), base=GT_TINT))
    labels.append("1 · Input mesh")
    subs.append(f"{tmesh.triangles.shape[0]:,} triangles")

    # 2 -- the cells that were allocated at all.
    v2, f2, _ = R.cube_mesh(centres, cell * 0.88)
    panels.append(shot(v2, f2, base=R.CYAN, wireframe=0.02))
    labels.append("2 · Narrow-band cells")
    subs.append(f"{centres.shape[0]:,} cells, not {RES_PIPE ** 3:,}")

    # 3 -- the field those cells carry.
    # Narrow-band cells all sit near the isosurface, so percentile scaling would
    # flatten them into one colour. Normalise symmetrically about zero across a
    # couple of cell widths instead: blue inside, red outside, white at the
    # surface -- which is what the sign actually means.
    import matplotlib.cm as cm

    span = 2.5 * cell_world
    t = (centre_sdf.detach().cpu().numpy() / span).clip(-1.0, 1.0) * 0.5 + 0.5
    cols = torch.tensor(cm.get_cmap("coolwarm")(t)[:, :3],
                        dtype=torch.float32, device=DEV)
    # Seen from outside, every visible cell is positive and the field looks
    # uniform. Cutting the half nearest the camera away exposes the sign change
    # through the band: blue inside, red outside, white at the crossing.
    keep = centres[:, 0] >= 0.0
    v3, f3, c3 = R.cube_mesh(centres[keep], cell * 0.88, colors=cols[keep])
    panels.append(shot(v3, f3, colors=c3, wireframe=0.02, rim_strength=0.2))
    labels.append("3 · Signed distance")
    subs.append("cutaway: blue inside, red outside")

    # 4 -- the subset that actually emits geometry.
    v4, f4, _ = R.cube_mesh(centres[bipolar], cell * 0.88)
    panels.append(shot(v4, f4, base=R.AMBER, wireframe=0.02))
    labels.append("4 · Bipolar cells")
    subs.append(f"{int(bipolar.sum()):,} straddle the isolevel")

    # 5 -- the surface.
    ev, ef = first_two(dmc(gv, vox, sdf, iso=0.0))
    panels.append(shot(fit(ev), ef, base=R.GREEN))
    labels.append("5 · Extracted surface")
    subs.append(f"{ef.shape[0]:,} faces")

    accents = [(150, 158, 176), (34, 211, 238), (200, 90, 120),
               (251, 191, 36), (118, 185, 0)]
    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs,
                              accents=accents),
                 OUT / "fig-pipeline.png")



# --------------------------------------------------------------------------- #
# Figures 9-11 -- spatial data structures
# --------------------------------------------------------------------------- #

POINT_SIZE = 0.0085


def point_cloud(rnd, pts, colors, *, azimuth=210, elevation=14, size=POINT_SIZE, **kw):
    """Render a point set as small shaded cubes.

    Cubes rather than sprites: they catch the light, so depth and density read
    the way they do for the voxel figures, and the same renderer handles both.
    """
    v, f, c = R.cube_mesh(pts, size, colors=colors)
    return rnd.render(v, f, colors=c, flat=True, azimuth=azimuth,
                      elevation=elevation, **kw)


def fig_zcurve(rnd):
    """Morton ordering: the same points, before and after the space-filling sort."""
    from conquer3d.data_structure import z_curve_sort

    tmesh = load_mesh(_asset("Armadillo"))
    n = 45_000
    pts = tmesh.sample_points(n)[0].contiguous()
    fit = R.normalize_mesh(pts)

    rank = np.linspace(0.0, 1.0, n)
    ramp = lambda t: torch.tensor(compose.colormap(t, "turbo", robust=0.0),
                                  dtype=torch.float32, device=DEV)

    # Morton codes assume the unit cube.
    lo, hi = pts.min(0).values, pts.max(0).values
    unit = ((pts - lo) / (hi - lo).clamp(min=1e-8)).contiguous()
    out = z_curve_sort(unit)
    order = out[1] if isinstance(out, tuple) else out
    order = order.reshape(-1).long()

    panels = [
        point_cloud(rnd, fit, ramp(rank)),
        point_cloud(rnd, fit[order], ramp(rank)),
    ]
    labels = ["Insertion order", "Morton order"]
    subs = [f"{n:,} surface samples", "after z_curve_sort"]

    print(f"    {n:,} points")
    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs,
                              accents=[(150, 158, 176), (34, 211, 238)]),
                 OUT / "fig-zcurve.png")


def fig_meshbvh(rnd):
    """One ray against a sphere: which triangles it hits, and which voxels."""
    from conquer3d._C import BVH, MeshBVH
    from conquer3d.creation import create_sphere
    from conquer3d.data_structure import create_voxel_grid_from_tmesh, TriangleMesh

    verts, tris = create_sphere(26, 16, 1.0)
    verts = verts.cuda().float().contiguous()
    tris = tris.cuda().int().contiguous()

    # A near-tangent ray. On a diametric ray the entry and exit triangles face
    # opposite ways and one is always hidden; grazing the cap puts both on the
    # same visible side.
    origin = torch.tensor([[-3.0, 0.84, 0.0]], device=DEV)
    direction = torch.tensor([[1.0, 0.0, 0.0]], device=DEV)

    tv = verts[tris.long()]
    mesh_bvh = MeshBVH(tv.min(1).values.contiguous(), tv.max(1).values.contiguous())
    _, hit_tris, hit_pts, hit_dist = mesh_bvh.get_ray_intersection(
        origin, direction, verts, tris, True)
    hit_tris = hit_tris.reshape(-1).long()
    print(f"    sphere: {tris.shape[0]:,} triangles, ray hits {hit_tris.numel()}")

    # The ray is drawn as a dense chain of cubes, which works for any direction
    # without needing oriented box geometry.
    def ray_rod(t0, t1, n=260):
        t = torch.linspace(t0, t1, n, device=DEV)[:, None]
        return origin[0][None, :] + t * direction[0][None, :]

    scale = 1.0 / 1.0  # sphere already has unit radius
    fit = lambda p: p * scale

    def compose_scene(base_v, base_f, base_cols, extra):
        """Concatenate meshes into one draw so a single render shows them all."""
        vs, fs, cs, off = [base_v], [base_f], [base_cols], base_v.shape[0]
        for v, f, c in extra:
            vs.append(v); fs.append(f + off); cs.append(c); off += v.shape[0]
        return (torch.cat(vs).contiguous(), torch.cat(fs).int().contiguous(),
                torch.cat(cs).contiguous())

    NEUTRAL = torch.tensor([0.62, 0.66, 0.74], device=DEV)
    HIT = torch.tensor([0.46, 0.73, 0.0], device=DEV)
    RAY = torch.tensor([0.98, 0.75, 0.14], device=DEV)

    panels, labels, subs = [], [], []

    # --- panel 1: triangles the ray intersects ------------------------------
    tri_cols = NEUTRAL[None, :].expand(tris.shape[0], 3).clone()
    tri_cols[hit_tris] = HIT
    # Per-face colour needs per-face vertices, so the sphere is exploded first.
    ev = verts[tris.long()].reshape(-1, 3)
    ef = torch.arange(ev.shape[0], device=DEV, dtype=torch.int32).reshape(-1, 3)
    ec = tri_cols[:, None, :].expand(-1, 3, -1).reshape(-1, 3)

    rv, rf, rc = R.cube_mesh(ray_rod(0.0, 4.2), 0.028,
                             colors=RAY[None, :].expand(260, 3))
    v1, f1, c1 = compose_scene(fit(ev), ef, ec, [(fit(rv), rf, rc)])
    panels.append(rnd.render(v1, f1, colors=c1, flat=True, azimuth=28,
                             elevation=46, fit_radius=1.3, rim_strength=0.25))
    labels.append("Ray vs triangles")
    subs.append(f"{tris.shape[0]:,} triangles · {hit_tris.numel()} hit")

    # --- panel 2: voxels the ray intersects ---------------------------------
    tmesh = TriangleMesh(verts, tris)
    res = 22
    gv, vox = create_voxel_grid_from_tmesh(
        grid_min=[-1.3] * 3, grid_max=[1.3] * 3, res=[res] * 3,
        tmesh=tmesh, pad=1, return_normals=False)[:2]
    corners = gv[vox.long()]
    vmin = corners.min(1).values.contiguous()
    vmax = corners.max(1).values.contiguous()

    voxel_bvh = BVH(vmin, vmax)
    _, hit_vox = voxel_bvh.query_ray(origin, direction)
    hit_vox = hit_vox.reshape(-1).long().unique()
    print(f"    voxels: {vox.shape[0]:,} cells, ray crosses {hit_vox.numel()}")

    centres = 0.5 * (vmin + vmax)
    cell = float((vmax[0] - vmin[0]).max())
    vox_cols = NEUTRAL[None, :].expand(centres.shape[0], 3).clone() * 0.72
    vox_cols[hit_vox] = HIT
    sizes = torch.full((centres.shape[0], 1), cell * 0.20, device=DEV)
    sizes[hit_vox] = cell * 0.98

    cv, cf, cc = R.cube_mesh(centres, sizes, colors=vox_cols)
    v2, f2, c2 = compose_scene(fit(cv), cf, cc,
                               [(fit(rv), rf, rc)])
    panels.append(rnd.render(v2, f2, colors=c2, flat=True, azimuth=28,
                             elevation=46, fit_radius=1.3, rim_strength=0.2))
    labels.append("Ray vs voxels")
    subs.append(f"{vox.shape[0]:,} cells · {hit_vox.numel()} crossed")

    # --- panel 3: five rays, five origins -----------------------------------
    # One hierarchy, many independent queries. Each ray gets its own colour and
    # paints the triangles it hits in that colour, so which ray produced which
    # hit is readable without a legend.
    n_fan = 5
    a_r, e_r = math.radians(28.0), math.radians(46.0)
    eye = torch.tensor([math.cos(e_r) * math.sin(a_r), math.sin(e_r),
                        math.cos(e_r) * math.cos(a_r)], device=DEV)
    fwd = -eye
    right = torch.nn.functional.normalize(
        torch.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=DEV), dim=0), dim=0)
    up = torch.cross(right, fwd, dim=0)

    # Aim at five spots spread over the camera-facing cap, each approached from
    # its own direction, so the origins are genuinely apart rather than a fan.
    th = torch.linspace(0.0, 2.0 * math.pi, n_fan + 1, device=DEV)[:n_fan]
    tgt = (0.62 * (torch.cos(th)[:, None] * right[None, :]
                   + torch.sin(th)[:, None] * up[None, :])
           - 0.78 * fwd[None, :])
    phi = th + 0.9
    fan_d = torch.nn.functional.normalize(
        fwd[None, :] + 0.42 * (torch.cos(phi)[:, None] * right[None, :]
                               + torch.sin(phi)[:, None] * up[None, :]), dim=-1)
    fan_o = (tgt - 2.5 * fan_d).contiguous()
    fan_d = fan_d.contiguous()

    RAY_COLS = torch.tensor([[0.13, 0.83, 0.93], [0.46, 0.73, 0.00],
                             [0.65, 0.55, 0.98], [0.98, 0.75, 0.14],
                             [0.98, 0.44, 0.52]], device=DEV)

    fq, ftri, _, fdist = mesh_bvh.get_ray_intersection(fan_o, fan_d, verts, tris, True)
    fq = fq.reshape(-1).long()
    ftri = ftri.reshape(-1).long()
    print(f"    5 rays from 5 origins: {ftri.numel()} hits on "
          f"{ftri.unique().numel()} distinct triangles")

    fan_cols = NEUTRAL[None, :].expand(tris.shape[0], 3).clone()
    fan_cols[ftri] = RAY_COLS[fq % n_fan]
    ec3 = fan_cols[:, None, :].expand(-1, 3, -1).reshape(-1, 3)

    # Each ray stops at its own first hit; drawing full-length rods instead
    # builds a wall of geometry in front of the sphere.
    stop = torch.full((n_fan,), 3.4, device=DEV)
    if fq.numel():
        stop.scatter_reduce_(0, fq, fdist.reshape(-1), reduce="amin",
                             include_self=True)
    steps = 80
    frac = torch.linspace(0.0, 1.0, steps, device=DEV)[None, :, None]
    rod_pts = (fan_o[:, None, :]
               + frac * stop[:, None, None] * fan_d[:, None, :]).reshape(-1, 3)
    rod_cols = RAY_COLS[:, None, :].expand(n_fan, steps, 3).reshape(-1, 3)
    fv, ff, fc = R.cube_mesh(rod_pts, 0.026, colors=rod_cols)
    v3, f3, c3 = compose_scene(fit(ev), ef, ec3, [(fit(fv), ff, fc)])
    panels.append(rnd.render(v3, f3, colors=c3, flat=True, azimuth=28,
                             elevation=46, fit_radius=1.45, rim_strength=0.2))
    labels.append("5 rays, 5 origins")
    subs.append(f"{ftri.unique().numel()} triangles hit")

    compose.save(compose.grid(compose.trim(panels), labels, sublabels=subs,
                              accents=[(118, 185, 0), (251, 191, 36),
                                       (34, 211, 238)]),
                 OUT / "fig-meshbvh.png")


RES_NORM = 256
AZ_NORM = 215

NEUTRAL_N = 0.62  # grey stand-in for a normal that could not be computed


def normal_rgb(n):
    """The standard normal-map encoding: a unit direction as an RGB colour.

    Each axis maps to a channel, so a colour names an orientation directly and
    two normal fields can be compared by eye rather than against a legend.
    Vertices whose normal is not finite are left neutral grey rather than
    silently rendering as black.
    """
    bad = ~torch.isfinite(n).all(dim=-1, keepdim=True)
    safe = torch.where(bad, torch.zeros_like(n), n)
    rgb = (torch.nn.functional.normalize(safe, dim=-1) * 0.5 + 0.5).clamp(0.0, 1.0)
    return torch.where(bad, torch.full_like(rgb, NEUTRAL_N), rgb)


def vertex_normals(mesh):
    """Vertex normals, plus a mask of the ones that came back non-finite."""
    mesh.compute_vertex_normals(0)
    n = mesh.get_vertex_normals(0)
    return n, ~torch.isfinite(n).all(dim=-1)


def gt_normal_at(tmesh, gt_vn, points):
    """Ground-truth surface normal at the closest surface point to each input.

    The closest *face* normal is the obvious choice and the wrong one: it is
    piecewise constant, so comparing a smoothed extracted normal against it
    measures the reference mesh's own tessellation as much as the extraction.
    On Beast that self-disagreement is a median 5.64 degrees -- larger than the
    error being looked for. Interpolating the reference's vertex normals over
    the triangle the point projects onto compares like with like.
    """
    q = tmesh.query_points(points.contiguous(), return_sdf=False,
                           return_prj_pts=True, sign_mode=0)
    tri = q[1].reshape(-1).long()
    prj = q[2]
    T = tmesh.triangles.long()[tri]
    a, b, c = tmesh.vertices[T[:, 0]], tmesh.vertices[T[:, 1]], tmesh.vertices[T[:, 2]]

    e0, e1, e2 = b - a, c - a, prj - a
    d00 = (e0 * e0).sum(-1)
    d01 = (e0 * e1).sum(-1)
    d11 = (e1 * e1).sum(-1)
    d20 = (e2 * e0).sum(-1)
    d21 = (e2 * e1).sum(-1)
    den = (d00 * d11 - d01 * d01).clamp(min=1e-20)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    u = 1.0 - v - w
    return (u[:, None] * gt_vn[T[:, 0]]
            + v[:, None] * gt_vn[T[:, 1]]
            + w[:, None] * gt_vn[T[:, 2]])


def fig_normals(rnd):
    """Ground-truth normals against every extractor's, with angular deviation."""
    from conquer3d.data_structure import TriangleMesh
    from conquer3d.ops import (dc, dmc, marching_cubes,
                               marching_tetrahedra_grid, mca)

    tmesh = load_mesh(_asset("XYZRGBDragon"))

    # fix_normals rewrites the winding order in place, so the original has to be
    # captured first to say how many triangles actually turned.
    before = tmesh.triangles.clone()
    tmesh.fix_normals()
    flipped = int((tmesh.triangles != before).any(dim=1).sum())
    print(f"    fix_normals reoriented {flipped:,} of {before.shape[0]:,} triangles")

    gt_n, _ = vertex_normals(tmesh)
    tmesh.compute_triangle_normals()

    gv, vox, sdf, nrm = build_grid(tmesh, RES_NORM, normal_mode=0)
    runs = {
        "Marching Cubes": first_two(marching_cubes(gv, vox, sdf, iso=0.0)),
        "MC Asymptotic": first_two(mca(gv, vox, sdf, iso=0.0)),
        "Dual Contouring": first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0)),
        "Dual Marching Cubes": first_two(dmc(gv, vox, sdf, iso=0.0)),
        "Marching Tetrahedra": first_two(marching_tetrahedra_grid(gv, vox, sdf, iso=0.0)),
    }

    shot = dict(flat=False, azimuth=AZ_NORM, elevation=14, rim_strength=0.12)
    accent = [(34, 211, 238), (118, 185, 0), (167, 139, 250),
              (251, 191, 36), (251, 113, 133)]

    normal_panels, dev_panels = [], []
    n_labels, n_subs, d_labels, d_subs = [], [], [], []

    for name, (ev, ef) in runs.items():
        ex_n, ex_bad = vertex_normals(
            TriangleMesh(ev.contiguous(), ef.int().contiguous()))

        ref_n = gt_normal_at(tmesh, gt_n, ev)
        cos = (torch.nn.functional.normalize(ex_n, dim=-1)
               * torch.nn.functional.normalize(ref_n, dim=-1)).sum(-1).clamp(-1.0, 1.0)
        ang = torch.rad2deg(torch.arccos(cos))
        good = ang[~ex_bad]
        med = float(good.median())
        p99 = float(torch.quantile(good.float(), 0.99))
        # A median past 90 degrees means the whole surface is wound inward, which
        # is a convention flip rather than error. Surfaced rather than hidden, so
        # the figure cannot quietly misreport a regression here.
        inverted = med > 90.0
        outward = float((cos > 0).float().mean())
        print(f"    {name:<22} median {med:6.2f}deg  p99 {p99:6.2f}deg  "
              f"outward {100 * outward:5.1f}%  {int(ex_bad.sum()):>5} non-finite"
              f"{'  INVERTED' if inverted else ''}")

        rgb = torch.from_numpy(
            compose.colormap(torch.where(ex_bad, torch.zeros_like(ang), ang)
                             .detach().cpu().numpy(), "viridis", robust=15.0)
        ).float().to(DEV)
        rgb[ex_bad] = NEUTRAL_N

        normal_panels.append(render_mesh(rnd, ev, ef, colors=normal_rgb(ex_n), **shot))
        dev_panels.append(render_mesh(rnd, ev, ef, colors=rgb, **shot))
        n_labels.append(name)
        n_subs.append(f"{ef.shape[0]:,} faces"
                      + ("  \u00b7 winding inverted" if inverted else ""))
        d_labels.append("Angular deviation")
        d_subs.append(f"median {med:.1f}\u00b0 \u00b7 p99 {p99:.1f}\u00b0")

    # Trim every panel against one shared box so the reference and all ten
    # comparison panels keep the same scale.
    gt_v, gt_f = gt_mesh(tmesh)
    gt_panel = rnd.render(gt_v, gt_f, colors=normal_rgb(gt_n), **shot)
    panels = compose.trim([gt_panel] + normal_panels + dev_panels)

    left = compose.grid(panels[:1], ["Ground truth"],
                        sublabels=[f"{before.shape[0]:,} faces \u00b7 "
                                   f"{flipped:,} reoriented"],
                        accents=[(150, 158, 176)])
    right = compose.grid(panels[1:], n_labels + d_labels,
                         sublabels=n_subs + d_subs, cols=len(runs),
                         accents=accent + accent)
    compose.save(compose.hstack([left, right]), OUT / "fig-normals.png")


RES_HERM = 64
AZ_HERM = 218


def fig_hermite(rnd):
    """Dual methods with interpolated normals against exact Hermite data."""
    from conquer3d.ops import compute_hermite_from_mesh, dc, dmc

    tmesh = load_mesh(_asset("Fandisk"))
    tmesh.fix_normals()
    tmesh.compute_triangle_normals()
    gv, vox, sdf, nrm = build_grid(tmesh, RES_HERM, normal_mode=0)

    ep, en = compute_hermite_from_mesh(tmesh, gv, vox, sdf)
    crossings = int((en.norm(dim=-1) > 0).sum())
    print(f"    hermite data on {crossings:,} sign-crossing edges "
          f"of {en.shape[0] * 12:,}")

    runs = {
        "Dual Contouring": first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0)),
        "DC + Hermite": first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0,
                                     edge_points=ep, edge_normals=en)),
        "Dual Marching Cubes": first_two(dmc(gv, vox, sdf, iso=0.0)),
        "DMC + Hermite": first_two(dmc(gv, vox, sdf, iso=0.0,
                                       edge_points=ep, edge_normals=en)),
    }

    ref = tmesh.sample_points(SURFACE_SAMPLES)[0].contiguous()
    gt_vn, _ = vertex_normals(tmesh)

    # Crop where supplying Hermite data changes Dual Contouring most, so the
    # close row lands on a crease rather than on a hand-picked spot.
    target = divergence_point(R.normalize_mesh(runs["Dual Contouring"][0]),
                              R.normalize_mesh(runs["DC + Hermite"][0])).tolist()

    tints = [R.CYAN, R.GREEN, R.AMBER, R.VIOLET]
    accents = [(34, 211, 238), (118, 185, 0), (251, 191, 36), (167, 139, 250)]

    gt_v, gt_f = gt_mesh(tmesh)
    wide = [rnd.render(gt_v, gt_f, base=GT_TINT, flat=True,
                       azimuth=AZ_HERM, elevation=20)]
    close = [rnd.render(gt_v, gt_f, base=GT_TINT, flat=True, azimuth=AZ_HERM,
                        elevation=20, target=target, fit_radius=0.11,
                        wireframe=0.004)]
    labels = ["Ground truth"]
    subs = [f"{gt_f.shape[0]:,} faces"]

    for (name, (v, f)), tint in zip(runs.items(), tints):
        nv = R.normalize_mesh(v)
        wide.append(rnd.render(nv, f, base=tint, flat=True,
                               azimuth=AZ_HERM, elevation=20))
        close.append(rnd.render(nv, f, base=tint, flat=True, azimuth=AZ_HERM,
                                elevation=20, target=target, fit_radius=0.11,
                                wireframe=0.004))
        cd, hd = surface_metrics(v, f, ref)
        exn, bad = vertex_normals(
            __import__("conquer3d").data_structure.TriangleMesh(
                v.contiguous(), f.int().contiguous()))
        cos = (torch.nn.functional.normalize(exn, dim=-1)
               * torch.nn.functional.normalize(
                   gt_normal_at(tmesh, gt_vn, v), dim=-1)).sum(-1).clamp(-1.0, 1.0)
        med = float(torch.rad2deg(torch.arccos(cos))[~bad].median())
        labels.append(name)
        subs.append(f"normals {med:.2f}\u00b0 \u00b7 Hausdorff {hd:.2e}")
        print(f"    {name:<22} chamfer {cd:.3e}  hausdorff {hd:.3e}  "
              f"normal median {med:.2f}deg")

    wide = compose.trim(wide)
    close = compose.trim(close)
    cell = max(wide[0].shape[1], close[0].shape[1])
    wide = compose.pad_to(wide, cell)
    close = compose.pad_to(close, cell)
    acc = [(150, 158, 176)] + accents
    top = compose.grid(wide, labels, sublabels=subs, accents=acc)
    bot = compose.grid(close, ["Crease detail"] * len(close),
                       sublabels=["reference edge"] + [f"{RES_HERM}\u00b3 grid"] * 4,
                       accents=acc)
    compose.save(compose.stack([top, bot], pad=16), OUT / "fig-hermite.png")


AZ_QUALITY = 205


def fig_quality(rnd):
    """Every quality metric a TriangleMesh reports, drawn on the mesh."""
    tmesh = load_mesh(_asset("Lucy"))
    tmesh.fix_normals()
    tmesh.compute_triangle_normals()
    tmesh.compute_triangle_areas()
    tmesh.compute_vertices_to_triangle_map()
    tmesh.compute_edges_to_triangle_map()

    # Two of the seven are whole-mesh scalars rather than fields, so they label
    # the reference panel instead of colouring one: get_quality() is the (min,
    # mean) of the regularity field, and valence_567_percentage is a single
    # share of the vertices.
    q_min, q_mean = tmesh.get_quality()
    valence = tmesh.valence_567_percentage
    print(f"    get_quality() min {q_min:.4f} mean {q_mean:.4f} · "
          f"valence 5-7 {valence:.1f}%")

    fields = [
        ("Aspect ratio", tmesh.get_aspect_ratio(0)),
        ("Radii ratio", tmesh.get_radii_ratio()),
        ("Radius\u2013edge ratio", tmesh.get_radius_edge_ratio()),
        ("Regularity", tmesh.get_triangle_regularity()),
        ("Angle deviation", tmesh.get_angle_deviation()),
    ]

    # Lucy is stored lying down, long axis along z with the pedestal at -z (the
    # +z end is the wingspan, which is wider but is not the base). Standing her
    # up is display-only: every metric above is rotation-invariant and is
    # computed from the mesh exactly as stored.
    flat_v = R.normalize_mesh(tmesh.vertices)
    verts = torch.stack(
        [flat_v[:, 0], flat_v[:, 2], -flat_v[:, 1]], dim=-1
    ).contiguous()
    faces = tmesh.triangles.int().contiguous()
    # A per-triangle field needs per-triangle vertices.
    ex_v = verts[faces.long()].reshape(-1, 3).contiguous()
    ex_f = torch.arange(ex_v.shape[0], device=DEV, dtype=torch.int32).reshape(-1, 3)

    shot = dict(flat=True, azimuth=AZ_QUALITY, elevation=16, rim_strength=0.15)
    accents = [(150, 158, 176), (251, 113, 133), (251, 191, 36),
               (34, 211, 238), (118, 185, 0), (167, 139, 250)]

    panels = [rnd.render(verts, faces, base=GT_TINT, **shot)]
    labels = ["Quality"]
    subs = [f"min {q_min:.3f} \u00b7 mean {q_mean:.3f}"
            f"\nvalence 5\u20137: {valence:.1f}%"]

    for name, values in fields:
        vals = values.detach().float().reshape(-1)
        # One ramp for every field, so a colour means the same thing throughout.
        rgb = torch.from_numpy(
            compose.colormap(vals.cpu().numpy(), "viridis")
        ).float().to(DEV)
        cols = rgb[:, None, :].expand(-1, 3, -1).reshape(-1, 3).contiguous()
        panels.append(rnd.render(ex_v, ex_f, colors=cols, **shot))
        labels.append(name)
        subs.append(f"mean {float(vals.mean()):.3g}")
        print(f"    {name:<20} mean {float(vals.mean()):.4g}  "
              f"range {float(vals.min()):.3g} \u2013 {float(vals.max()):.3g}")

    compose.save(
        compose.grid(compose.trim(panels), labels, sublabels=subs, cols=3,
                     accents=accents),
        OUT / "fig-quality.png",
    )


def _asset(name):
    import conquer3d.data.assets as assets

    return getattr(assets, name)


FIGURES = [
    ("algorithms", fig_algorithms, True),
    ("normal modes", fig_normal_modes, True),
    ("sign modes", fig_sign_modes, True),
    ("curvature", fig_curvature, True),
    ("sdf slices", fig_sdf_slices, True),
    ("resolution", fig_resolution, True),
    ("memory", fig_memory, False),
    ("pipeline", fig_pipeline, True),
    ("zcurve", fig_zcurve, True),
    ("meshbvh", fig_meshbvh, True),
    ("normals", fig_normals, True),
    ("hermite", fig_hermite, True),
    ("quality", fig_quality, True),
]


def main() -> int:
    only = set(sys.argv[1:])
    OUT.mkdir(parents=True, exist_ok=True)
    rnd = R.Renderer(size=760, ssaa=2)

    failures = []
    for name, fn, needs_renderer in FIGURES:
        if only and name.split()[0] not in only:
            continue
        print(f"[{name}]")
        t0 = time.time()
        try:
            fn(rnd) if needs_renderer else fn()
            print(f"    done in {time.time() - t0:.1f}s")
        except Exception:
            failures.append(name)
            traceback.print_exc(limit=3)
        torch.cuda.empty_cache()

    print("\n" + "-" * 60)
    if failures:
        print("FAILED:", ", ".join(failures))
    else:
        print("all figures generated")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
