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

    tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [res] * 3)
    sdf = tmesh.query_points(grid_vertices, return_sdf=True, sign_mode=3)[-1]
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
    from conquer3d.ops import dc, dmc, marching_cubes, mca

    tmesh = load_mesh(_asset("Fandisk"))
    gv, vox, sdf, nrm = build_grid(tmesh, RES_ALGO, normal_mode=0)

    runs = {
        "Marching Cubes": first_two(marching_cubes(gv, vox, sdf, iso=0.0)),
        "MC Asymptotic": first_two(mca(gv, vox, sdf, iso=0.0)),
        "Dual Contouring": first_two(dc(gv, vox, sdf, grid_normals=nrm, iso=0.0)),
        "Dual Marching Cubes": first_two(dmc(gv, vox, sdf, iso=0.0)),
    }

    tints = [R.CYAN, R.GREEN, R.VIOLET, R.AMBER]
    accents = [(34, 211, 238), (118, 185, 0), (167, 139, 250), (251, 191, 36)]

    # Zoom where Marching Cubes and Dual Contouring disagree most: the crease.
    mc_v = R.normalize_mesh(runs["Marching Cubes"][0])
    dc_v = R.normalize_mesh(runs["Dual Contouring"][0])
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
                       sublabels=["reference edge"] + [f"{RES_ALGO}³ grid"] * 4,
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


def fig_sign_modes():
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

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.5), dpi=170)
    axes = np.atleast_1d(axes)
    for ax, (mode, field) in zip(axes, sorted(fields.items())):
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


def fig_sdf_slices():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tmesh = load_mesh(_asset("Spot"))
    res = 340
    tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [128] * 3)

    lin = torch.linspace(-1.0, 1.0, res, device=DEV)
    slices = [-0.25, 0.0, 0.25]
    fig, axes = plt.subplots(1, len(slices), figsize=(3.4 * len(slices), 3.6), dpi=170)
    for ax, z in zip(np.atleast_1d(axes), slices):
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


def fig_resolution(rnd):
    from conquer3d.ops import dmc

    tmesh = load_mesh(_asset("Armadillo"))

    gv_gt, gf_gt = gt_mesh(tmesh)
    panels = [rnd.render(gv_gt, gf_gt, base=GT_TINT, flat=True,
                         azimuth=AZ_RES, elevation=10)]
    labels = ["Ground truth"]
    subs = [f"{gf_gt.shape[0]:,} faces"]
    tints = [R.CYAN, R.GREEN, R.VIOLET, R.AMBER, R.ROSE]
    accents = [(150, 158, 176), (34, 211, 238), (118, 185, 0),
               (167, 139, 250), (251, 191, 36), (251, 113, 133)]

    for res, tint in zip((64, 128, 256, 512, 1024), tints):
        gv, vox, sdf, _ = build_grid(tmesh, res)
        v, f = first_two(dmc(gv, vox, sdf, iso=0.0))
        panels.append(render_mesh(rnd, v, f, base=tint, flat=True,
                                  azimuth=AZ_RES, elevation=10))
        labels.append(f"{res}³")
        subs.append(f"{f.shape[0]:,} faces")
        del gv, vox, sdf
        torch.cuda.empty_cache()

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


def _asset(name):
    import conquer3d.data.assets as assets

    return getattr(assets, name)


FIGURES = [
    ("algorithms", fig_algorithms, True),
    ("normal modes", fig_normal_modes, True),
    ("sign modes", fig_sign_modes, False),
    ("curvature", fig_curvature, True),
    ("sdf slices", fig_sdf_slices, False),
    ("resolution", fig_resolution, True),
    ("memory", fig_memory, False),
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
