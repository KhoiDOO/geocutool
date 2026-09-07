#!/usr/bin/env python
"""Measure Conquer3D on the local GPU and emit results for the benchmarks page.

Run with the project's conda environment::

    ~/anaconda3/envs/geocutool/bin/python docs/_figures/make_benchmarks.py

Three experiments, all timed with CUDA events around the operator alone -- grid
construction and field evaluation happen first and are not counted, except in
the experiment that measures them explicitly. Every timing is the median of
several runs after a warm-up, so one-off allocation costs do not leak in.

Writes ``docs/_figures/benchmarks.json``, which the site generator renders.
"""

from __future__ import annotations

import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from make_figures import BOUNDS_MAX, BOUNDS_MIN, _asset, build_grid, first_two, load_mesh

OUT = HERE / "benchmarks.json"

WARMUP = 2
REPEATS = 7


def timed(fn, warmup: int = WARMUP, repeats: int = REPEATS) -> float:
    """Median wall time of ``fn`` in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


# --------------------------------------------------------------------------- #
# Experiment 1 -- extraction latency across algorithms and resolutions
# --------------------------------------------------------------------------- #


def bench_extraction(asset: str, resolutions) -> list:
    from conquer3d.ops import (dc, dmc, marching_cubes, marching_tetrahedra_grid, mca)

    tmesh = load_mesh(_asset(asset))
    rows = []

    for res in resolutions:
        gv, vox, sdf, nrm = build_grid(tmesh, res)
        algos = {
            "Marching Cubes": lambda: marching_cubes(gv, vox, sdf, iso=0.0),
            "MC Asymptotic": lambda: mca(gv, vox, sdf, iso=0.0),
            "Dual Contouring": lambda: dc(gv, vox, sdf, grid_normals=nrm, iso=0.0),
            "Dual Marching Cubes": lambda: dmc(gv, vox, sdf, iso=0.0),
            "DMC (pure quads)": lambda: dmc(gv, vox, sdf, iso=0.0, quad_split=False),
            "Marching Tetrahedra": lambda: marching_tetrahedra_grid(gv, vox, sdf, iso=0.0),
        }
        for name, fn in algos.items():
            try:
                ms = timed(fn)
                v, f = first_two(fn())
                rows.append({
                    "algorithm": name, "res": res, "ms": round(ms, 3),
                    "vertices": int(v.shape[0]), "faces": int(f.shape[0]),
                    "faces_per_s": int(f.shape[0] / (ms / 1000.0)),
                })
                print(f"    {res:>5}³  {name:<22} {ms:7.2f} ms  "
                      f"{f.shape[0]:>9,} faces  {f.shape[0] / (ms / 1000) / 1e6:7.1f}M f/s")
            except Exception as exc:
                print(f"    {res:>5}³  {name:<22} failed: {str(exc)[:60]}")
        del gv, vox, sdf, nrm
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 2 -- cost of each sign determination mode
# --------------------------------------------------------------------------- #

SIGN_NAMES = {
    0: "Ray parity", 1: "Pseudonormal", 2: "Winding number",
    3: "Flood fill", 4: "Hybrid consensus", 5: "Coarse-fine fill",
}


# Point counts in millions. Mostly primes, so no size is a multiple of another
# and any cache or block-size resonance shows up rather than hiding.
SIGN_SWEEP_M = (1, 2, 3, 5, 7, 11, 13, 17, 19, 23)


def bench_sign_modes(asset: str) -> list:
    """Sweep every sign mode across a wide range of query-set sizes."""
    tmesh = load_mesh(_asset(asset))
    tmesh.build_flood_fill_data(BOUNDS_MIN, BOUNDS_MAX, [256] * 3)
    tmesh.build_flood_fill_cf_data(BOUNDS_MIN, BOUNDS_MAX, [256] * 3)

    rows = []
    for millions in SIGN_SWEEP_M:
        n = millions * 1_000_000
        torch.manual_seed(0)
        pts = (torch.rand((n, 3), device="cuda") * 2.0 - 1.0).contiguous()
        line = [f"    {millions:>3}M pts:"]
        for mode, name in SIGN_NAMES.items():
            try:
                fn = lambda m=mode: tmesh.query_points(pts, return_sdf=True, sign_mode=m)
                ms = timed(fn, warmup=1, repeats=3)
                rows.append({
                    "mode": mode, "name": name, "points": n, "ms": round(ms, 3),
                    "points_per_s": int(n / (ms / 1000.0)),
                })
                line.append(f"{name.split()[0][:5]}={ms:7.1f}")
            except Exception as exc:
                line.append(f"{name.split()[0][:5]}=ERR")
                print(f"      sign_mode={mode} @ {millions}M failed: {str(exc)[:50]}")
        print("  ".join(line))
        del pts
        torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 4 -- Chamfer and Hausdorff distance cost
# --------------------------------------------------------------------------- #

DIST_SWEEP = (100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000)


def bench_distances(asset: str) -> list:
    """Time the distance metrics on surface samples of increasing size."""
    from conquer3d.ops import (chamfer_distance, hausdorff_distance,
                               one_sided_chamfer_distance,
                               one_sided_hausdorff_distance)

    tmesh = load_mesh(_asset(asset))
    rows = []
    for n in DIST_SWEEP:
        try:
            a = tmesh.sample_points(n)[0].contiguous()
            b = tmesh.sample_points(n)[0].contiguous()
            ops = {
                "One-sided Chamfer": lambda: one_sided_chamfer_distance(a, b),
                "Chamfer": lambda: chamfer_distance(a, b, squared=False),
                "One-sided Hausdorff": lambda: one_sided_hausdorff_distance(a, b),
                "Hausdorff": lambda: hausdorff_distance(a, b, squared=False),
            }
            parts = [f"    {n / 1e6:5.2f}M pts:"]
            for name, fn in ops.items():
                ms = timed(fn, warmup=1, repeats=3)
                rows.append({
                    "metric": name, "points": n, "ms": round(ms, 3),
                    "points_per_s": int(n / (ms / 1000.0)),
                })
                parts.append(f"{name.split()[-1][:4]}{'1' if 'One' in name else ''}={ms:7.2f}")
            print("  ".join(parts))
            del a, b
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"    {n / 1e6:5.2f}M pts: failed: {str(exc)[:60]}")
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #
# Experiment 3 -- what the pipeline costs before extraction begins
# --------------------------------------------------------------------------- #


def bench_pipeline(asset: str, resolutions) -> list:
    from conquer3d.data_structure import create_voxel_grid_from_tmesh

    tmesh = load_mesh(_asset(asset))
    rows = []
    for res in resolutions:
        try:
            grid_fn = lambda r=res: create_voxel_grid_from_tmesh(
                grid_min=BOUNDS_MIN, grid_max=BOUNDS_MAX, res=[r] * 3,
                tmesh=tmesh, pad=1, return_normals=True, normal_mode=0,
            )
            grid_ms = timed(grid_fn, warmup=1, repeats=3)
            out = grid_fn()
            gv, vox = out[0], out[1]

            if res >= 1024:
                fill_fn = lambda r=res: tmesh.build_flood_fill_cf_data(
                    BOUNDS_MIN, BOUNDS_MAX, [r] * 3)
                mode = 5
            else:
                fill_fn = lambda r=res: tmesh.build_flood_fill_data(
                    BOUNDS_MIN, BOUNDS_MAX, [r] * 3)
                mode = 3
            fill_ms = timed(fill_fn, warmup=1, repeats=3)

            sdf_fn = lambda: tmesh.query_points(gv, return_sdf=True, sign_mode=mode)
            sdf_ms = timed(sdf_fn, warmup=1, repeats=3)

            rows.append({
                "res": res, "grid_ms": round(grid_ms, 3), "fill_ms": round(fill_ms, 3),
                "sdf_ms": round(sdf_ms, 3), "cells": int(vox.shape[0]),
                "grid_vertices": int(gv.shape[0]),
                "dense_cells": res ** 3,
                "sparsity": round(100.0 * vox.shape[0] / res ** 3, 4),
            })
            print(f"    {res:>5}³  grid {grid_ms:7.1f} ms  fill {fill_ms:7.1f} ms  "
                  f"sdf {sdf_ms:7.1f} ms  {vox.shape[0]:>9,} cells "
                  f"({100 * vox.shape[0] / res**3:.2f}% of dense)")
            del gv, vox, out
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"    {res:>5}³  failed: {str(exc)[:70]}")
            torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

PALETTE = {
    "Marching Cubes": "#8b93a7", "MC Asymptotic": "#22d3ee",
    "Dual Contouring": "#76b900", "Dual Marching Cubes": "#a78bfa",
    "DMC (pure quads)": "#fbbf24", "Marching Tetrahedra": "#fb7185",
}


def plot_all(results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from make_figures import style_axes, to_webp

    img = HERE.parent / "assets" / "img"
    img.mkdir(parents=True, exist_ok=True)

    # --- extraction latency ------------------------------------------------
    rows = results["extraction"]
    algos = list(dict.fromkeys(r["algorithm"] for r in rows))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=170)

    for algo in algos:
        pts = sorted((r["res"], r["ms"], r["faces_per_s"]) for r in rows
                     if r["algorithm"] == algo)
        xs = [p[0] for p in pts]
        ax1.plot(xs, [p[1] for p in pts], "o-", lw=2, ms=5,
                 color=PALETTE.get(algo, "#8b93a7"), label=algo)
        ax2.plot(xs, [p[2] / 1e6 for p in pts], "o-", lw=2, ms=5,
                 color=PALETTE.get(algo, "#8b93a7"), label=algo)

    for ax, ylabel, title in (
        (ax1, "Extraction latency (ms)", "Latency"),
        (ax2, "Throughput (M faces/s)", "Throughput"),
    ):
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024])
        ax.set_xticklabels(["128³", "256³", "512³", "1024³"])
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=12, pad=9)
        ax.grid(alpha=0.15, color="#8b93a7", ls=":")
        style_axes(ax, fig)
    ax1.set_yscale("log")
    leg = ax1.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color("#7b8496")
    fig.tight_layout()
    fig.savefig(img / "fig-bench-extraction.png", transparent=True,
                bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    to_webp(img / "fig-bench-extraction.png")

    # --- pipeline cost breakdown -------------------------------------------
    rows = results["pipeline"]
    if rows:
        res = [r["res"] for r in rows]
        x = range(len(res))
        grid = [r["grid_ms"] for r in rows]
        fill = [r["fill_ms"] for r in rows]
        sdf = [r["sdf_ms"] for r in rows]

        fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=170)
        ax.bar(x, grid, 0.6, label="Grid construction", color="#76b900")
        ax.bar(x, fill, 0.6, bottom=grid, label="Flood fill build", color="#22d3ee")
        ax.bar(x, sdf, 0.6, bottom=[g + f for g, f in zip(grid, fill)],
               label="SDF query", color="#a78bfa")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{r}³" for r in res])
        ax.set_xlabel("Grid resolution")
        ax.set_ylabel("Setup time (ms)")
        ax.set_yscale("log")
        ax.grid(alpha=0.15, color="#8b93a7", ls=":", axis="y")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color("#7b8496")
        style_axes(ax, fig)
        fig.tight_layout()
        fig.savefig(img / "fig-bench-pipeline.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        to_webp(img / "fig-bench-pipeline.png")

    # --- sign mode cost across query-set size -------------------------------
    rows = results.get("sign_modes", [])
    if rows:
        modes = sorted({r["mode"] for r in rows})
        colors = {0: "#fb7185", 1: "#76b900", 2: "#22d3ee",
                  3: "#a78bfa", 4: "#fbbf24", 5: "#8b93a7"}
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.4), dpi=170)
        for mode in modes:
            pts = sorted((r["points"], r["ms"], r["points_per_s"])
                         for r in rows if r["mode"] == mode)
            name = next(r["name"] for r in rows if r["mode"] == mode)
            xs = [p[0] / 1e6 for p in pts]
            ax1.plot(xs, [p[1] for p in pts], "o-", lw=2, ms=4,
                     color=colors.get(mode, "#8b93a7"), label=f"{name} ({mode})")
            ax2.plot(xs, [p[2] / 1e6 for p in pts], "o-", lw=2, ms=4,
                     color=colors.get(mode, "#8b93a7"), label=f"{name} ({mode})")

        for ax, ylabel, title, logy in (
            (ax1, "Query time (ms)", "Cost scales linearly", True),
            (ax2, "Throughput (M points/s)", "Rate is flat with size", False),
        ):
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            ax.set_xlabel("Query points (millions)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=12, pad=9)
            ax.grid(alpha=0.15, color="#8b93a7", ls=":", which="both")
            style_axes(ax, fig)
        leg = ax2.legend(frameon=False, fontsize=8.5, loc="lower right")
        for t in leg.get_texts():
            t.set_color("#7b8496")
        fig.tight_layout()
        fig.savefig(img / "fig-bench-signmodes.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        to_webp(img / "fig-bench-signmodes.png")

    # --- distance metric cost ------------------------------------------------
    rows = results.get("distances", [])
    if rows:
        metrics = list(dict.fromkeys(r["metric"] for r in rows))
        colors = {"One-sided Chamfer": "#76b900", "Chamfer": "#22d3ee",
                  "One-sided Hausdorff": "#fbbf24", "Hausdorff": "#a78bfa"}
        fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=170)
        for metric in metrics:
            pts = sorted((r["points"], r["ms"]) for r in rows if r["metric"] == metric)
            ax.plot([p[0] / 1e6 for p in pts], [p[1] for p in pts], "o-", lw=2, ms=5,
                    color=colors.get(metric, "#8b93a7"), label=metric)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Points per cloud (millions)")
        ax.set_ylabel("Time (ms)")
        ax.grid(alpha=0.15, color="#8b93a7", ls=":", which="both")
        leg = ax.legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color("#7b8496")
        style_axes(ax, fig)
        fig.tight_layout()
        fig.savefig(img / "fig-bench-distance.png", transparent=True,
                    bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
        to_webp(img / "fig-bench-distance.png")


def main() -> int:
    torch.cuda.init()
    meta = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "measured": time.strftime("%Y-%m-%d"),
        "warmup": WARMUP,
        "repeats": REPEATS,
    }
    print(f"{meta['gpu']} · torch {meta['torch']} · CUDA {meta['cuda']}")

    results = {"meta": meta}

    print("\n[extraction latency]  asset=Fandisk")
    results["extraction"] = bench_extraction("Fandisk", [128, 256, 512, 1024])

    print("\n[sign mode cost]  asset=Bimba, 1M-23M query points")
    results["sign_modes"] = bench_sign_modes("Bimba")

    print("\n[distance metrics]  asset=Armadillo")
    results["distances"] = bench_distances("Armadillo")

    print("\n[pipeline cost]  asset=Fandisk")
    results["pipeline"] = bench_pipeline("Fandisk", [128, 256, 512, 1024])

    OUT.write_text(json.dumps(results, indent=2))

    print("\n[plots]")
    plot_all(results)
    print(f"\nwrote {OUT.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
