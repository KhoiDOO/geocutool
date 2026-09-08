r"""Exact Hermite data for sharp-feature Dual Contouring.

Dual Contouring (Ju et al., 2002) places one vertex per cell at the minimiser of

$$E(v) = \sum_i \left( n_i \cdot (v - p_i) \right)^2$$

and that minimiser only lands on a crease when each $(p_i, n_i)$ is the *true*
surface intersection of the edge and the *true* surface normal there -- the
Hermite data the method is named for. Deriving $n_i$ instead by interpolating
normals stored at the two grid corners blends across the crease, because the
two corners belong to different faces, and the QEF then reconstructs neither
face and rounds the feature off.

This module builds genuine Hermite data for the 12 edges of every cell, in the
edge order the CUDA kernel uses, ready to hand to
:func:`conquer3d.ops.dual_contouring` through its ``edge_points`` and
``edge_normals`` arguments.

Two sources are provided:

- :func:`compute_hermite_from_mesh` -- for a :class:`~conquer3d.data_structure.TriangleMesh`,
  using the BVH to find the closest surface point and its face normal.
- :func:`compute_hermite_from_field` -- for an implicit or neural field, refining the
  crossing along the edge and taking the normal from the field gradient.

Example:
    >>> from conquer3d.ops import compute_hermite_from_mesh, dual_contouring
    >>> ep, en = compute_hermite_from_mesh(tmesh, grid_vertices, voxels, sdf)
    >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf,
    ...                                edge_points=ep, edge_normals=en)
"""

from typing import Callable, Optional, Tuple
import torch

__all__ = ["DC_EDGE_CORNERS", "compute_hermite_from_mesh", "compute_hermite_from_field"]

#: Corner pair spanned by each of the 12 voxel-local edges, matching
#: ``dc_edge_corners`` in ``conquer3d/csrc/ops/dc_data.h``. The order matters:
#: ``edge_points[m, e]`` must describe the same edge the kernel reads at slot ``e``.
DC_EDGE_CORNERS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (3, 2), (0, 3),
    (4, 5), (5, 6), (7, 6), (4, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def _bipolar(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    iso: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Locates every sign-crossing voxel edge and its linearly interpolated crossing.

    Returns:
        Tuple of (mask, p0, p1, p_lin), each shaped ``(M, 12, ...)``. The mask marks
        edges the isosurface crosses; ``p_lin`` is the crossing under the linear
        assumption, used only as a seed for the exact solve.
    """
    vx = voxels.long()
    m = vx.shape[0]
    dev = grid_vertices.device

    mask = torch.zeros((m, 12), dtype=torch.bool, device=dev)
    p0 = torch.zeros((m, 12, 3), dtype=torch.float32, device=dev)
    p1 = torch.zeros((m, 12, 3), dtype=torch.float32, device=dev)
    p_lin = torch.zeros((m, 12, 3), dtype=torch.float32, device=dev)

    for e, (ca, cb) in enumerate(DC_EDGE_CORNERS):
        i0, i1 = vx[:, ca], vx[:, cb]
        s0, s1 = sdf[i0], sdf[i1]
        mask[:, e] = ((s0 < iso) & (s1 >= iso)) | ((s0 >= iso) & (s1 < iso))
        a, b = grid_vertices[i0], grid_vertices[i1]
        denom = torch.where(s1 - s0 == 0, torch.ones_like(s0), s1 - s0)
        t = ((iso - s0) / denom).clamp(0.0, 1.0).unsqueeze(-1)
        p0[:, e], p1[:, e], p_lin[:, e] = a, b, a + (b - a) * t
    return mask, p0, p1, p_lin


@torch.no_grad()
def compute_hermite_from_mesh(
    tmesh,
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    iso: float = 0.0,
    chunk: int = 4_000_000,
    exact: bool = True,
    ray_chunk: int = 1_000_000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds exact Hermite data for every sign-crossing voxel edge from a mesh.

    Each crossing is found by casting the edge as a bounded segment against the mesh
    and keeping its first hit, which gives both the true intersection point and the
    triangle the edge actually pierces. A cell straddling a crease therefore receives
    the two genuine face normals rather than a blend of them, and the QEF minimum
    lands on the crease. Edges the cast misses -- possible on an open or degenerate
    mesh -- fall back to snapping the linear seed to the nearest surface point.

    Args:
        tmesh (TriangleMesh): Mesh whose surface the field represents, on CUDA.
        grid_vertices (torch.Tensor): `(V, 3)` float32 grid coordinates on CUDA.
        voxels (torch.Tensor): `(M, 8)` int32 corner indices on CUDA.
        sdf (torch.Tensor): `(V,)` float32 scalar field at the grid vertices.
        iso (float, optional): Isolevel being extracted. Defaults to 0.0.
        chunk (int, optional): Maximum crossings per BVH query, to bound peak
            memory on large grids. Defaults to 4,000,000.
        exact (bool, optional): Ray-cast each edge as a bounded segment to obtain
            the true intersection and the triangle actually pierced, instead of
            snapping the linear seed to the nearest surface point. Defaults to True.
        ray_chunk (int, optional): Maximum edges per ray-cast batch, to bound the
            hit buffer. Defaults to 1,000,000.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(edge_points, edge_normals)``, both
        `(M, 12, 3)` float32 on CUDA. Rows for edges the surface does not cross are
        left zero and are ignored by the extractor.

    Raises:
        ValueError: If the tensors are not CUDA-resident or are shaped unexpectedly.
    """
    if not (grid_vertices.is_cuda and voxels.is_cuda and sdf.is_cuda):
        raise ValueError("grid_vertices, voxels and sdf must be CUDA tensors")
    if voxels.dim() != 2 or voxels.shape[1] != 8:
        raise ValueError(f"voxels must have shape (M, 8), got {tuple(voxels.shape)}")

    grid_vertices = grid_vertices.contiguous().float()
    sdf = sdf.contiguous().float()

    mask, p0, p1, p_lin = _bipolar(grid_vertices, voxels, sdf, iso)
    edge_points = torch.zeros_like(p_lin)
    edge_normals = torch.zeros_like(p_lin)

    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return edge_points, edge_normals

    a = p0.reshape(-1, 3)[idx]
    b = p1.reshape(-1, 3)[idx]
    tri_normals = tmesh.triangle_normals

    # Baseline: snap the linear seed to the surface. Used directly when exact
    # ray-casting is disabled, and as the fallback for any edge the ray misses.
    seeds = p_lin.reshape(-1, 3)[idx]
    out_p = torch.empty_like(seeds)
    out_n = torch.empty_like(seeds)
    for s in range(0, seeds.shape[0], chunk):
        q = seeds[s:s + chunk].contiguous()
        res = tmesh.query_points(q, return_sdf=False, return_prj_pts=True, sign_mode=0)
        out_p[s:s + chunk] = res[2]
        out_n[s:s + chunk] = tri_normals.index_select(0, res[1].long())

    if exact:
        # Cast each edge as a bounded segment and keep its first hit. This is the
        # true edge-surface intersection and the triangle actually pierced, which
        # is what the QEF needs to reconstruct both faces of a crease.
        seg = b - a
        lens = seg.norm(dim=-1)
        good = lens > 1e-12
        dirs = torch.zeros_like(seg)
        dirs[good] = seg[good] / lens[good].unsqueeze(-1)
        tol = 1e-6
        for s in range(0, a.shape[0], ray_chunk):
            e = min(s + ray_chunk, a.shape[0])
            o_c = a[s:e].contiguous()
            d_c = dirs[s:e].contiguous()
            l_c = lens[s:e]
            rid, tid, hit, dist = tmesh.get_ray_intersection(o_c, d_c, True)
            if rid.numel() == 0:
                continue
            rid = rid.long()
            keep = (dist >= -tol) & (dist <= l_c.index_select(0, rid) + tol)
            if not bool(keep.any()):
                continue
            rid, tid, hit, dist = rid[keep], tid[keep], hit[keep], dist[keep]
            # first hit per ray: sort by (ray, distance) and take each run's head
            order = torch.argsort(rid.double() * 1e12 + dist.double().clamp_min(0))
            rid, tid, hit = rid[order], tid[order], hit[order]
            head = torch.ones_like(rid, dtype=torch.bool)
            head[1:] = rid[1:] != rid[:-1]
            rid, tid, hit = rid[head], tid[head], hit[head]
            out_p[s:e].index_copy_(0, rid, hit)
            out_n[s:e].index_copy_(0, rid, tri_normals.index_select(0, tid.long()))

    edge_points.reshape(-1, 3)[idx] = out_p
    edge_normals.reshape(-1, 3)[idx] = out_n
    return edge_points.contiguous(), edge_normals.contiguous()


@torch.no_grad()
def compute_hermite_from_field(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    field_fn: Callable[[torch.Tensor], torch.Tensor],
    iso: float = 0.0,
    refine_iters: int = 24,
    grad_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    eps: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds Hermite data from an implicit or neural field, with no mesh required.

    The crossing is refined along the edge by bisection on ``field_fn`` rather than
    assumed linear, which matters near a crease where the field is not linear along
    the edge. The normal comes from ``grad_fn`` when given, otherwise from central
    differences of ``field_fn``.

    Args:
        grid_vertices (torch.Tensor): `(V, 3)` float32 grid coordinates on CUDA.
        voxels (torch.Tensor): `(M, 8)` int32 corner indices on CUDA.
        sdf (torch.Tensor): `(V,)` float32 field sampled at the grid vertices.
        field_fn (Callable): Maps `(N, 3)` points to `(N,)` field values.
        iso (float, optional): Isolevel being extracted. Defaults to 0.0.
        refine_iters (int, optional): Bisection steps per edge. Defaults to 24.
        grad_fn (Callable, optional): Maps `(N, 3)` points to `(N, 3)` gradients.
            Falls back to central differences of `field_fn`. Defaults to None.
        eps (float, optional): Central-difference step. Defaults to 1e-4.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: ``(edge_points, edge_normals)``, both
        `(M, 12, 3)` float32 on CUDA, zero on edges the surface does not cross.
    """
    grid_vertices = grid_vertices.contiguous().float()
    sdf = sdf.contiguous().float()

    mask, p0, p1, _ = _bipolar(grid_vertices, voxels, sdf, iso)
    edge_points = torch.zeros_like(p0)
    edge_normals = torch.zeros_like(p0)

    idx = mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        return edge_points, edge_normals

    a = p0.reshape(-1, 3)[idx]
    b = p1.reshape(-1, 3)[idx]
    fa = field_fn(a) - iso
    neg_at_a = fa < 0

    lo = torch.zeros((a.shape[0], 1), device=a.device)
    hi = torch.ones_like(lo)
    for _ in range(refine_iters):
        mid = 0.5 * (lo + hi)
        fm = field_fn(a + (b - a) * mid) - iso
        same = (fm < 0) == neg_at_a
        same = same.unsqueeze(-1)
        lo = torch.where(same, mid, lo)
        hi = torch.where(same, hi, mid)
    pts = a + (b - a) * (0.5 * (lo + hi))

    if grad_fn is not None:
        n = grad_fn(pts)
    else:
        n = torch.empty_like(pts)
        for k in range(3):
            off = torch.zeros_like(pts)
            off[:, k] = eps
            n[:, k] = field_fn(pts + off) - field_fn(pts - off)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    edge_points.reshape(-1, 3)[idx] = pts
    edge_normals.reshape(-1, 3)[idx] = n
    return edge_points.contiguous(), edge_normals.contiguous()
