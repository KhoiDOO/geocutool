"""GPU-accelerated Differentiable Dual Marching Cubes (DMC).

Dual Marching Cubes (Schaefer & Warren, 2004) combines the sharp feature preservation
and clean quad topology of Dual Contouring with the strict 2-manifold topological guarantees
of Marching Cubes. Unlike standard Dual Contouring which generates at most 1 dual vertex
per voxel cell (causing topological pinch points and self-intersections when multiple surface
sheets intersect a cell), Dual Marching Cubes generates multiple dual vertices per cell—one
for each independent MC contour—and projects them onto the exact trilinear zero-isosurface
using Newton-Raphson level-set iterations.

Example:
    >>> import torch
    >>> from conquer3d.ops import dual_marching_cubes
    >>> # Extract strictly 2-manifold isosurface mesh
    >>> verts, faces = dual_marching_cubes(grid_vertices, voxels, sdf, iso=0.0, project_iters=5)
"""

from typing import Tuple, Optional, Union
import torch
from .. import _C


# NOTE: Differentiable autograd functionality is currently disabled/commented out.
# class DiffDualMarchingCubes(torch.autograd.Function):
#     """PyTorch autograd Function for Differentiable Dual Marching Cubes on CUDA.
# 
#     Executes forward surface extraction with independent cell contours and Newton-Raphson
#     level-set projections, with analytical backward gradient propagation with respect to
#     input scalar SDF values and vertex colors.
#     """
# 
#     @staticmethod
#     def forward(
#         ctx,
#         grid_vertices: torch.Tensor,
#         voxels: torch.Tensor,
#         sdf: torch.Tensor,
#         colors: Optional[torch.Tensor] = None,
#         voxel_vertices: Optional[torch.Tensor] = None,
#         iso: float = 0.0,
#         quad_split: bool = True,
#         project_iters: int = 5
#     ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
#         if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
#             raise RuntimeError("dual_marching_cubes requires CUDA tensors")
# 
#         grid_vertices = grid_vertices.contiguous().float()
#         voxels = voxels.contiguous().int()
#         sdf = sdf.contiguous().float()
# 
#         if colors is not None:
#             colors = colors.contiguous().float()
#         if voxel_vertices is not None:
#             voxel_vertices = voxel_vertices.contiguous().float()
# 
#         verts, faces, out_colors = _C.dual_marching_cubes(
#             grid_vertices, voxels, sdf, colors, voxel_vertices, iso, quad_split, project_iters
#         )
# 
#         ctx.save_for_backward(grid_vertices, voxels, sdf, colors)
#         ctx.iso = iso
#         ctx.project_iters = project_iters
#         ctx.has_colors = colors is not None
# 
#         return verts, faces, out_colors
# 
#     @staticmethod
#     def backward(ctx, grad_verts, grad_faces, grad_colors):
#         grid_vertices, voxels, sdf, colors = ctx.saved_tensors
#         iso = ctx.iso
#         project_iters = ctx.project_iters
# 
#         if not sdf.requires_grad and (colors is None or not colors.requires_grad):
#             return None, None, None, None, None, None, None, None
# 
#         grad_sdf, grad_colors_in = _C.dual_marching_cubes_backward(
#             grad_verts.contiguous(),
#             grad_colors.contiguous() if grad_colors is not None else None,
#             grid_vertices,
#             voxels,
#             sdf,
#             colors,
#             iso,
#             project_iters
#         )
# 
#         return None, None, grad_sdf, grad_colors_in, None, None, None, None


def dual_marching_cubes(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    voxel_vertices: Optional[torch.Tensor] = None,
    iso: float = 0.0,
    quad_split: bool = True,
    project_iters: int = 5,
    edge_points: Optional[torch.Tensor] = None,
    edge_normals: Optional[torch.Tensor] = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extracts a strictly 2-manifold surface mesh from a voxel grid using Differentiable Dual Marching Cubes.

    Unlike standard Dual Contouring which generates at most 1 dual vertex per voxel cell,
    Dual Marching Cubes decomposes each voxel cell into independent contours using asymptotic
    deciders and extracts multiple dual vertices per cell, eliminating topological pinch points
    and guaranteeing manifold surfaces. Dual vertices can be iteratively projected onto the exact
    trilinear level set via Newton-Raphson optimization or assigned directly from `voxel_vertices`.

    Args:
        grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` containing grid vertex
            coordinates on CUDA. Must be contiguous.
        voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` containing 8 corner vertex indices
            per voxel cell on CUDA.
        sdf (torch.Tensor): Float32 tensor of shape `(N,)` containing scalar SDF values on CUDA.
        colors (torch.Tensor, optional): Float32 tensor of shape `(N, C)` containing vertex feature
            colors on CUDA. Defaults to None.
        voxel_vertices (torch.Tensor, optional): Float32 tensor of shape `(M, 3)` containing
            precomputed inside-voxel vertex coordinates on CUDA. If provided, DMC bypasses
            Newton-Raphson level-set projections and assigns these coordinates directly. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
        quad_split (bool, optional): If True, splits dual quadrilaterals into 2 triangles using
            the optimal Delaunay angle criterion; if False, returns quads of shape `(Q, 4)`.
            Defaults to True.
        edge_points (torch.Tensor, optional): Float32 tensor of shape `(M, 12, 3)` giving the
            exact surface intersection of each voxel-local edge, in
            :data:`conquer3d.ops.hermite.DC_EDGE_CORNERS` order. Supplied together with
            `edge_normals`, this replaces the centroid-and-projection placement with a
            quadratic error function solved per contour, which is what lets Dual Marching
            Cubes reproduce sharp creases instead of rounding them. Solving per contour
            rather than per cell keeps a cell that carries several contours emitting
            several distinct vertices, preserving the manifold guarantee. Defaults to None.
        edge_normals (torch.Tensor, optional): Float32 tensor of shape `(M, 12, 3)` giving the
            exact surface normal at each `edge_points` entry. Must accompany `edge_points`.
            Defaults to None.
        project_iters (int, optional): Number of Newton-Raphson level-set projection iterations.
            Defaults to 5.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `colors` is None: Returns `(vertices, faces)`.
            - If `colors` is provided: Returns `(vertices, faces, colors)`.
            - `vertices`: Float32 tensor of shape `(V, 3)` on CUDA.
            - `faces`: Int32 tensor of shape `(F, 3)` for triangles or `(Q, 4)` for quads on CUDA.
            - `colors`: Float32 tensor of shape `(V, C)` on CUDA.

    Raises:
        RuntimeError: If inputs are not on CUDA or not contiguous.
        ValueError: If `voxel_vertices` does not have shape `(M, 3)`.

    Example:
        >>> import torch
        >>> from conquer3d.ops import dual_marching_cubes
        >>> # Extract strictly 2-manifold surface mesh
        >>> verts, faces = dual_marching_cubes(grid_vertices, voxels, sdf, iso=0.0)
    """
    grid_vertices = grid_vertices.contiguous().float()
    voxels = voxels.contiguous().int()
    sdf = sdf.contiguous().float()
    if colors is not None:
        colors = colors.contiguous().float()
    if voxel_vertices is not None:
        if not voxel_vertices.is_cuda:
            raise RuntimeError("voxel_vertices must be on CUDA")
        if voxel_vertices.dtype != torch.float32:
            voxel_vertices = voxel_vertices.float()
        if voxel_vertices.ndim != 2 or voxel_vertices.shape[0] != voxels.shape[0] or voxel_vertices.shape[1] != 3:
            raise ValueError(f"voxel_vertices must have shape (M, 3) matching voxels ({voxels.shape[0]}, 3), got {voxel_vertices.shape}")
        voxel_vertices = voxel_vertices.contiguous()

    # if sdf.requires_grad or (colors is not None and colors.requires_grad):
    #     verts, faces, out_colors = DiffDualMarchingCubes.apply(
    #         grid_vertices, voxels, sdf, colors, voxel_vertices, iso, quad_split, project_iters
    #     )
    # else:
    if (edge_points is None) != (edge_normals is None):
        raise ValueError("edge_points and edge_normals must be supplied together")
    if edge_points is not None:
        edge_points = edge_points.contiguous().float()
        edge_normals = edge_normals.contiguous().float()
        for name, t in (("edge_points", edge_points), ("edge_normals", edge_normals)):
            if not t.is_cuda:
                raise RuntimeError(f"{name} must be on CUDA")
            if t.shape != (voxels.shape[0], 12, 3):
                raise ValueError(
                    f"{name} must have shape ({voxels.shape[0]}, 12, 3), got {tuple(t.shape)}"
                )

    verts, faces, out_colors = _C.dual_marching_cubes(
        grid_vertices, voxels, sdf, colors, voxel_vertices, iso, quad_split, project_iters,
        edge_points, edge_normals
    )

    if colors is None:
        return verts, faces
    return verts, faces, out_colors


# Public alias
dmc = dual_marching_cubes
