"""GPU-accelerated Differentiable Dual Contouring (DC) with Jacobi QEF solvers.

Dual Contouring (Ju et al., 2002) extracts explicit surface meshes from volumetric scalar
fields while preserving sharp features, creases, and corners. For each active voxel cell
intersected by the isosurface, a single dual vertex is placed at the feature point minimizing
the Quadratic Error Function (QEF):

$$E(v) = \\sum_{i} \\left( n_i \\cdot (v - p_i) \\right)^2$$

where $p_i$ are Hermite edge intersection points and $n_i$ are the corresponding surface normals.
The QEF minimum is solved in parallel on the GPU using cyclic Jacobi Singular Value
Decomposition (SVD) on register arrays.

Example:
    >>> import torch
    >>> from conquer3d.ops import dual_contouring
    >>> # Extract sharp mesh from voxel grid and signed distance field
    >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf, iso=0.0, quad_split=True)
"""

from typing import Tuple, Optional, Union
import torch
from .. import _C


# NOTE: Differentiable autograd functionality is currently disabled/commented out.
# class DiffDualContouring(torch.autograd.Function):
#     """PyTorch autograd Function for Differentiable Dual Contouring on CUDA.
# 
#     Implements forward surface extraction with Jacobi QEF solvers and backward
#     adjoint gradient propagation with respect to input scalar SDF values and vertex colors.
#     """
# 
#     @staticmethod
#     def forward(
#         ctx,
#         grid_vertices: torch.Tensor,
#         voxels: torch.Tensor,
#         sdf: torch.Tensor,
#         grid_normals: Optional[torch.Tensor] = None,
#         colors: Optional[torch.Tensor] = None,
#         voxel_vertices: Optional[torch.Tensor] = None,
#         iso: float = 0.0,
#         quad_split: bool = True
#     ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
#         if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
#             raise RuntimeError("dual_contouring requires CUDA tensors")
# 
#         grid_vertices = grid_vertices.contiguous().float()
#         voxels = voxels.contiguous().int()
#         sdf = sdf.contiguous().float()
# 
#         if grid_normals is not None:
#             grid_normals = grid_normals.contiguous().float()
#         if colors is not None:
#             colors = colors.contiguous().float()
#         if voxel_vertices is not None:
#             voxel_vertices = voxel_vertices.contiguous().float()
# 
#         verts, faces, out_colors = _C.dual_contouring(
#             grid_vertices, voxels, sdf, grid_normals, colors, voxel_vertices, iso, quad_split
#         )
# 
#         ctx.save_for_backward(grid_vertices, voxels, sdf, grid_normals, colors)
#         ctx.iso = iso
#         ctx.has_colors = colors is not None
# 
#         return verts, faces, out_colors
# 
#     @staticmethod
#     def backward(ctx, grad_verts, grad_faces, grad_colors):
#         grid_vertices, voxels, sdf, grid_normals, colors = ctx.saved_tensors
#         iso = ctx.iso
# 
#         if not sdf.requires_grad and (colors is None or not colors.requires_grad):
#             return None, None, None, None, None, None, None, None
# 
#         grad_sdf, grad_colors_in = _C.dual_contouring_backward(
#             grad_verts.contiguous(),
#             grad_colors.contiguous() if grad_colors is not None else None,
#             grid_vertices,
#             voxels,
#             sdf,
#             grid_normals,
#             colors,
#             iso
#         )
# 
#         return None, None, grad_sdf, None, grad_colors_in, None, None, None


def dual_contouring(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    grid_normals: Optional[torch.Tensor] = None,
    colors: Optional[torch.Tensor] = None,
    voxel_vertices: Optional[torch.Tensor] = None,
    iso: float = 0.0,
    quad_split: bool = True,
    edge_points: Optional[torch.Tensor] = None,
    edge_normals: Optional[torch.Tensor] = None
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extracts sharp surface meshes from volumetric scalar fields using Differentiable Dual Contouring.

    Preserves sharp creases, mechanical edges, and corners by positioning dual vertices
    at the optimal Quadratic Error Function (QEF) minimizer using register-level Jacobi SVD,
    or directly from precomputed inside-voxel vertex coordinates `voxel_vertices`.

    Args:
        grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` containing grid vertex
            coordinates on CUDA. Must be contiguous.
        voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` containing 8 corner vertex indices
            per voxel cell on CUDA.
        sdf (torch.Tensor): Float32 tensor of shape `(N,)` containing scalar SDF values on CUDA.
        grid_normals (torch.Tensor, optional): Float32 tensor of shape `(N, 3)` containing explicit
            surface normals at grid vertices. If None, evaluated on-the-fly via analytical
            trilinear cell gradients. Defaults to None.
        colors (torch.Tensor, optional): Float32 tensor of shape `(N, C)` containing vertex feature
            colors on CUDA. Defaults to None.
        voxel_vertices (torch.Tensor, optional): Float32 tensor of shape `(M, 3)` containing
            precomputed inside-voxel vertex coordinates on CUDA. If provided, DC bypasses QEF
            solving and directly uses these coordinates for active surface cells. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.
        quad_split (bool, optional): If True, splits dual quadrilaterals into 2 triangles using
            the optimal Delaunay angle criterion; if False, returns quads of shape `(Q, 4)`.
            Defaults to True.
        edge_points (torch.Tensor, optional): Float32 tensor of shape `(M, 12, 3)` giving the
            exact surface intersection point of each voxel-local edge, in the
            :data:`conquer3d.ops.hermite.DC_EDGE_CORNERS` order. Supplying this together with
            `edge_normals` is what lets Dual Contouring reproduce sharp creases exactly: the
            QEF is then built from true Hermite data instead of from normals interpolated
            between grid vertices, which blend across a crease and round it off. Entries for
            edges that are not sign-crossing are ignored. Defaults to None.
        edge_normals (torch.Tensor, optional): Float32 tensor of shape `(M, 12, 3)` giving the
            exact surface normal at each `edge_points` entry. Must accompany `edge_points`;
            either both are used or neither. Defaults to None.

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
        >>> from conquer3d.ops import dual_contouring
        >>> # Extract sharp 3D surface mesh with QEF
        >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf, iso=0.0)
        >>> # Or extract with precomputed inside vertices
        >>> verts, faces = dual_contouring(grid_vertices, voxels, sdf, voxel_vertices=precomputed_pts, iso=0.0)
    """
    grid_vertices = grid_vertices.contiguous().float()
    voxels = voxels.contiguous().int()
    sdf = sdf.contiguous().float()
    if grid_normals is not None:
        grid_normals = grid_normals.contiguous().float()
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
    #     verts, faces, out_colors = DiffDualContouring.apply(
    #         grid_vertices, voxels, sdf, grid_normals, colors, voxel_vertices, iso, quad_split
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

    verts, faces, out_colors = _C.dual_contouring(
        grid_vertices, voxels, sdf, grid_normals, colors, voxel_vertices, iso, quad_split,
        edge_points, edge_normals
    )

    if colors is None:
        return verts, faces
    return verts, faces, out_colors


# Public alias
dc = dual_contouring
