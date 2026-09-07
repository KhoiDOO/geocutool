r"""GPU-accelerated Marching Cubes with Asymptotic Decider (Topologically Consistent MC).

Classical Marching Cubes (Lorensen & Cline, 1987) suffers from topological ambiguity on
voxel faces where diagonally opposite corners share identical signs, leading to holes,
cracks, and non-manifold surfaces in extracted meshes. The Asymptotic Decider (Nielson &
Hamann, 1991) resolves these ambiguities by computing the hyperbolic asymptotic saddle point
value $S = \frac{B_{00} B_{11} - B_{01} B_{10}}{B_{00} + B_{11} - B_{01} - B_{10}}$ on each
ambiguous bilinear face, connecting positive vertices if $S > \\text{iso}$ and negative vertices
otherwise. This guarantees extracted meshes are watertight and topologically consistent.

Example:
    >>> import torch
    >>> from conquer3d.ops import marching_cubes_asymptotic
    >>> # Extract watertight 2-manifold surface mesh
    >>> verts, faces = marching_cubes_asymptotic(grid_vertices, voxels, sdf, iso=0.0)
"""

from typing import Tuple, Optional, Union
import torch
from .. import _C


# NOTE: Differentiable autograd functionality is currently disabled/commented out.
# class DiffMarchingCubesAsymptotic(torch.autograd.Function):
#     """PyTorch autograd Function for Topologically Consistent Marching Cubes on CUDA."""
# 
#     @staticmethod
#     def forward(
#         ctx,
#         grid_vertices: torch.Tensor,
#         voxels: torch.Tensor,
#         sdf: torch.Tensor,
#         colors: Optional[torch.Tensor] = None,
#         iso: float = 0.0
#     ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
#         if not grid_vertices.is_cuda or not voxels.is_cuda or not sdf.is_cuda:
#             raise RuntimeError("marching_cubes_asymptotic requires CUDA tensors")
# 
#         grid_vertices = grid_vertices.contiguous().float()
#         voxels = voxels.contiguous().int()
#         sdf = sdf.contiguous().float()
#         if colors is not None:
#             colors = colors.contiguous().float()
# 
#         verts, faces, out_colors = _C.marching_cubes_asymptotic(
#             grid_vertices, voxels, sdf, colors, iso
#         )
# 
#         ctx.save_for_backward(grid_vertices, voxels, sdf, colors, verts, faces)
#         ctx.iso = iso
#         ctx.has_colors = colors is not None
# 
#         return verts, faces, out_colors
# 
#     @staticmethod
#     def backward(ctx, grad_verts, grad_faces, grad_colors):
#         grid_vertices, voxels, sdf, colors, verts, faces = ctx.saved_tensors
#         iso = ctx.iso
# 
#         if not sdf.requires_grad and (colors is None or not colors.requires_grad):
#             return None, None, None, None, None
# 
#         grad_sdf = torch.zeros_like(sdf)
#         grad_colors_in = torch.zeros_like(colors) if colors is not None else None
# 
#         return None, None, grad_sdf, grad_colors_in, None


def marching_cubes_asymptotic(
    grid_vertices: torch.Tensor,
    voxels: torch.Tensor,
    sdf: torch.Tensor,
    colors: Optional[torch.Tensor] = None,
    iso: float = 0.0
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extracts a watertight 2-manifold surface mesh using Marching Cubes with Asymptotic Deciders.
    
    Dynamically resolves all 6 ambiguous bilinear face configurations on the GPU to guarantee
    a topologically consistent, crack-free surface.

    Args:
        grid_vertices (torch.Tensor): Float32 tensor of shape `(N, 3)` containing 3D grid
            vertex coordinates on CUDA. Must be contiguous.
        voxels (torch.Tensor): Int32 tensor of shape `(M, 8)` containing 8 corner vertex indices
            per voxel cell on CUDA.
        sdf (torch.Tensor): Float32 tensor of shape `(N,)` containing scalar SDF values on CUDA.
        colors (torch.Tensor, optional): Float32 tensor of shape `(N, C)` containing vertex feature
            colors on CUDA. Defaults to None.
        iso (float, optional): Isosurface extraction threshold. Defaults to 0.0.

    Returns:
        Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
            - If `colors` is None: Returns `(vertices, faces)`.
            - If `colors` is provided: Returns `(vertices, faces, colors)`.
            - `vertices`: Float32 tensor of shape `(V, 3)` on CUDA.
            - `faces`: Int32 tensor of shape `(F, 3)` on CUDA.
            - `colors`: Float32 tensor of shape `(V, C)` on CUDA.

    Raises:
        RuntimeError: If inputs are not on CUDA or not contiguous.

    Example:
        >>> import torch
        >>> from conquer3d.ops import marching_cubes_asymptotic
        >>> # Extract watertight 2-manifold surface mesh
        >>> verts, faces = marching_cubes_asymptotic(grid_vertices, voxels, sdf, iso=0.0)
    """
    grid_vertices = grid_vertices.contiguous().float()
    voxels = voxels.contiguous().int()
    sdf = sdf.contiguous().float()
    if colors is not None:
        colors = colors.contiguous().float()

    verts, faces, out_colors = _C.marching_cubes_asymptotic(
        grid_vertices, voxels, sdf, colors, iso
    )
    if colors is None:
        return verts, faces
    return verts, faces, out_colors


# Public alias
mca = marching_cubes_asymptotic
