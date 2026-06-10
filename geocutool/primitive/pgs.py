import torch
from typing import Tuple, Optional

from .._C import (
    query_pgs_voxel_pair_intersection_brute_force,
    query_pgs_edge_intersection_brute_force
)

def query_pgs_voxel_pair_intersection(
    vx_aabb_mins: torch.Tensor,
    vx_aabb_maxs: torch.Tensor,
    means: torch.Tensor,
    normals: torch.Tensor,
    covis: torch.Tensor,
    gs_aabb_mins: torch.Tensor,
    gs_aabb_maxs: torch.Tensor,
    iso: float = 11.345,
    return_centroids: bool = False,
    return_centroid_densities: bool = False,
    max_capacity: int = 10000000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Queries intersections between Gaussians and voxels using a brute-force approach.

    Args:
        vx_aabb_mins (torch.Tensor): (M, 3) tensor of voxel AABB minimum corners.
        vx_aabb_maxs (torch.Tensor): (M, 3) tensor of voxel AABB maximum corners.
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        normals (torch.Tensor): (N, 3) tensor of Gaussian principal axes.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        gs_aabb_mins (torch.Tensor): (N, 3) tensor of Gaussian AABB minimum corners.
        gs_aabb_maxs (torch.Tensor): (N, 3) tensor of Gaussian AABB maximum corners.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        return_centroids (bool, optional): Whether to return the centroids and densities. Defaults to False.
        max_capacity (int, optional): Maximum number of intersections to store. Defaults to 10000000.

    Returns:
        hit_mask: torch.Tensor: (M,) boolean mask of voxels that experienced at least one hit.
        out_voxel_ids: torch.Tensor: (K,) tensor of voxel IDs with intersections.
        out_gaus_ids: torch.Tensor: (K,) tensor of Gaussian IDs with intersections.
        centroids: torch.Tensor or None: (K, 3) tensor of centroids of the intersections, if requested.
        densities: torch.Tensor or None: (K,) tensor of densities of the intersections, if requested.
    """
    
    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not all(t.is_cuda for t in [
        vx_aabb_mins, vx_aabb_maxs, means, normals, covis, gs_aabb_mins, gs_aabb_maxs
    ]):
        raise ValueError("All input tensors must be CUDA tensors.")

    # Contiguous casting ensures memory layout matches C++ pointer expectations
    hit_mask, out_voxel_ids, out_gaus_ids, centroids, densities = query_pgs_voxel_pair_intersection_brute_force(
        vx_aabb_mins.contiguous().to(torch.float32),
        vx_aabb_maxs.contiguous().to(torch.float32),
        means.contiguous().to(torch.float32),
        normals.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        gs_aabb_mins.contiguous().to(torch.float32),
        gs_aabb_maxs.contiguous().to(torch.float32),
        iso,
        return_centroids,
        return_centroid_densities,
        max_capacity
    )

    return hit_mask, out_voxel_ids, out_gaus_ids, centroids, densities

def query_pgs_edge_intersection(
    edge_starts: torch.Tensor,
    edge_ends: torch.Tensor,
    means: torch.Tensor,
    normals: torch.Tensor,
    opacities: torch.Tensor,
    covis: torch.Tensor,
    iso: float = 11.345
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Queries intersections. For each edge, returns the ID of the Gaussian 
    that had the thickest volume overlap.
    
    Args:
        edge_starts (torch.Tensor): (E, 3) tensor of edge start points.
        edge_ends (torch.Tensor): (E, 3) tensor of edge end points.
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        normals (torch.Tensor): (N, 3) tensor of Gaussian principal axes.
        opacities (torch.Tensor): (N,) tensor of Gaussian opacities.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
    
    Returns:
        hit_mask: (E,) boolean mask of edges that hit at least one Gaussian.
        out_gaus_ids: (E,) tensor of the 'best' Gaussian IDs (-1 if no hit).
    """
    if not all(t.is_cuda for t in [edge_starts, edge_ends, means, normals, opacities, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")

    hit_mask, out_gaus_ids = query_pgs_edge_intersection_brute_force(
        edge_starts.contiguous().to(torch.float32),
        edge_ends.contiguous().to(torch.float32),
        means.contiguous().to(torch.float32),
        normals.contiguous().to(torch.float32),
        opacities.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        iso
    )

    return hit_mask, out_gaus_ids