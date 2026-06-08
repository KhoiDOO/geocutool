import torch
from typing import Tuple, Optional

# Explicitly import the compiled CMake target
from .._C import (
    compute_aabb_wrapper, 
    query_gs_voxel_pair_intersection_brute_force_wrapper,
    query_gs_edge_pair_intersection_brute_force_wrapper,
    query_gs_edge_intersection_brute_force_wrapper
)

def compute_gaussian_aabb(
    means: torch.Tensor,
    rotations: torch.Tensor,
    scales: torch.Tensor,
    level: int,
    iso: float = 11.345,
    tol: float = 1. / 8.,
    rotnorm: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the Axis-Aligned Bounding Box (AABB) for 3D Gaussians in world space.

    Args:
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        rotations (torch.Tensor): (N, 4) tensor of quaternions [w, x, y, z].
        scales (torch.Tensor): (N, 3) tensor of scale factors (pre-activation).
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        tol (float, optional): Safety tolerance multiplier. Defaults to 0.125.
        level (int, optional): Octree subdivision level. Resolution will be at :math:`2^level`
        rotnorm (bool, optional): Set to True to force normalizing quaternions.

    Returns:
        aabb_min: torch.Tensor: (N, 3) tensor of AABB minimum corners.
        aabb_max: torch.Tensor: (N, 3) tensor of AABB maximum corners.
        contact_points: torch.Tensor: (N, 3) tensor of contact points on the AABB surface.
        covi: torch.Tensor: (N, 6) tensor of covariance inverses (flattened upper-triangular) for each Gaussian. 
            The 6 values correspond to [Cxx, Cxy, Cxz, Cyy, Cyz, Czz] where C is the covariance matrix of the Gaussian.
    """
    
    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not all(t.is_cuda for t in [means, rotations, scales]):
        raise ValueError("All input tensors must be CUDA tensors.")
        
    means_c = means.contiguous().to(torch.float32)
    rotations_c = rotations.contiguous().to(torch.float32)
    scales_c = scales.contiguous().to(torch.float32)

    # Call the explicit C++ binding
    aabb_min, aabb_max, contact_points, covi = compute_aabb_wrapper(
        means_c,
        rotations_c,
        scales_c,
        iso,
        tol,
        level,
        rotnorm
    )

    return aabb_min, aabb_max, contact_points, covi

def query_gs_voxel_pair_intersection(
    vx_aabb_mins: torch.Tensor,
    vx_aabb_maxs: torch.Tensor,
    means: torch.Tensor,
    covis: torch.Tensor,
    opacities: torch.Tensor,
    gs_aabb_mins: torch.Tensor,
    gs_aabb_maxs: torch.Tensor,
    contact_points: torch.Tensor,
    iso: float = 11.345,
    ar_threshold: float = 0.1,
    p_threshold: float = 0.1,
    return_centroids: bool = False,
    max_capacity: int = 10000000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Queries intersections between Gaussians and voxels using a brute-force approach.

    Args:
        vx_aabb_mins (torch.Tensor): (M, 3) tensor of voxel AABB minimum corners.
        vx_aabb_maxs (torch.Tensor): (M, 3) tensor of voxel AABB maximum corners.
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        opacities (torch.Tensor): (N,) tensor of Gaussian opacities.
        gs_aabb_mins (torch.Tensor): (N, 3) tensor of Gaussian AABB minimum corners.
        gs_aabb_maxs (torch.Tensor): (N, 3) tensor of Gaussian AABB maximum corners.
        contact_points (torch.Tensor): (N, 9) tensor of contact points on the Gaussian surfaces.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        ar_threshold (float, optional): The aspect ratio threshold. Defaults to 0.1.
        p_threshold (float, optional): The penetration threshold. Defaults to 0.1.
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
    if not all(t.is_cuda for t in [vx_aabb_mins, vx_aabb_maxs, means, covis, opacities, gs_aabb_mins, gs_aabb_maxs, contact_points]):
        raise ValueError("All input tensors must be CUDA tensors.")

    # Contiguous casting ensures memory layout matches C++ pointer expectations
    hit_mask, out_voxel_ids, out_gaus_ids, centroids, densities = query_gs_voxel_pair_intersection_brute_force_wrapper(
        vx_aabb_mins.contiguous().to(torch.float32),
        vx_aabb_maxs.contiguous().to(torch.float32),
        means.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        opacities.contiguous().to(torch.float32),
        gs_aabb_mins.contiguous().to(torch.float32),
        gs_aabb_maxs.contiguous().to(torch.float32),
        contact_points.contiguous().to(torch.float32),
        iso,
        ar_threshold,
        p_threshold,
        return_centroids,
        max_capacity
    )

    return hit_mask, out_voxel_ids, out_gaus_ids, centroids, densities

def query_gs_edge_pair_intersection(
    edge_starts: torch.Tensor,
    edge_ends: torch.Tensor,
    means: torch.Tensor,
    covis: torch.Tensor,
    iso: float = 11.345,
    max_capacity: int = 10000000
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Queries intersections between Gaussians and edges using a brute-force approach.

    Args:
        edge_starts (torch.Tensor): (E, 3) tensor of edge start points.
        edge_ends (torch.Tensor): (E, 3) tensor of edge end points.
        means (torch.Tensor): (N, 3) tensor of Gaussian center positions.
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
        max_capacity (int, optional): Maximum number of intersections to store. Defaults to 10000000.

    Returns:
        hit_mask: torch.Tensor: (E,) boolean mask of edges that experienced at least one hit.
        out_edge_ids: torch.Tensor: (K,) tensor of edge IDs with intersections.
        out_gaus_ids: torch.Tensor: (K,) tensor of Gaussian IDs with intersections.
    """

    # Safety Boundary: Ensure memory is perfectly aligned before hitting C++
    if not all(t.is_cuda for t in [edge_starts, edge_ends, means, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")

    # Contiguous casting ensures memory layout matches C++ pointer expectations
    hit_mask, out_edge_ids, out_gaus_ids = query_gs_edge_pair_intersection_brute_force_wrapper(
        edge_starts.contiguous().to(torch.float32),
        edge_ends.contiguous().to(torch.float32),
        means.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        iso,
        max_capacity
    )

    return hit_mask, out_edge_ids, out_gaus_ids

def query_gs_edge_intersection(
    edge_starts: torch.Tensor,
    edge_ends: torch.Tensor,
    means: torch.Tensor,
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
        covis (torch.Tensor): (N, 6) tensor of covariance inverses.
        iso (float, optional): The opacity cutoff threshold. Defaults to 11.345.
    
    Returns:
        hit_mask: (E,) boolean mask of edges that hit at least one Gaussian.
        out_gaus_ids: (E,) tensor of the 'best' Gaussian IDs (-1 if no hit).
    """
    if not all(t.is_cuda for t in [edge_starts, edge_ends, means, covis]):
        raise ValueError("All input tensors must be CUDA tensors.")

    hit_mask, out_gaus_ids = query_gs_edge_intersection_brute_force_wrapper(
        edge_starts.contiguous().to(torch.float32),
        edge_ends.contiguous().to(torch.float32),
        means.contiguous().to(torch.float32),
        covis.contiguous().to(torch.float32),
        iso
    )

    return hit_mask, out_gaus_ids