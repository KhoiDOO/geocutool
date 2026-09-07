r"""GPU-accelerated spatial indexing data structures and geometric representations.

This package exposes:
- KD-Tree for $O(\log N)$ nearest neighbor queries (`KDTree`).
- Radix Linear Bounding Volume Hierarchy for fast spatial queries (`BVH`).
- Triangle Mesh BVH with Fast Winding Number (FWN) and exact SDF (`MeshBVH`).
- 3D Gaussian and Periodic Gaussian BVHs (`GSBVH`, `PGSBVH`).
- Half-edge Discrete Differential Geometry Triangle Mesh container (`TriangleMesh`).
- Voxel grid generators and active surface extractors (`create_voxel_grid`, etc.).
- Morton space-filling Z-curve sorting (`z_curve_sort`).
"""

from .._C import (
    KDTree,
    BVH,
    GSBVH,
    PGSBVH,
    MeshBVH,
    TriangleMesh
)
from .grid import (
    create_voxel_grid, 
    compute_grid_normal, 
    compute_active_voxels, 
    create_voxel_grid_from_tmesh,
    get_active_voxel_ids_from_depth,
    build_sparse_grid_from_active_voxels
)
from .sort import z_curve_sort

spatial_data_structures = ['KDTree', 'BVH', 'GSBVH', 'PGSBVH', 'MeshBVH']
mesh_data_structures = ['TriangleMesh']
grid_data_structures = [
    'create_voxel_grid', 
    'compute_grid_normal', 
    'compute_active_voxels', 
    'create_voxel_grid_from_tmesh',
    'get_active_voxel_ids_from_depth',
    'build_sparse_grid_from_active_voxels',
]
sort_data_structures = ['z_curve_sort']

__all__ = spatial_data_structures + mesh_data_structures + grid_data_structures + sort_data_structures
