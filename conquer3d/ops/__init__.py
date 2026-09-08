"""GPU-accelerated geometric operations, distances, and differentiable isosurface extraction.

This package provides:
- Classical and differentiable Marching Cubes (`marching_cubes`, `diff_marching_cubes`).
- Asymptotic decider Marching Cubes (`mca`).
- Differentiable Dual Marching Cubes (`dmc`).
- Dual Contouring with GPU Jacobi QEF solver (`dc`).
- Marching Tetrahedra on arbitrary tets (`marching_tetrahedra`) and cubic grids (`marching_tetrahedra_grid`, `diff_marching_tetrahedra_grid`).
- GPU KD-Tree accelerated Chamfer and Hausdorff distances (`chamfer_distance`, `hausdorff_distance`).
- TSDF RGB-D volume integration (`single_view_volume_integral`).
"""

from .marching_cubes import marching_cubes
from .marching_cubes_asymptotic import marching_cubes_asymptotic, mca
from .dual_contouring import dual_contouring, dc
from .hermite import compute_hermite_from_mesh, compute_hermite_from_field, DC_EDGE_CORNERS
from .dual_marching_cubes import dual_marching_cubes, dmc
from .diff_marching_cubes import diff_marching_cubes

from .delaunay_triangulation import tetrahedralize, get_edges
from .marching_tetrahedra import marching_tetrahedra
from .marching_tetrahedra_grid import marching_tetrahedra_grid
from .diff_marching_tetrahedra_grid import diff_marching_tetrahedra_grid

from .distance import one_sided_chamfer_distance, chamfer_distance, one_sided_hausdorff_distance, hausdorff_distance
from .volint import single_view_volume_integral
from .dpsr import dpsr, DPSR

__all__ = [
    "marching_cubes",
    "marching_cubes_asymptotic",
    "mca",
    "dual_contouring",
    "dc",
    "compute_hermite_from_mesh",
    "compute_hermite_from_field",
    "DC_EDGE_CORNERS",
    "dual_marching_cubes",
    "dmc",
    "diff_marching_cubes",
    "dpsr",
    "DPSR",
    "one_sided_chamfer_distance",
    "chamfer_distance",
    "one_sided_hausdorff_distance",
    "hausdorff_distance",
    "tetrahedralize",
    "get_edges",
    "marching_tetrahedra",
    "marching_tetrahedra_grid",
    "diff_marching_tetrahedra_grid",
    "single_view_volume_integral"
]
