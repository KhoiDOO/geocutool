#include <torch/extension.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

/**
 * @brief Registers 3D Gaussian splatting operators (covariance, neighbour radii, AABBs) on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_gs(py::module_& m);
/**
 * @brief Registers periodic Gaussian splatting operators on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_pgs(py::module_& m);
/**
 * @brief Registers the ::Triangle geometric primitive on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_triangle(py::module_& m);
/**
 * @brief Registers the ::Ray geometric primitive on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_primitive_ray(py::module_& m);

/**
 * @brief Registers the ::KDTree spatial acceleration structure on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_kdtree(py::module_& m);
/**
 * @brief Registers the ::BVH linear bounding volume hierarchy on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_bvh(py::module_& m);
/**
 * @brief Registers the ::GSBVH Gaussian splatting hierarchy on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_gs_bvh(py::module_& m);
/**
 * @brief Registers the ::PGSBVH periodic Gaussian splatting hierarchy on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_pgs_bvh(py::module_& m);
/**
 * @brief Registers the ::MeshBVH triangle mesh hierarchy on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_mesh_bvh(py::module_& m);
/**
 * @brief Registers the ::TriangleMesh half-edge mesh structure on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_triangle_mesh(py::module_& m);
/**
 * @brief Registers voxel grid construction and depth-map carving functions on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_grid(py::module_& m);
/**
 * @brief Registers Morton Z-curve code computation on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ds_zcurve(py::module_& m);

/**
 * @brief Registers procedural mesh generators on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_creation_triangle_creation(py::module_& m);

/**
 * @brief Registers the Marching Cubes operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mc(py::module_& m);
/**
 * @brief Registers the Marching Cubes Asymptotic operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mca(py::module_& m);
/**
 * @brief Registers the Dual Contouring operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_dc(py::module_& m);
/**
 * @brief Registers the Dual Marching Cubes operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_dmc(py::module_& m);
/**
 * @brief Registers the Marching Tetrahedra operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mt(py::module_& m);
/**
 * @brief Registers the grid-based Marching Tetrahedra operator and its backward pass on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_mtg(py::module_& m);
/**
 * @brief Registers the one-sided Chamfer distance operator on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_chamfer(py::module_& m);
/**
 * @brief Registers the volumetric flood fill operator on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_flood_fill(py::module_& m);
/**
 * @brief Registers the coarse-fine volumetric flood fill operator on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_flood_fill_cf(py::module_& m);
/**
 * @brief Registers the single-view volume integration operator on the extension module.
 * @param[in,out] m The `conquer3d._C` module object.
 */
void bind_ops_volint(py::module_& m);

/**
 * @brief Entry point assembling the `conquer3d._C` extension module.
 * @details Every `bind_*` function receives the *same* root module object, so the compiled
 * extension presents one flat namespace rather than nested submodules. The call order is
 * grouped by concern -- primitives, then data structures, then creation, then operators --
 * which is presentational only and carries no dependency between the groups.
 */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Conquer3D Python bindings";

    bind_primitive_gs(m);
    bind_primitive_pgs(m);
    bind_primitive_triangle(m);
    bind_primitive_ray(m);

    bind_ds_kdtree(m);
    bind_ds_bvh(m);
    bind_ds_gs_bvh(m);
    bind_ds_pgs_bvh(m);
    bind_ds_mesh_bvh(m);
    bind_ds_triangle_mesh(m);
    bind_ds_grid(m);
    bind_ds_zcurve(m);
    
    bind_creation_triangle_creation(m);
    
    bind_ops_mc(m);
    bind_ops_mca(m);
    bind_ops_dc(m);
    bind_ops_dmc(m);
    bind_ops_mt(m);
    bind_ops_mtg(m);
    bind_ops_chamfer(m);
    bind_ops_flood_fill(m);
    bind_ops_flood_fill_cf(m);
    bind_ops_volint(m);
}