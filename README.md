<div align="center">

# ⚡ Conquer3D

### *High-Performance GPU-Accelerated Differentiable Geometry, Spatial Computing & Neural Rendering Toolbox*

[![Documentation](https://img.shields.io/badge/📖_Documentation-conquer3d-76B900?style=for-the-badge)](https://khoidoo.github.io/conquer3d/)
[![PyPI Version](https://img.shields.io/pypi/v/conquer3d.svg?color=blue&style=for-the-badge)](https://pypi.org/project/conquer3d/)
[![Docker Image](https://img.shields.io/badge/Docker-kohido%2Fconquer3d-2496ED?logo=docker&logoColor=white&style=for-the-badge)](https://hub.docker.com/r/kohido/conquer3d)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

[![Python Version](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-76B900?logo=nvidia&logoColor=white&style=for-the-badge)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white&style=for-the-badge)](https://pytorch.org/)
[![API Coverage](https://img.shields.io/badge/API_docs-1018_symbols_·_100%25-76B900?style=for-the-badge)](https://khoidoo.github.io/conquer3d/api/index.html)

<p align="center">
  <b><a href="https://khoidoo.github.io/conquer3d/">🌐 Website</a></b> •
  <a href="https://khoidoo.github.io/conquer3d/documentation.html">Guide</a> •
  <a href="https://khoidoo.github.io/conquer3d/api/index.html">API Reference</a> •
  <a href="https://khoidoo.github.io/conquer3d/benchmarks.html">Benchmarks</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-features--highlights">Features</a>
</p>

</div>

---

> [!NOTE]
> The API documentation and the [documentation website](https://khoidoo.github.io/conquer3d/)
> were written with [Claude](https://claude.ai/code). The library itself — every CUDA kernel,
> data structure and operator — is the author's own work.

---

## 🌟 Overview

**Conquer3D** is an ultra-fast, GPU-native computational geometry and differentiable spatial computing library engineered in **PyTorch and CUDA**. Designed from the ground up for 3D computer vision, generative AI, neural surface reconstruction, and differentiable rendering, **Conquer3D** delivers up to **~1.0 Billion faces/second** isosurface extraction, exact CAD sharp crease preservation, and memory-efficient spatial acceleration structures.

Every operator consumes and produces PyTorch tensors **in place** — no host round-trip, no format conversion — so meshing a field is an operation *inside* a training step rather than a preprocessing stage around it.

```bash
pip install -U conquer3d
```

```python
import torch
from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import dmc

grid_vertices, voxels, _ = create_voxel_grid(
    grid_min=[-1.0] * 3, grid_max=[1.0] * 3, res=[64, 64, 64], device="cuda"
)

sdf = (torch.norm(grid_vertices, dim=-1) - 0.6).requires_grad_(True)
verts, faces = dmc(grid_vertices, voxels, sdf, iso=0.0)

verts.sum().backward()          # gradients flow back into the field
```

---

## 📖 Documentation

**Full documentation lives at [khoidoo.github.io/conquer3d](https://khoidoo.github.io/conquer3d/).**

| | |
| :--- | :--- |
| **[Showcase](https://khoidoo.github.io/conquer3d/)** | What the library is for, and how to start. |
| **[Documentation](https://khoidoo.github.io/conquer3d/documentation.html)** | Installation, core concepts, architecture, and worked pipelines. |
| **[API Reference](https://khoidoo.github.io/conquer3d/api/index.html)** | Every symbol, across seven tiers. |
| **[Benchmarks](https://khoidoo.github.io/conquer3d/benchmarks.html)** | Measured extraction throughput on real geometry. |
| **[About](https://khoidoo.github.io/conquer3d/about.html)** | Motivation and the research this builds on. |

The API reference does not stop at the Python surface. It descends through the pybind11 bindings and host dispatchers into the CUDA kernels themselves, the device helpers they call, and the inline math those are built from — **1018 symbols, fully documented**:

| Tier | Layer | Symbols |
| :--- | :--- | ---: |
| **T1** | Python API | 159 |
| **T2** | Native bindings (`conquer3d._C`) | 151 |
| **T3** | Host dispatchers & C++ declarations | 437 |
| **T4** | CUDA `__global__` kernels | 109 |
| **T5** | CUDA `__device__` helpers | 69 |
| **T6** | Inline math primitives | 65 |
| **T7** | `__constant__` topology tables | 28 |

---

## ⚡ Benchmarks

*Empirical extraction performance on **NVIDIA GeForce RTX 4090** on the **Fandisk CAD model** at **$1024^3$ resolution** (6.86M sparse vertices, 5.15M active voxels):*

| Algorithm | Mesh Output | Extracted Vertices | Extracted Faces / Quads | Latency (ms) | Throughput | Watertight |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Dual Marching Cubes (DMC)** | Triangles | **1,716,386** | **3,432,768** | **3.44 ms** | **~1.0B faces/s** | **Yes** |
| **Dual Marching Cubes (DMC)** | Pure Quads | **1,716,386** | **1,716,384** | **3.22 ms** | **533M quads/s** | **Yes** |
| **Dual Contouring (DC + Normals)** | Triangles | 1,716,386 | 3,432,768 | **4.79 ms** | **717M faces/s** | **Yes** |
| **Marching Cubes Asymptotic (MCA)** | Triangles | 1,716,384 | 3,432,764 | 16.42 ms | **209M faces/s** | **Yes** |

---

## ✨ Features & Highlights

<details open>
<summary><b>🔷 1. Isosurface Extraction &amp; Meshing</b></summary>
<br>

- **Dual Marching Cubes (`dmc`)**: High-throughput manifold surface extraction with pure quad $(Q, 4)$ or triangle $(F, 3)$ outputs (~1.0B faces/s).
- **Dual Contouring (`dc`)**: Sharp CAD crease and mechanical corner preservation via GPU Quadratic Error Function (QEF) solving.
- **Marching Cubes Asymptotic (`mca`)**: Decider-enhanced Marching Cubes resolving topological face ambiguities on-the-fly.
- **Differentiable Poisson Surface Reconstruction (`dpsr`, `DPSR`)**: Differentiable spectral Fourier Poisson indicator field solving directly from oriented point clouds.
- **Differentiable Marching Cubes (`diff_marching_cubes`)**: PyTorch autograd gradient backpropagation through levelsets into SDF and color fields.
- **Marching Tetrahedra (`marching_tetrahedra`, `marching_tetrahedra_grid`)**: Consistent isosurface extraction across unstructured meshes and regular cubic tetrahedral grids.

</details>

<details>
<summary><b>🚀 2. Hardware-Accelerated Spatial Acceleration</b></summary>
<br>

- **Triangle Mesh BVH (`MeshBVH`)**: GPU Bounding Volume Hierarchy for fast ray intersections, point projections, and signed distance queries.
- **Gaussian Splatting BVHs (`GSBVH`, `PGSBVH`)**: Spatial hierarchies specialized for 3D Gaussian Splatting and Periodic Gaussian Splatting.
- **Radix Linear BVH (`BVH`)**: High-throughput parallel bounding volume hierarchy for generic 3D primitives.
- **GPU KD-Tree (`KDTree`)**: Parallel nearest-neighbor search with $O(\log N)$ query complexity.
- **Morton Z-Curve Sorting (`z_curve_sort`)**: 64-bit Radix-sorted space-filling curve indexing for maximizing GPU cache locality.
- **Volumetric 3D Flood Fill**: Fast GPU bitmask ray-boundary flood fill for sign evaluation across complex geometries.

</details>

<details>
<summary><b>🌐 3. Volumetric Grids &amp; Surface Conversions</b></summary>
<br>

- **Narrow-Band Sparse Voxel Grids (`create_voxel_grid_from_tmesh`)**: Direct surface-confined voxelization bypassing dense 3D memory.
- **Multi-Mode Normal Field Generation**: On-the-fly extraction of exact CAD face normals (`mode=0`), smooth vertex normals (`mode=1`), or SDF gradients (`mode=2`).
- **Mesh-to-Grid Pipelines (`tmesh2voxel`, `tmesh2sparse`)**: One-line conversion from 3D meshes to dense or sparse Signed Distance Fields.
- **Depth-Map Surface Carving (`get_active_voxel_ids_from_depth`)**: Multi-view RGB-D depth image backprojection into active voxel grids.
- **Sparse & Occupancy Conversions (`sparse_coo2dense_occ`, `dense_occ2sparse_coo`)**: Seamless conversion between sparse coordinate tensors and dense binary occupancy volumes.

</details>

<details>
<summary><b>🎨 4. Radiance Fields &amp; Differentiable Primitives</b></summary>
<br>

- **3D Gaussian Splatting Primitives (`compute_gs_covi`, `compute_gs_aabb`)**: GPU-accelerated covariance matrices, Mahalanobis distances, and tight AABB computation.
- **Periodic Gaussian Splatting (`solve_pgs_cluster_tangency_radius`)**: Spatial frequency-aware radiance field query operators.
- **Geometric Primitive Structures (`Ray`, `Triangle`, `AABB`, `Edge`)**: Vectorized GPU ray tracing and collision primitives.

</details>

<details>
<summary><b>📐 5. Geometric Distances &amp; Volume Integrals</b></summary>
<br>

- **Exact Chamfer Distances (`chamfer_distance`, `one_sided_chamfer_distance`)**: Point-to-point and point-to-mesh geometric error evaluation.
- **Exact Hausdorff Distances (`hausdorff_distance`, `one_sided_hausdorff_distance`)**: Maximum deviation metrics for surface fidelity assessment.
- **Single-View Volume Integrals (`single_view_volume_integral`)**: Analytical divergence-theorem ray volume integration.

</details>

<details>
<summary><b>📦 6. 3D Data Loading, Augmentation &amp; Collation</b></summary>
<br>

- **Benchmark 3D Datasets (`Digit3D`, `PointDigit3D`, `RedWood`, `MeshDataset`)**: Ready-to-use PyTorch dataset abstractions.
- **Standard Benchmark 3D Assets (`conquer3d.data.assets`)**: One-click download & caching for Stanford Bunny, Armadillo, Dragon, Lucy, Happy Buddha, Spot, Cow, Teapot, Suzanne, Fandisk, Iphigenia, and more.
- **Geometric Data Augmentations (`Rotation`, `Scale`, `RandomRotation`, `RandomScale`, `Sequence`, `MeshSequence`)**: Composable spatial transformation pipelines.
- **Custom PyTorch Batch Collation (`bmesh_collate_fn`, `sparse_collate_fn`)**: Efficient handling of variable-sized meshes and sparse tensors in DataLoaders.

</details>

<details>
<summary><b>🛠️ 7. Procedural Generation &amp; File I/O</b></summary>
<br>

- **Procedural Mesh Generators (`create_sphere`, `create_tetrahedra`)**: Parametric geometric shape creation utilities.
- **High-Performance Mesh I/O (`read_obj`, `write_obj`, `read_off`)**: Native loading and saving for vertices, faces, and vertex colors.

</details>

---

## 🚀 Quickstart Examples

### 1. Differentiable Dual Marching Cubes (DMC)
```python
import torch
from conquer3d.data_structure import create_voxel_grid
from conquer3d.ops import dmc

# Create 3D Voxel Grid
grid_vertices, voxels, _ = create_voxel_grid(
    grid_min=[-1.0, -1.0, -1.0], grid_max=[1.0, 1.0, 1.0], res=[64, 64, 64], device="cuda"
)

# Evaluate SDF & Feature Colors with autograd tracking
sdf = (torch.norm(grid_vertices, dim=-1) - 0.6).requires_grad_(True)
colors = torch.rand((grid_vertices.shape[0], 3), device="cuda", requires_grad=True)

# Extract 2-manifold triangle mesh with Newton-Raphson levelset projection
verts, faces, out_colors = dmc(
    grid_vertices, voxels, sdf, colors=colors, iso=0.0, quad_split=True
)

# Extract pure quads (shape: (Q, 4))
quad_verts, quad_faces = dmc(grid_vertices, voxels, sdf, iso=0.0, quad_split=False)

# Backward gradient propagation
loss = verts.sum() + out_colors.sum()
loss.backward()
print(f"SDF Gradient Norm: {sdf.grad.norm().item():.4f}")
```

### 2. Dual Contouring (DC) with Sharp CAD Feature Preservation
```python
import torch
from conquer3d.data.assets import Fandisk
from conquer3d.data_structure import TriangleMesh, create_voxel_grid_from_tmesh
from conquer3d.ops import dc

fandisk = Fandisk()
v, f, _ = fandisk.get()
tmesh = TriangleMesh(v.cuda(), f.cuda().int())

# Build sparse grid and extract exact CAD triangle normals (mode=0)
grid_vertices, voxels, _, grid_normals = create_voxel_grid_from_tmesh(
    grid_min=[-1.0, -1.0, -1.0], grid_max=[1.0, 1.0, 1.0],
    res=[256, 256, 256], tmesh=tmesh, pad=1, return_normals=True, normal_mode=0
)

# Query signed distance field via GPU Flood Fill
tmesh.build_flood_fill_data([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], [256, 256, 256])
_, _, _, sdfs = tmesh.query_points(grid_vertices, return_sdf=True, sign_mode=3)

# Extract sharp mesh using GPU Jacobi SVD Quadratic Error Function solver
verts, faces = dc(grid_vertices, voxels, sdfs, grid_normals=grid_normals, iso=0.0)
print(f"Extracted Sharp CAD Mesh: {verts.shape[0]:,} vertices, {faces.shape[0]:,} faces")
```

### 3. GPU Spatial Queries & Mesh BVH
```python
import torch
from conquer3d.data_structure import TriangleMesh

# Query closest points, projections, and signed distances with Ray Casting / Flood Fill
query_pts = torch.randn((100000, 3), device="cuda")
query_ids, closest_tri_ids, projected_pts, sdfs = tmesh.query_points(
    query_pts, return_sdf=True, return_prj_pts=True, sign_mode=0
)
```

> [!TIP]
> More worked pipelines, the core concepts behind them, and the full operator reference are on the
> **[documentation site](https://khoidoo.github.io/conquer3d/documentation.html)**.

---

## 📦 Installation

### 1. Install via PyPI
```bash
pip install -U conquer3d
```

Prebuilt CUDA wheels are attached to each release for **Python 3.10–3.14** against **PyTorch 2.8 and 2.11** (CUDA 12.8).

### 2. Docker (Recommended for instant GPU / CUDA environment)
```bash
# Pull and run the pre-built image
docker pull kohido/conquer3d:latest
docker run --rm --gpus all -it kohido/conquer3d:latest bash
```

Or build locally:
```bash
docker build -t conquer3d:latest .
docker run --rm --gpus all -it conquer3d:latest bash
```

### 3. Build from Source
Ensure you have a compatible CUDA toolkit ($\ge 12.0$) and PyTorch installed:

```bash
# Optional: Setup Conda environment
conda create -c conda-forge -n conquer3d python=3.10 gxx_linux-64=13 gcc_linux-64=13 sparsehash -y
conda activate conquer3d
conda install nvidia::cuda-toolkit==12.8.2 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install pybind11-stubgen

# Install Conquer3D
git clone https://github.com/KhoiDOO/conquer3d.git
cd conquer3d
pip install -e . --no-build-isolation
```

> [!NOTE]
> Importing `conquer3d` loads the compiled `_C` extension at module scope, so a CUDA-capable device
> and a matching PyTorch build are required.

---

## 📚 Acknowledgements & References

For further theoretical background, GPU collision detection guides, and related literature, please visit:

- **[Research Papers](acknowledgement/REFERENCE.md)**: Computational geometry, differential topology, and acceleration structure foundations.
- **[Blog Posts](acknowledgement/BLOG_POST.md)**: GPU spatial traversal, parallel BVH construction, and CUDA optimization guides.
- **[Related Repositories](acknowledgement/REPOSITORY.md)**: Open-source geometric deep learning ecosystem.
- **[Books](acknowledgement/BOOK.md)**: Core computational geometry references.

A rendered bibliography is also available on the **[About page](https://khoidoo.github.io/conquer3d/about.html)**.

---

## 📝 Citation

```bibtex
@software{conquer3d,
  title   = {Conquer3D: GPU-Accelerated Differentiable Geometry and Spatial Computing},
  author  = {Do, Hoang Khoi},
  year    = {2025},
  url     = {https://github.com/KhoiDOO/conquer3d}
}
```

---

## 📄 License

Conquer3D is licensed under the [MIT License](LICENSE).

<div align="center">
<sub>Built with CUDA and PyTorch · <a href="https://khoidoo.github.io/conquer3d/">khoidoo.github.io/conquer3d</a></sub>
</div>
