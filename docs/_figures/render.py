"""A small nvdiffrast renderer for the documentation figures.

Deliberately self-contained -- no kaolin, no scene graph. Figures need exactly
one thing: put a mesh on a transparent background, lit so that geometry reads
clearly, at a repeatable camera.

Two shading choices matter for what these figures have to show:

* **Flat shading** uses per-face normals, which is what makes a sharp crease
  look sharp. Smooth shading would average the very discontinuity the Dual
  Contouring figure exists to demonstrate.
* **Transparent background** lets one image sit correctly on both the light and
  dark themes of the site, so no figure needs two variants.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import nvdiffrast.torch as dr
import torch

DEVICE = "cuda"

# Site accent ramp, reused so figures and page agree.
GREEN = (0.463, 0.725, 0.000)
CYAN = (0.133, 0.827, 0.933)
VIOLET = (0.655, 0.545, 0.980)
AMBER = (0.984, 0.749, 0.141)
ROSE = (0.984, 0.443, 0.522)
SLATE = (0.62, 0.66, 0.74)


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #


def look_at(eye, at, up) -> torch.Tensor:
    eye = torch.as_tensor(eye, dtype=torch.float32, device=DEVICE)
    at = torch.as_tensor(at, dtype=torch.float32, device=DEVICE)
    up = torch.as_tensor(up, dtype=torch.float32, device=DEVICE)

    f = torch.nn.functional.normalize(at - eye, dim=0)
    s = torch.nn.functional.normalize(torch.cross(f, up, dim=0), dim=0)
    u = torch.cross(s, f, dim=0)

    m = torch.eye(4, dtype=torch.float32, device=DEVICE)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -torch.stack([torch.dot(s, eye), torch.dot(u, eye), torch.dot(-f, eye)])
    return m


def perspective(fovy_deg: float, aspect: float, near: float, far: float) -> torch.Tensor:
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = torch.zeros(4, 4, dtype=torch.float32, device=DEVICE)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


# --------------------------------------------------------------------------- #
# Mesh preparation
# --------------------------------------------------------------------------- #


def normalize_mesh(vertices: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Centre a mesh on the origin and fit it into a unit-radius sphere."""
    v = vertices.float()
    lo, hi = v.min(0).values, v.max(0).values
    centre = 0.5 * (lo + hi)
    v = v - centre
    radius = v.norm(dim=-1).max().clamp(min=1e-8)
    return v / radius * scale


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    tri = vertices[faces.long()]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    return torch.nn.functional.normalize(n, dim=-1)


def smooth_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    fn = face_normals(vertices, faces)
    vn = torch.zeros_like(vertices)
    for k in range(3):
        vn.index_add_(0, faces.long()[:, k], fn)
    return torch.nn.functional.normalize(vn, dim=-1)


def explode_flat(
    vertices: torch.Tensor, faces: torch.Tensor, colors: Optional[torch.Tensor] = None
):
    """Duplicate vertices per face so each triangle carries its own normal.

    This is what produces genuinely faceted shading; sharing vertices between
    faces would smooth exactly the creases these figures are about.
    """
    idx = faces.long()
    v = vertices[idx].reshape(-1, 3)
    n = face_normals(vertices, faces).repeat_interleave(3, dim=0)
    f = torch.arange(v.shape[0], device=vertices.device, dtype=torch.int32).reshape(-1, 3)
    c = colors[idx].reshape(-1, 3) if colors is not None else None
    return v, f, n, c


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #


class Renderer:
    """Rasterise a mesh to an RGBA image with a transparent background."""

    def __init__(self, size: int = 900, ssaa: int = 2):
        self.ctx = dr.RasterizeCudaContext()
        self.size = size
        self.ssaa = ssaa

    def render(
        self,
        vertices: torch.Tensor,
        faces: torch.Tensor,
        *,
        colors: Optional[torch.Tensor] = None,
        base=SLATE,
        flat: bool = True,
        azimuth: float = 35.0,
        elevation: float = 22.0,
        fov: float = 32.0,
        wireframe: float = 0.0,
        rim=CYAN,
        rim_strength: float = 0.55,
        target: Optional[Tuple[float, float, float]] = None,
        fit_radius: float = 1.0,
        margin: float = 1.12,
    ) -> np.ndarray:
        res = self.size * self.ssaa
        faces = faces.int().contiguous()

        if flat:
            v, f, n, c = explode_flat(vertices, faces, colors)
        else:
            v, f = vertices, faces
            n = smooth_normals(vertices, faces)
            c = colors

        # --- camera ------------------------------------------------------
        # Distance is derived from the sphere we want to frame rather than set
        # by hand, so a full view and a close crop use the same code path and
        # neither can silently clip.
        a = math.radians(azimuth)
        e = math.radians(elevation)
        ctr = torch.zeros(3, device=DEVICE) if target is None else torch.tensor(
            target, dtype=torch.float32, device=DEVICE
        )
        dist = margin * fit_radius / math.sin(math.radians(fov) / 2.0)
        eye = ctr + dist * torch.tensor(
            [math.cos(e) * math.sin(a), math.sin(e), math.cos(e) * math.cos(a)],
            dtype=torch.float32, device=DEVICE,
        )
        target = ctr
        view = look_at(eye, target, (0.0, 1.0, 0.0))
        proj = perspective(fov, 1.0, 0.01, 100.0)
        # nvdiffrast expects a flipped Y relative to the usual GL convention.
        proj = proj.clone()
        proj[1, 1] = -proj[1, 1]
        mvp = proj @ view

        homo = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
        clip = (mvp @ homo.T).T.contiguous().unsqueeze(0)

        rast, _ = dr.rasterize(self.ctx, clip, f, resolution=[res, res])
        mask = (rast[..., 3:] > 0).float()

        nrm = dr.interpolate(n.contiguous().unsqueeze(0), rast, f)[0]
        nrm = torch.nn.functional.normalize(nrm, dim=-1)

        if c is not None:
            albedo = dr.interpolate(c.contiguous().unsqueeze(0), rast, f)[0]
        else:
            albedo = torch.tensor(base, device=DEVICE).view(1, 1, 1, 3).expand_as(nrm)

        # --- shading -----------------------------------------------------
        view_dir = torch.nn.functional.normalize(eye - target, dim=0)
        key = torch.nn.functional.normalize(
            torch.tensor([0.55, 0.72, 0.42], device=DEVICE), dim=0
        )
        fill = torch.nn.functional.normalize(
            torch.tensor([-0.6, 0.15, 0.5], device=DEVICE), dim=0
        )

        # Dual methods split quads into triangles with mixed winding, which under
        # one-sided shading paints scattered bright slivers that read as holes in
        # an otherwise closed surface. Flipping normals towards the camera makes
        # shading two-sided so the figure shows geometry, not winding.
        facing = torch.where((nrm * view_dir).sum(-1, keepdim=True) < 0.0, -1.0, 1.0)
        nrm = nrm * facing

        ndl = (nrm * key).sum(-1, keepdim=True)
        # Half-Lambert keeps the unlit side readable instead of crushing to black.
        diff = (0.5 * ndl + 0.5).clamp(0, 1) ** 1.6
        ndf = (nrm * fill).sum(-1, keepdim=True).clamp(0, 1)

        ndv = (nrm * view_dir).sum(-1, keepdim=True).abs()
        rim_term = (1.0 - ndv).clamp(0, 1) ** 3.0
        rim_col = torch.tensor(rim, device=DEVICE).view(1, 1, 1, 3)

        col = albedo * (0.22 + 0.85 * diff + 0.25 * ndf)
        col = col + rim_col * rim_term * rim_strength

        if wireframe > 0.0:
            bary = rast[..., :2]
            third = 1.0 - bary[..., :1] - bary[..., 1:2]
            edge = torch.cat([bary, third], dim=-1).min(-1, keepdim=True).values
            line = (edge < wireframe).float()
            col = torch.lerp(col, col * 0.25, line * 0.9)

        col = col.clamp(0, 1) ** (1 / 2.2)
        img = torch.cat([col, mask], dim=-1)
        img = dr.antialias(img.contiguous(), rast, clip, f)

        out = (img[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
        if self.ssaa > 1:
            from PIL import Image

            out = np.asarray(
                Image.fromarray(out, "RGBA").resize(
                    (self.size, self.size), Image.LANCZOS
                )
            )
        return out
