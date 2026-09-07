"""Panel composition for the documentation figures.

Every figure is a grid of rendered panels with a caption strip. Keeping the
composition here means the individual figure builders only have to produce
images, and the whole set stays visually consistent.

Output is RGBA with a transparent background so a single file works on both the
light and dark themes of the site.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(
    "/home/koi/anaconda3/envs/geocutool/lib/python3.10/site-packages/"
    "matplotlib/mpl-data/fonts/ttf"
)

# Mid-tone text works on both themes without a per-theme variant.
INK = (139, 147, 167, 255)
INK_STRONG = (123, 132, 150, 255)
ACCENT = (118, 185, 0, 255)


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except Exception:
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    return int(draw.textbbox((0, 0), text, font=font)[2])


def trim(panels: Sequence[np.ndarray], margin: int = 12) -> List[np.ndarray]:
    """Crop panels to their shared content box.

    The *union* box is used rather than each panel's own, so every panel keeps
    the same scale and centre -- essential when the figure exists to compare
    them. Only genuinely empty margins are removed.
    """
    boxes = []
    for panel in panels:
        ys, xs = np.nonzero(panel[..., 3])
        if len(xs):
            boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    if not boxes:
        return list(panels)
    x0 = max(0, min(b[0] for b in boxes) - margin)
    y0 = max(0, min(b[1] for b in boxes) - margin)
    x1 = min(panels[0].shape[1], max(b[2] for b in boxes) + margin)
    y1 = min(panels[0].shape[0], max(b[3] for b in boxes) + margin)
    return [p[y0:y1, x0:x1] for p in panels]


def grid(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    sublabels: Optional[Sequence[str]] = None,
    cols: Optional[int] = None,
    pad: int = 18,
    label_h: int = 78,
    accents: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """Arrange rendered panels into a labelled grid."""
    n = len(panels)
    cols = cols or n
    rows = (n + cols - 1) // cols

    ph, pw = panels[0].shape[0], panels[0].shape[1]
    W = cols * pw + (cols + 1) * pad
    H = rows * (ph + label_h) + (rows + 1) * pad

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    f_label = _font(30, bold=True)
    f_sub = _font(22)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (pw + pad)
        y = pad + r * (ph + label_h + pad)
        canvas.alpha_composite(Image.fromarray(panel, "RGBA"), (x, y))

        # Accent rule above each caption ties the panel to its label.
        accent = tuple(accents[i]) + (255,) if accents else ACCENT
        ty = y + ph + 12
        draw.line([(x, ty), (x + 34, ty)], fill=accent, width=3)

        draw.text((x, ty + 9), labels[i], font=f_label, fill=INK_STRONG)
        if sublabels and sublabels[i]:
            w = _text_width(draw, labels[i], f_label)
            sw = _text_width(draw, sublabels[i], f_sub)
            # Trimmed panels can be narrower than label + sublabel side by side,
            # which would run the text into the neighbouring panel.
            if w + 12 + sw <= pw:
                draw.text((x + w + 12, ty + 12), sublabels[i], font=f_sub, fill=INK)
            else:
                draw.text((x, ty + 44), sublabels[i], font=f_sub, fill=INK)

    return canvas


def stack(images: Sequence[Image.Image], pad: int = 10) -> Image.Image:
    """Stack composed rows vertically, centred."""
    W = max(im.width for im in images)
    H = sum(im.height for im in images) + pad * (len(images) - 1)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    y = 0
    for im in images:
        out.alpha_composite(im, ((W - im.width) // 2, y))
        y += im.height + pad
    return out


def save(image: Image.Image, path: Path, max_width: int = 2000) -> None:
    """Write a figure, downscaling and stripping unused alpha bounds."""
    path.parent.mkdir(parents=True, exist_ok=True)
    bbox = image.getbbox()
    if bbox:
        m = 14
        image = image.crop((max(0, bbox[0] - m), max(0, bbox[1] - m),
                            min(image.width, bbox[2] + m),
                            min(image.height, bbox[3] + m)))
    if image.width > max_width:
        h = round(image.height * max_width / image.width)
        image = image.resize((max_width, h), Image.LANCZOS)
    # WebP with alpha is roughly 4x smaller than PNG for these renders at
    # visually indistinguishable quality, which keeps the gallery light enough
    # to commit.
    path = path.with_suffix(".webp")
    image.save(path, "WEBP", quality=86, method=6, lossless=False)
    kb = path.stat().st_size / 1024
    print(f"    wrote {path.name}  {image.width}x{image.height}  {kb:.0f} KB")


def colormap(values: np.ndarray, name: str = "viridis", robust: float = 2.0) -> np.ndarray:
    """Map a scalar field to RGB in [0, 1], clipping outliers.

    Curvature and quality fields have heavy tails -- a couple of degenerate
    triangles would otherwise flatten the entire colour range, so percentile
    clipping is applied before normalising.
    """
    import matplotlib.cm as cm

    v = np.asarray(values, dtype=np.float64).ravel()
    finite = v[np.isfinite(v)]
    lo, hi = np.percentile(finite, [robust, 100 - robust])
    if hi - lo < 1e-12:
        lo, hi = finite.min(), finite.max() + 1e-9
    t = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return cm.get_cmap(name)(t)[:, :3]
