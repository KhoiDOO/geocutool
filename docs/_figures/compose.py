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


def pad_to(panels, width: int):
    """Centre each panel in a transparent frame of the given width.

    Two grids stacked into one figure only read as columns if their cells are
    the same width; a wide shot and a close crop trim to different aspects, so
    the narrower row is padded rather than rescaled, which would break the
    shared scale that makes the panels comparable.
    """
    out = []
    for panel in panels:
        h, w = panel.shape[0], panel.shape[1]
        if w >= width:
            out.append(panel)
            continue
        frame = np.zeros((h, width, panel.shape[2]), dtype=panel.dtype)
        x = (width - w) // 2
        frame[:, x:x + w] = panel
        out.append(frame)
    return out


_CAP_TOP = 12   #: accent rule to label baseline
_CAP_GAP = 8    #: label to a sublabel placed on its own line


def _caption_height(draw, labels, sublabels, pw, f_label, f_sub) -> int:
    """Height needed by the tallest caption in the set.

    A sublabel may sit beside its label, drop to a line of its own, or carry
    several lines. A fixed band silently clips the last line as soon as the
    type or the text grows, so it is measured instead of assumed.
    """
    lh = f_label.getbbox("Ag")[3]
    sh = f_sub.getbbox("Ag")[3]
    tallest = 0
    for i, label in enumerate(labels):
        sub = sublabels[i] if sublabels and i < len(sublabels) else ""
        lines = sub.split("\n") if sub else []
        inline = (
            len(lines) == 1
            and _text_width(draw, label, f_label) + 16
            + _text_width(draw, lines[0], f_sub) <= pw
        )
        height = _CAP_TOP + lh
        if lines and not inline:
            height += _CAP_GAP + len(lines) * (sh + 8)
        tallest = max(tallest, height + 16)
    return tallest


def grid(
    panels: Sequence[np.ndarray],
    labels: Sequence[str],
    *,
    sublabels: Optional[Sequence[str]] = None,
    cols: Optional[int] = None,
    pad: int = 18,
    label_h: int = 104,
    accents: Optional[Sequence[Tuple[int, int, int]]] = None,
) -> Image.Image:
    """Arrange rendered panels into a labelled grid."""
    n = len(panels)
    cols = cols or n
    rows = (n + cols - 1) // cols

    ph, pw = panels[0].shape[0], panels[0].shape[1]
    f_label = _font(42, bold=True)
    f_sub = _font(32)

    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    label_h = max(label_h, _caption_height(probe, labels, sublabels, pw, f_label, f_sub))

    W = cols * pw + (cols + 1) * pad
    H = rows * (ph + label_h) + (rows + 1) * pad

    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        x = pad + c * (pw + pad)
        y = pad + r * (ph + label_h + pad)
        canvas.alpha_composite(Image.fromarray(panel, "RGBA"), (x, y))

        # Accent rule above each caption ties the panel to its label.
        accent = tuple(accents[i]) + (255,) if accents else ACCENT
        ty = y + ph + 16
        draw.line([(x, ty), (x + 46, ty)], fill=accent, width=4)

        # A narrow panel -- a tall subject trims to one -- cannot hold a long
        # label at full size, and an overrun writes straight over its
        # neighbour's caption.
        f_lab = f_label
        size = 42
        while size > 22 and _text_width(draw, labels[i], f_lab) > pw:
            size -= 2
            f_lab = _font(size, bold=True)
        label_w = _text_width(draw, labels[i], f_lab)
        draw.text((x, ty + _CAP_TOP), labels[i], font=f_lab, fill=INK_STRONG)

        sub = sublabels[i] if sublabels and i < len(sublabels) else ""
        if sub:
            lines = sub.split("\n")
            # Trimmed panels can be narrower than label + sublabel side by side,
            # which would run the text into the neighbouring panel.
            inline = (
                len(lines) == 1
                and label_w + 16 + _text_width(draw, lines[0], f_sub) <= pw
            )
            if inline:
                draw.text((x + label_w + 16, ty + _CAP_TOP + 8), sub,
                          font=f_sub, fill=INK)
            else:
                # On its own line it can still overrun a narrow panel, so step
                # the size down until it fits rather than bleeding into the
                # neighbour.
                font, size = f_sub, 32
                while size > 18 and max(
                    _text_width(draw, line, font) for line in lines
                ) > pw:
                    size -= 2
                    font = _font(size)
                y0 = ty + _CAP_TOP + f_lab.getbbox("Ag")[3] + _CAP_GAP
                step = font.getbbox("Ag")[3] + 8
                for k, line in enumerate(lines):
                    draw.text((x, y0 + k * step), line, font=font, fill=INK)

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


def hstack(images: Sequence[Image.Image], pad: int = 18) -> Image.Image:
    """Place composed blocks side by side, centred vertically.

    The counterpart to :func:`stack`, for a figure where one panel stands apart
    from a grid -- a reference the whole grid is read against.
    """
    H = max(im.height for im in images)
    W = sum(im.width for im in images) + pad * (len(images) - 1)
    out = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    x = 0
    for im in images:
        out.alpha_composite(im, (x, (H - im.height) // 2))
        x += im.width + pad
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
