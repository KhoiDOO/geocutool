#!/usr/bin/env python3
"""Build the Conquer3D documentation site.

Usage::

    python3 docs/generate.py [--strict]

Stdlib only -- nothing is imported from ``conquer3d`` itself, because the
package pulls in its compiled CUDA extension at module scope and therefore
cannot be imported without a matching GPU build. Python comes from ``ast``,
native code from Doxygen XML (see ``_gen/nativeapi.py``).

``--strict`` turns coverage regressions and dead internal links into a non-zero
exit status, which is what makes "every symbol is shown" enforceable in CI
rather than merely asserted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_gen"))

import content  # noqa: E402
import pyapi  # noqa: E402
import render  # noqa: E402
from model import TIERS, TIER_ORDER, Group  # noqa: E402

try:
    import nativeapi  # noqa: E402
except Exception:  # pragma: no cover - native tiers are optional
    nativeapi = None


def read_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "0.0.0"


def group_href(group: Group) -> str:
    return f"api/{TIERS[group.tier]['slug']}-{group.slug}.html"


# --------------------------------------------------------------------------- #
# API index
# --------------------------------------------------------------------------- #


def render_api_index(groups: List[Group], version: str) -> str:
    by_tier: Dict[str, List[Group]] = {}
    for group in groups:
        by_tier.setdefault(group.tier, []).append(group)

    blocks = []
    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        meta = TIERS[tier]
        documented = total = 0
        cards = []
        for group in by_tier[tier]:
            d, t = group.counts
            documented += d
            total += t
            pct = round(100 * d / t) if t else 100
            meter = (
                f'<div class="cov" style="margin-top:13px">'
                f'<span>{d}/{t}</span><span class="cov-bar">'
                f'<span class="cov-fill" style="width:{pct}%"></span></span>'
                f"<span>{pct}%</span></div>"
                if t else
                '<div class="cov" style="margin-top:13px"><span>Overview page</span></div>'
            )
            cards.append(
                f'<div class="card"><span class="idx">{esc_(group.source_file)}</span>'
                f'<h3><a href="../{group_href(group)}">{esc_(group.title)}</a></h3>'
                f"<p>{esc_(group.subtitle)}</p>{meter}</div>"
            )
        pct = round(100 * documented / total) if total else 100
        blocks.append(
            f'<section class="section" style="padding:52px 0">'
            f'<div class="section-head" style="margin-bottom:26px">'
            f'<span class="kicker">{tier} · {esc_(meta["name"])}</span>'
            f"<h2>{esc_(meta['blurb'])}</h2>"
            f'<div class="cov" style="margin-top:14px"><span>{documented}/{total} documented</span>'
            f'<span class="cov-bar"><span class="cov-fill" style="width:{pct}%"></span></span>'
            f"<span>{pct}%</span></div></div>"
            f'<div class="grid grid-3">{"".join(cards)}</div></section>'
        )

    total_all = sum(len(g.all_symbols) for g in groups)
    return f"""
<div class="wrap" style="padding-top:52px">
<div class="page-head">
  <div class="crumbs"><a href="../index.html">Home</a> › API Reference</div>
  <h1 style="font-family:var(--sans)">API Reference</h1>
  <p class="sub">{total_all} documented symbols across seven tiers — from the Python surface
  down to individual CUDA kernels, device helpers, and the constant tables that drive
  isosurface extraction.</p>
</div>
</div>
<div class="wrap">{"".join(blocks)}</div>
"""


def esc_(text: str) -> str:
    return render.esc(text)


# --------------------------------------------------------------------------- #
# Search index
# --------------------------------------------------------------------------- #


def build_search_index(groups: List[Group]) -> List[dict]:
    entries = []
    for group in groups:
        href = group_href(group)
        for symbol in group.all_symbols:
            if symbol.kind == "module":
                continue
            entries.append(
                {
                    "n": symbol.name,
                    "q": symbol.qualname or symbol.name,
                    "k": symbol.kind,
                    "t": symbol.tier,
                    "u": f"{href}#{symbol.anchor}",
                    "s": symbol.doc.summary[:150],
                }
            )
    return entries


# --------------------------------------------------------------------------- #
# Link checking
# --------------------------------------------------------------------------- #

_HREF = re.compile(r'href="([^"#][^"]*?)"')


def check_links(out_dir: Path) -> List[str]:
    """Verify every relative href resolves to a file that exists."""
    problems = []
    for page in sorted(out_dir.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        for href in _HREF.findall(text):
            if href.startswith(("http://", "https://", "mailto:", "data:", "//")):
                continue
            target = (page.parent / href.split("#")[0]).resolve()
            if not target.exists():
                problems.append(f"{page.relative_to(out_dir)} -> {href}")
    return problems


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Conquer3D documentation site.")
    parser.add_argument("--strict", action="store_true", help="fail on dead links or empty tiers")
    parser.add_argument("--skip-native", action="store_true", help="skip Doxygen-derived tiers")
    args = parser.parse_args()

    version = read_version()
    print(f"Conquer3D docs · v{version}")
    print("-" * 66)

    groups: List[Group] = []
    native_failed = False

    print("T1  parsing conquer3d/**/*.py")
    groups.extend(pyapi.collect_python(ROOT))

    print("T2  parsing conquer3d/_C.pyi")
    groups.extend(pyapi.collect_bindings(ROOT))

    if nativeapi is not None and not args.skip_native:
        print("T3-T7  parsing csrc/ via Doxygen")
        try:
            groups.extend(nativeapi.collect_native(ROOT, DOCS))
        except Exception as exc:
            print(f"    ! native tiers unavailable: {exc}")
            native_failed = True
    elif not args.skip_native:
        print("T3-T7  skipped (nativeapi not present yet)")

    # ---------------------------------------------------------------- pages
    api_dir = DOCS / "api"
    api_dir.mkdir(parents=True, exist_ok=True)

    # Clear previously emitted pages so a renamed page cannot linger as a stale
    # file that still resolves and silently serves outdated content.
    for stale in api_dir.rglob("*.html"):
        stale.unlink()

    for group in groups:
        page = render.shell(
            title=f"{group.title} · Conquer3D",
            description=group.subtitle or f"API reference for {group.title}",
            base="../",
            active="api/index.html",
            version=version,
            sidebar=render.build_sidebar(groups, "../", group.slug),
            toc=render.build_toc(group),
            body=_group_body(group),
        )
        (DOCS / group_href(group)).write_text(page, encoding="utf-8")

    (api_dir / "index.html").write_text(
        render.shell(
            title="API Reference · Conquer3D",
            description="Complete API reference: Python, pybind11 bindings, CUDA kernels.",
            base="../",
            active="api/index.html",
            version=version,
            body=render_api_index(groups, version),
            wide=True,
        ),
        encoding="utf-8",
    )

    stats = {"total": sum(g.counts[1] for g in groups)}

    tier_stats = []
    for tier in TIER_ORDER:
        tier_groups = [g for g in groups if g.tier == tier]
        if not tier_groups:
            continue
        documented = sum(g.counts[0] for g in tier_groups)
        total = sum(g.counts[1] for g in tier_groups)
        tier_stats.append(
            (tier, TIERS[tier]["name"], documented, total, round(100 * documented / total) if total else 100)
        )

    (DOCS / "benchmarks.html").write_text(
        render.shell(
            title="Benchmarks · Conquer3D",
            description="Measured extraction throughput and the internal architecture of Conquer3D.",
            base="",
            active="benchmarks.html",
            version=version,
            body=content.benchmarks(version, stats),
            wide=True,
        ),
        encoding="utf-8",
    )

    (DOCS / "index.html").write_text(
        render.shell(
            title="Conquer3D · GPU-Accelerated Differentiable Geometry",
            description=(
                "A GPU-native toolbox for isosurface extraction, spatial acceleration, "
                "and differentiable geometry, written in CUDA and exposed as PyTorch tensors."
            ),
            base="",
            active="index.html",
            version=version,
            hero=content.hero(version, stats),
            body=content.showcase(version, stats),
            wide=True,
        ),
        encoding="utf-8",
    )

    (DOCS / "documentation.html").write_text(
        render.shell(
            title="Documentation · Conquer3D",
            description="Installation, core concepts, and worked examples for Conquer3D.",
            base="",
            active="documentation.html",
            version=version,
            body=content.documentation(version, stats, tier_stats),
            wide=True,
        ),
        encoding="utf-8",
    )

    (DOCS / "about.html").write_text(
        render.shell(
            title="About · Conquer3D",
            description="Motivation, architecture, and the research Conquer3D builds on.",
            base="",
            active="about.html",
            version=version,
            body=content.about(version, ROOT / "acknowledgement", stats),
            wide=True,
        ),
        encoding="utf-8",
    )

    index = build_search_index(groups)
    (DOCS / "search-index.json").write_text(
        json.dumps(index, separators=(",", ":")), encoding="utf-8"
    )

    # ---------------------------------------------------------------- report
    print("-" * 66)
    grand_doc = grand_tot = 0
    for tier in TIER_ORDER:
        tier_groups = [g for g in groups if g.tier == tier]
        if not tier_groups:
            continue
        documented = sum(g.counts[0] for g in tier_groups)
        total = sum(g.counts[1] for g in tier_groups)
        grand_doc += documented
        grand_tot += total
        pct = 100 * documented / total if total else 100.0
        bar = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
        print(
            f"{tier} {TIERS[tier]['name']:<20} {documented:>5}/{total:<5} "
            f"[{bar}] {pct:5.1f}%  ({len(tier_groups)} pages)"
        )
    overall = 100 * grand_doc / grand_tot if grand_tot else 100.0
    print("-" * 66)
    print(f"   {'TOTAL':<20} {grand_doc:>5}/{grand_tot:<5} {overall:29.1f}%")
    print(f"   search index: {len(index)} entries")

    problems = check_links(DOCS)
    if problems:
        print(f"\n! {len(problems)} dead internal link(s):")
        for problem in problems[:20]:
            print(f"    {problem}")
    else:
        print("   internal links: all resolve")

    # Coverage gate: this is what makes "every symbol is documented" an
    # enforceable property rather than a claim that quietly rots.
    undocumented = [
        s
        for g in groups
        for s in g.all_symbols
        if s.kind != "module"
        and "stale stub entry" not in s.flags
        and not s.documented
    ]
    if undocumented:
        print(f"\n! {len(undocumented)} undocumented symbol(s):")
        for symbol in undocumented[:20]:
            print(f"    {symbol.tier} {symbol.qualname or symbol.name}"
                  f"  ({symbol.source_file}:{symbol.source_line})")

    # A silently missing Doxygen would drop five whole tiers, so strict mode
    # must treat that as a failure rather than reporting 100% of what remains.
    if args.strict and (problems or undocumented or native_failed):
        return 1
    return 0


def _group_body(group: Group) -> str:
    """Render one API page body."""
    documented, total = group.counts
    pct = round(100 * documented / total) if total else 100
    tier_meta = TIERS[group.tier]

    # A page that only re-exports (the package root) has no symbols of its own,
    # so a coverage ratio would read "0/0" and mean nothing.
    if total:
        coverage = (
            f"<span>{documented}/{total} documented</span>"
            f'<span class="cov-bar"><span class="cov-fill" style="width:{pct}%"></span></span>'
            f"<span>{pct}%</span>"
        )
    else:
        coverage = "<span>Overview page — no symbols of its own</span>"

    head = (
        f'<div class="page-head">'
        f'<div class="crumbs"><a href="../index.html">Home</a> › '
        f'<a href="index.html">API</a> › {esc_(group.tier)}</div>'
        f"<h1>{esc_(group.title)}</h1>"
        f'<p class="sub">{esc_(group.subtitle)}</p>'
        f'<div class="cov" style="margin-top:15px">'
        f'<span class="badge badge-{group.tier.lower()}">{esc_(group.tier)} · {esc_(tier_meta["name"])}</span>'
        f"{coverage}</div></div>"
    )

    if not group.doc.is_empty:
        summary = (
            f'<p class="summary">{render.md_inline(group.doc.summary)}</p>'
            if group.doc.summary
            else ""
        )
        detail = (
            f'<div class="prose">{render.md_block(group.doc.description)}</div>'
            if group.doc.description
            else ""
        )
        head += f'<div class="sym" style="background:transparent;border-style:dashed">' \
                f'<div class="sym-body">{summary}{detail}</div></div>'

    if group.related:
        cards = "".join(
            f'<div class="card"><h3><a href="{href}">{esc_(title)} →</a></h3>'
            f"<p>{count} members</p></div>"
            for title, href, count in group.related
        )
        head += (
            f'<div class="field-head" style="margin-top:22px">Classes in this section</div>'
            f'<div class="grid grid-3" style="margin-bottom:8px">{cards}</div>'
        )

    parts = [head]
    for symbol in group.symbols:
        if symbol.kind == "module" and symbol.children:
            # Module header band, then its members flattened beneath it.
            parts.append(
                f'<div class="module-band" id="{symbol.anchor}">'
                f"<h2>{esc_(symbol.name)}</h2>"
                f'<span class="badge">module</span>'
                f'<span class="src">{esc_(symbol.source_file)}</span></div>'
            )
            if symbol.doc.summary:
                parts.append(f'<p class="prose">{render.md_inline(symbol.doc.summary)}</p>')
            for child in symbol.children:
                parts.append(render.render_symbol(child, base="../"))
        else:
            parts.append(render.render_symbol(symbol, base="../"))

    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
