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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List

DOCS = Path(__file__).resolve().parent
ROOT = DOCS.parent
sys.path.insert(0, str(DOCS / "_gen"))

import content  # noqa: E402
import pyapi  # noqa: E402
import render  # noqa: E402
import versions  # noqa: E402
from model import TIERS, TIER_ORDER, Group  # noqa: E402

try:
    import nativeapi  # noqa: E402
except Exception:  # pragma: no cover - native tiers are optional
    nativeapi = None


@dataclass(frozen=True)
class ApiLayout:
    """Where one version's API pages live, and how deep they are.

    Every path and every ``../`` in the site is derived from this rather than
    written literally, so the same rendering code emits the flat current
    reference at ``api/`` and a released one at ``api/v0.7.4/`` with no
    special-casing.
    """

    root: str  #: site-root-relative directory, e.g. "api" or "api/v0.7.4"
    version: str
    source_ref: str = "main"  #: git ref the source links should point at
    current: bool = True

    @property
    def depth(self) -> int:
        return len(PurePosixPath(self.root).parts)

    @property
    def base(self) -> str:
        """Prefix from a page in this directory back to the site root."""
        return "../" * self.depth

    def page(self, group: Group) -> str:
        """File name of a group's page, without any directory."""
        return f"{TIERS[group.tier]['slug']}-{group.slug}.html"

    def href(self, group: Group) -> str:
        """Site-root-relative URL of a group's page."""
        return f"{self.root}/{self.page(group)}"

    @property
    def index_href(self) -> str:
        return f"{self.root}/index.html"

    @property
    def search_href(self) -> str:
        return f"{self.root}/search-index.json"


def read_version() -> str:
    return versions.read_version(ROOT)


# --------------------------------------------------------------------------- #
# API index
# --------------------------------------------------------------------------- #


def render_api_index(groups: List[Group], layout: ApiLayout) -> str:
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
                f'<h3><a href="{esc_(layout.page(group))}">{esc_(group.title)}</a></h3>'
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
  <div class="crumbs"><a href="{layout.base}index.html">Home</a> › API Reference</div>
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


def build_search_index(groups: List[Group], layout: ApiLayout) -> List[dict]:
    entries = []
    for group in groups:
        href = layout.href(group)
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
# Shared assets
# --------------------------------------------------------------------------- #

_ASSET_V = re.compile(r"(assets/(?:css|js)/[A-Za-z0-9_.-]+)\?v=[0-9a-f]+")


def relink_assets(out_dir: Path) -> int:
    """Point every page, including frozen ones, at the current shared assets.

    A released version's pages are written once and never rebuilt, so a later
    change to the stylesheet leaves them requesting a hash the browser still
    holds from before that change -- and the page renders with CSS that
    predates it. Content stays frozen; only the references to the shared
    assets are refreshed, which is what keeps every version looking the same.

    Returns:
        int: Number of pages rewritten.
    """
    cache = {}

    def current(match):
        rel = match.group(1)[len("assets/"):]
        if rel not in cache:
            cache[rel] = render._fingerprint(rel)
        return f"{match.group(1)}?v={cache[rel]}"

    touched = 0
    for page in out_dir.rglob("*.html"):
        text = page.read_text(encoding="utf-8")
        updated = _ASSET_V.sub(current, text)
        if updated != text:
            page.write_text(updated, encoding="utf-8")
            touched += 1
    return touched


# --------------------------------------------------------------------------- #
# Link checking
# --------------------------------------------------------------------------- #

_HREF = re.compile(r'(?:href|src)="([^"#][^"]*?)"')


def check_links(out_dir: Path) -> List[str]:
    """Verify every relative href resolves to a file that exists."""
    problems = []
    for page in sorted(out_dir.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        for href in _HREF.findall(text):
            if href.startswith(("http://", "https://", "mailto:", "data:", "//")):
                continue
            # Assets carry a ?v=<hash> cache buster; strip it before
            # resolving, or every stylesheet link reads as dead.
            path = href.split("#")[0].split("?")[0]
            target = (page.parent / path).resolve()
            if not target.exists():
                problems.append(f"{page.relative_to(out_dir)} -> {href}")
    return problems


# --------------------------------------------------------------------------- #
# One version's pages
# --------------------------------------------------------------------------- #


def write_api(groups: List[Group], layout: ApiLayout, manifest: List[dict]) -> None:
    """Emit every API page for one version into its own directory."""
    out_dir = DOCS / layout.root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scoped to this version's directory: a recursive sweep would delete every
    # other version's pages on each build.
    for stale in out_dir.glob("*.html"):
        stale.unlink()

    for group in groups:
        switcher = render.build_switcher(
            manifest, layout.root, layout.page(group), layout.base
        )
        page = render.shell(
            title=f"{group.title} · Conquer3D v{layout.version}",
            description=group.subtitle or f"API reference for {group.title}",
            base=layout.base,
            active="api/index.html",
            version=layout.version,
            search=f"{layout.base}{layout.search_href}",
            sidebar=render.build_sidebar(
                groups, layout.base, group.slug,
                href_for=layout.href, switcher=switcher,
            ),
            toc=render.build_toc(group),
            body=_group_body(group, layout),
        )
        (out_dir / layout.page(group)).write_text(page, encoding="utf-8")

    (out_dir / "index.html").write_text(
        render.shell(
            title=f"API Reference · Conquer3D v{layout.version}",
            description="Complete API reference: Python, pybind11 bindings, CUDA kernels.",
            base=layout.base,
            active="api/index.html",
            version=layout.version,
            search=f"{layout.base}{layout.search_href}",
            body=render_api_index(groups, layout),
            wide=True,
        ),
        encoding="utf-8",
    )

    (out_dir / "search-index.json").write_text(
        json.dumps(build_search_index(groups, layout), separators=(",", ":")),
        encoding="utf-8",
    )


def manifest_entry(layout: ApiLayout, groups: List[Group], tag: str = "") -> dict:
    """One row of api/versions.json, describing a built version."""
    return {
        "version": layout.version,
        "tag": tag,
        "path": layout.root,
        "current": layout.current,
        "documented": sum(g.counts[0] for g in groups),
        "total": sum(g.counts[1] for g in groups),
        # Sorted so a freshly built version and one read back from disk
        # serialise identically; otherwise every rebuild reorders the file.
        "page": dict(sorted((layout.page(g), layout.href(g)) for g in groups)),
    }


def _recorded_counts(path: str) -> tuple:
    """Coverage a previous build recorded for a version directory.

    Re-parsing a frozen release just to count its symbols would mean a Doxygen
    run per version on every build. The numbers were computed when it was built,
    so they are read back from the manifest instead -- otherwise an ordinary
    rebuild rewrites them to zero and dirties the file every time.
    """
    try:
        rows = json.loads((DOCS / "api" / "versions.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0, 0
    for row in rows:
        if row.get("path") == path:
            return row.get("documented", 0), row.get("total", 0)
    return 0, 0


def manifest_from_disk(layout: ApiLayout, tag: str) -> dict:
    """Describe an already-built version without re-parsing its sources.

    A version directory is immutable once written, so its page set is read back
    from the files themselves rather than kept in a side-car that could drift.
    """
    out_dir = DOCS / layout.root
    pages = {
        p.name: f"{layout.root}/{p.name}"
        for p in sorted(out_dir.glob("*.html"))
        if p.name != "index.html"
    }
    documented, total = _recorded_counts(layout.root)
    return {
        "version": layout.version,
        "tag": tag,
        "path": layout.root,
        "current": False,
        "documented": documented,
        "total": total,
        "page": pages,
    }


# --------------------------------------------------------------------------- #
# Releases
# --------------------------------------------------------------------------- #


def parse_source_tree(root: Path, scratch: Path, skip_native: bool):
    """Extract every tier from one source tree.

    Args:
        root: Working tree or a checkout of a tag.
        scratch: Private Doxygen output directory for this tree. Sharing one
            across trees folds a deleted symbol from an earlier tree into a
            later one, because Doxygen leaves stale XML behind.
    """
    groups: List[Group] = []
    groups.extend(pyapi.collect_python(root))
    groups.extend(pyapi.collect_bindings(root))
    if nativeapi is not None and not skip_native:
        groups.extend(nativeapi.collect_native(root, DOCS, scratch))
    return groups


def collect_releases(args) -> List[tuple]:
    """Build, or recognise as already built, every requested release.

    Returns a list of ``(ref, layout, groups)``. ``groups`` is None when the
    directory already existed and was left alone -- a released version cannot
    change, so it is written once and skipped on every later run.
    """
    if args.versions is None:
        return []

    refs = (versions.discover(ROOT) if not args.versions
            else versions.resolve(ROOT, args.versions))
    if not refs:
        print("  ! no release tags found")
        return []

    scratch_root = DOCS / "_build" / "versions"
    scratch_root.mkdir(parents=True, exist_ok=True)

    out = []
    print("-" * 66)
    print(f"releases: {len(refs)} requested")
    for ref in refs:
        layout = ApiLayout(
            root=f"api/{ref.slug}", version=ref.version,
            source_ref=ref.tag, current=False,
        )
        target = DOCS / layout.root
        if target.exists() and not args.refresh:
            print(f"  {ref.tag:<9} skipped (already built)")
            out.append((ref, layout, None))
            continue
        try:
            with versions.worktree(ROOT, ref, parent=scratch_root) as src:
                groups = parse_source_tree(
                    src, scratch_root / ref.slug, args.skip_native
                )
            if not groups:
                print(f"  {ref.tag:<9} ! no symbols parsed -- skipped")
                continue
            documented = sum(g.counts[0] for g in groups)
            total = sum(g.counts[1] for g in groups)
            pct = round(100 * documented / total) if total else 0
            print(f"  {ref.tag:<9} {total:>5} symbols  {pct:>3}% documented")
            out.append((ref, layout, groups))
        except Exception as exc:
            # The layout has changed a great deal across 45 releases; a tag that
            # cannot be parsed is reported and passed over, never fatal.
            print(f"  {ref.tag:<9} ! skipped: {str(exc)[:90]}")
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the Conquer3D documentation site.")
    parser.add_argument("--strict", action="store_true", help="fail on dead links or empty tiers")
    parser.add_argument("--skip-native", action="store_true", help="skip Doxygen-derived tiers")
    parser.add_argument(
        "--versions", nargs="*", metavar="TAG",
        help="also build these releases (tag or bare version); no value means all",
    )
    parser.add_argument(
        "--refresh", action="store_true",
        help="rebuild release directories that already exist",
    )
    args = parser.parse_args()

    version = read_version()
    print(f"Conquer3D docs · v{version}")
    print("-" * 66)

    current_layout = ApiLayout(root="api", version=version, source_ref="main", current=True)

    groups: List[Group] = []
    native_failed = False

    print("T1  parsing conquer3d/**/*.py")
    groups.extend(pyapi.collect_python(ROOT))

    print("T2  parsing conquer3d/_C.pyi")
    groups.extend(pyapi.collect_bindings(ROOT))

    if nativeapi is not None and not args.skip_native:
        print("T3-T7  parsing csrc/ via Doxygen")
        try:
            groups.extend(
                nativeapi.collect_native(ROOT, DOCS, DOCS / "_build" / "doxygen")
            )
        except Exception as exc:
            print(f"    ! native tiers unavailable: {exc}")
            native_failed = True
    elif not args.skip_native:
        print("T3-T7  skipped (nativeapi not present yet)")

    # ------------------------------------------------------------- releases
    releases = collect_releases(args)
    manifest = [manifest_entry(current_layout, groups)]

    # Every release directory present on disk belongs in the manifest, whether
    # or not this run asked for it -- otherwise an ordinary build would drop
    # versions it simply did not mention, and the switcher would lose them.
    seen = {ref.slug for ref, _, _ in releases}
    rows = [
        (ref, layout, manifest_entry(layout, release_groups, ref.tag))
        if release_groups is not None
        else (ref, layout, manifest_from_disk(layout, ref.tag))
        for ref, layout, release_groups in releases
    ]
    for ref in versions.discover(ROOT):
        if ref.slug in seen or not (DOCS / "api" / ref.slug).is_dir():
            continue
        layout = ApiLayout(
            root=f"api/{ref.slug}", version=ref.version,
            source_ref=ref.tag, current=False,
        )
        rows.append((ref, layout, manifest_from_disk(layout, ref.tag)))

    rows.sort(key=lambda row: row[0].key, reverse=True)
    manifest.extend(row[2] for row in rows)

    (DOCS / "api").mkdir(parents=True, exist_ok=True)
    (DOCS / "api" / "versions.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8"
    )

    # ---------------------------------------------------------------- pages
    write_api(groups, current_layout, manifest)
    for ref, layout, release_groups in releases:
        if release_groups is not None:
            write_api(release_groups, layout, manifest)

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

    # The root index serves the non-API pages and points into the current
    # version; each version directory carries its own alongside its pages.
    index = build_search_index(groups, current_layout)
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

    relinked = relink_assets(DOCS)
    if relinked:
        print(f"   relinked shared assets in {relinked} page(s)")

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


def _group_body(group: Group, layout: ApiLayout) -> str:
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
        f'<div class="crumbs"><a href="{layout.base}index.html">Home</a> › '
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
                parts.append(render.render_symbol(child, base=layout.base,
                                              source_ref=layout.source_ref))
        else:
            parts.append(render.render_symbol(symbol, base=layout.base,
                                          source_ref=layout.source_ref))

    return "".join(parts)


if __name__ == "__main__":
    raise SystemExit(main())
