"""HTML rendering for the Conquer3D documentation site.

Everything emitted here is plain static HTML: no client-side framework, no build
step, servable directly by GitHub Pages from ``docs/``. The only external assets
are KaTeX (the docstrings are full of ``$...$``) and a webfont, both pinned.

The renderer deliberately knows nothing about where a symbol came from -- it
consumes :mod:`model` objects only, which is what lets a CUDA kernel and a
Python function render with identical grammar.
"""

from __future__ import annotations

import html
import re
from typing import Dict, Iterable, List, Optional

from model import TIERS, TIER_ORDER, Doc, Group, Symbol

KATEX = "https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9"
FONTS = (
    "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700"
    "&family=JetBrains+Mono:wght@400;500;600&display=swap"
)
REPO = "https://github.com/KhoiDOO/conquer3d"

TABS = [
    ("index.html", "Showcase"),
    ("documentation.html", "Documentation"),
    ("api/index.html", "API Reference"),
    ("benchmarks.html", "Benchmarks"),
    ("about.html", "About"),
]


# --------------------------------------------------------------------------- #
# Inline text
# --------------------------------------------------------------------------- #

_CODE_SPAN = re.compile(r"``([^`]+)``|`([^`]+)`")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_ITALIC = re.compile(r"(?<![*\w])\*([^*\n]+)\*(?!\*)")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BARE_URL = re.compile(r"(?<![\"'=(])\bhttps?://[^\s<>)\"]+")
_MATH = re.compile(r"\$\$(.+?)\$\$|\$([^$\n]+?)\$", re.DOTALL)


def esc(text: str) -> str:
    return html.escape(str(text), quote=True)


def md_inline(text: str) -> str:
    """Render inline markdown, protecting code spans and LaTeX from each other.

    Math is left as literal ``$...$`` for KaTeX auto-render to typeset in the
    browser; it is only shielded from the markdown passes so that a ``\\times``
    or ``_i`` is not mangled into emphasis.
    """
    if not text:
        return ""

    shields: List[str] = []

    def shield(payload: str) -> str:
        shields.append(payload)
        return f"\x00{len(shields) - 1}\x00"

    def take_math(match: re.Match) -> str:
        block, inline = match.group(1), match.group(2)
        body = block if block is not None else inline
        wrapper = "$$%s$$" if block is not None else "$%s$"
        return shield(wrapper % esc(body))

    def take_code(match: re.Match) -> str:
        body = match.group(1) or match.group(2) or ""
        return shield(f"<code>{esc(body)}</code>")

    text = _MATH.sub(take_math, text)
    text = _CODE_SPAN.sub(take_code, text)

    out = esc(text)
    out = _LINK.sub(lambda m: f'<a href="{esc(m.group(2))}">{esc(m.group(1))}</a>', out)
    out = _BARE_URL.sub(lambda m: f'<a href="{m.group(0)}">{m.group(0)}</a>', out)
    out = _BOLD.sub(r"<strong>\1</strong>", out)
    out = _ITALIC.sub(r"<em>\1</em>", out)

    for i, payload in enumerate(shields):
        out = out.replace(f"\x00{i}\x00", payload)
    return out


# --------------------------------------------------------------------------- #
# Syntax highlighting
# --------------------------------------------------------------------------- #

_PY_KW = set("""and as assert async await break class continue def del elif else except finally
for from global if import in is lambda nonlocal not or pass raise return try while with yield
None True False self cls""".split())

_CPP_KW = set("""alignas alignof auto bool break case catch char class const constexpr continue
default delete do double else enum explicit extern false float for friend goto if inline int long
namespace new nullptr operator private protected public register return short signed sizeof static
struct switch template this throw true try typedef typename union unsigned using virtual void
volatile while __global__ __device__ __host__ __shared__ __constant__ __restrict__ __forceinline__
uint32_t int32_t uint8_t int8_t size_t float3 float4 float2 dim3 size_t""".split())

_TOKEN = re.compile(
    r"(?P<com>#[^\n]*|//[^\n]*|/\*.*?\*/)"
    r"|(?P<str>\"\"\".*?\"\"\"|'''.*?'''|\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
    r"|(?P<pre>^\s*(?:>>>|\.\.\.))"
    r"|(?P<num>\b\d+\.?\d*(?:[eE][-+]?\d+)?[fFuUlL]*\b)"
    r"|(?P<word>[A-Za-z_]\w*)"
    r"|(?P<call>\()",
    re.DOTALL | re.MULTILINE,
)


def highlight(code: str, lang: str = "python") -> str:
    """Tokenise a code block into span-wrapped HTML."""
    keywords = _CPP_KW if lang in ("cpp", "cuda", "c") else _PY_KW
    out: List[str] = []
    pos = 0

    for match in _TOKEN.finditer(code):
        out.append(esc(code[pos : match.start()]))
        pos = match.end()
        text = match.group(0)

        if match.lastgroup == "com":
            out.append(f'<span class="tok-com">{esc(text)}</span>')
        elif match.lastgroup == "str":
            out.append(f'<span class="tok-str">{esc(text)}</span>')
        elif match.lastgroup == "pre":
            out.append(f'<span class="tok-pre">{esc(text)}</span>')
        elif match.lastgroup == "num":
            out.append(f'<span class="tok-num">{esc(text)}</span>')
        elif match.lastgroup == "word":
            if text in keywords:
                out.append(f'<span class="tok-kw">{esc(text)}</span>')
            elif code[match.end() : match.end() + 1] == "(":
                out.append(f'<span class="tok-fn">{esc(text)}</span>')
            else:
                out.append(esc(text))
        else:
            out.append(esc(text))

    out.append(esc(code[pos:]))
    return "".join(out)


_FENCE = re.compile(r"^```(\w*)\s*$")


def md_block(text: str, lang: str = "python") -> str:
    """Render a block of docstring prose: paragraphs, fences, and lists."""
    if not text or not text.strip():
        return ""

    lines = text.split("\n")
    out: List[str] = []
    buffer: List[str] = []
    listing: List[str] = []
    fence: Optional[List[str]] = None
    fence_lang = lang

    def flush_para():
        if buffer:
            out.append(f'<p>{md_inline(" ".join(l.strip() for l in buffer))}</p>')
            buffer.clear()

    def flush_list():
        if listing:
            items = "".join(f"<li>{md_inline(i)}</li>" for i in listing)
            out.append(f"<ul>{items}</ul>")
            listing.clear()

    for line in lines:
        fence_match = _FENCE.match(line.strip())
        if fence_match:
            if fence is None:
                flush_para(); flush_list()
                fence = []
                fence_lang = fence_match.group(1) or lang
            else:
                out.append(f'<pre><code>{highlight(chr(10).join(fence), fence_lang)}</code></pre>')
                fence = None
            continue

        if fence is not None:
            fence.append(line)
            continue

        stripped = line.strip()
        if not stripped:
            flush_para(); flush_list()
        elif stripped.startswith((">>> ", "... ")) or (buffer and buffer[-1].strip().startswith(">>>")):
            # A doctest run inside prose becomes its own code block.
            flush_para(); flush_list()
            out.append(f'<pre><code>{highlight(line.strip(), "python")}</code></pre>')
        elif re.match(r"^[-*+]\s+", stripped):
            flush_para()
            listing.append(re.sub(r"^[-*+]\s+", "", stripped))
        else:
            flush_list()
            buffer.append(line)

    if fence is not None:
        out.append(f'<pre><code>{highlight(chr(10).join(fence), fence_lang)}</code></pre>')
    flush_para(); flush_list()
    return "".join(out)


def code_block(text: str, lang: str = "python") -> str:
    return f'<pre><code>{highlight(text, lang)}</code></pre>'


# --------------------------------------------------------------------------- #
# Page shell
# --------------------------------------------------------------------------- #


def shell(
    *,
    title: str,
    description: str,
    base: str,
    active: str,
    body: str,
    version: str,
    sidebar: str = "",
    toc: str = "",
    hero: str = "",
    wide: bool = False,
) -> str:
    """Wrap page content in the site chrome."""
    tabs = "".join(
        f'<a href="{base}{href}" class="{"active" if href == active else ""}">{esc(label)}</a>'
        for href, label in TABS
    )

    if sidebar or toc:
        main = (
            f'<div class="layout">'
            f'<aside class="sidebar">{sidebar}</aside>'
            f'<main class="content">{body}</main>'
            f'<aside class="toc">{toc}</aside>'
            f"</div>"
        )
    elif wide:
        main = f"<main>{body}</main>"
    else:
        main = f'<main class="wrap">{body}</main>'

    return f"""<!doctype html>
<html lang="en" data-base="{base}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='7' fill='%2376b900'/></svg>">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<link rel="stylesheet" href="{KATEX}/katex.min.css">
<link rel="stylesheet" href="{base}assets/css/site.css">
<script>
  // Applied before first paint so the theme never flashes.
  (function(){{
    try {{
      var s = localStorage.getItem('c3d-theme');
      if (!s) s = matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', s);
    }} catch(e) {{ document.documentElement.setAttribute('data-theme','dark'); }}
  }})();
</script>
</head>
<body>
<header class="nav">
  <a class="brand" href="{base}index.html">
    <span class="mark"></span> Conquer3D <span class="ver">v{esc(version)}</span>
  </a>
  <nav class="nav-tabs">{tabs}</nav>
  <div class="nav-right">
    <button class="search-trigger" id="search-trigger" aria-label="Search documentation">
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="11" cy="11" r="7"/><path d="m21 21-4.3-4.3"/></svg>
      <span>Search</span><kbd>⌘K</kbd>
    </button>
    <button class="icon-btn" id="theme-toggle" aria-label="Toggle colour theme">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>
    </button>
    <a class="icon-btn" href="{REPO}" aria-label="GitHub repository">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 .5C5.7.5.5 5.7.5 12c0 5.1 3.3 9.4 7.9 10.9.6.1.8-.2.8-.6v-2c-3.2.7-3.9-1.5-3.9-1.5-.5-1.3-1.3-1.7-1.3-1.7-1-.7.1-.7.1-.7 1.2.1 1.8 1.2 1.8 1.2 1 1.8 2.7 1.3 3.4 1 .1-.8.4-1.3.7-1.6-2.6-.3-5.3-1.3-5.3-5.8 0-1.3.5-2.3 1.2-3.2-.1-.3-.5-1.5.1-3.1 0 0 1-.3 3.3 1.2a11.4 11.4 0 0 1 6 0C17.6 4.7 18.6 5 18.6 5c.6 1.6.2 2.8.1 3.1.8.9 1.2 1.9 1.2 3.2 0 4.5-2.7 5.5-5.3 5.8.4.4.8 1.1.8 2.2v3.3c0 .4.2.7.8.6 4.6-1.5 7.9-5.8 7.9-10.9C23.5 5.7 18.3.5 12 .5z"/></svg>
    </a>
  </div>
</header>
{hero}
{main}
<footer class="footer"><div class="footer-inner">
  <span>Conquer3D v{esc(version)} — MIT licensed</span>
  <span class="spacer"></span>
  <a href="{REPO}">GitHub</a>
  <a href="https://pypi.org/project/conquer3d/">PyPI</a>
  <a href="https://hub.docker.com/r/kohido/conquer3d">Docker</a>
  <a href="{base}about.html">References</a>
</div></footer>

<div class="lightbox" id="lightbox">
  <button class="lightbox-close" aria-label="Close">✕</button>
  <img src="" alt="">
  <div class="lightbox-cap"></div>
</div>

<div class="search-modal" id="search-modal">
  <div class="search-box">
    <input id="search-input" type="text" placeholder="Search the API — functions, kernels, classes…" autocomplete="off" spellcheck="false">
    <div class="search-results" id="search-results"></div>
  </div>
</div>

<script defer src="{KATEX}/katex.min.js"></script>
<script defer src="{KATEX}/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],ignoredTags:['script','noscript','style','textarea','pre','code'],throwOnError:false}});"></script>
<script src="{base}assets/js/site.js"></script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Symbol rendering
# --------------------------------------------------------------------------- #

_KIND_LABEL = {
    "function": "function",
    "method": "method",
    "property": "property",
    "class": "class",
    "module": "module",
    "kernel": "__global__",
    "device": "__device__",
    "inline": "inline",
    "struct": "struct",
    "table": "__constant__",
    "macro": "macro",
    "typedef": "typedef",
    "constant": "constant",
}


def _signature_html(symbol: Symbol) -> str:
    if not symbol.signature:
        return ""
    lang = "cpp" if symbol.language in ("cuda", "cpp") else "python"
    name = symbol.qualname or symbol.name
    return f'<div class="signature">{highlight(name + symbol.signature, lang)}</div>'


def _params_table(title: str, params: Iterable, *, with_default: bool = True) -> str:
    params = list(params)
    if not params:
        return ""
    show_default = with_default and any(p.default for p in params)
    show_type = any(p.type for p in params)

    head = "<th>Name</th>"
    if show_type:
        head += "<th>Type</th>"
    if show_default:
        head += "<th>Default</th>"
    head += "<th>Description</th>"

    rows = []
    for param in params:
        cells = f'<td class="pname">{esc(param.name)}</td>'
        if show_type:
            cells += f'<td class="ptype">{esc(param.type)}</td>'
        if show_default:
            cells += f'<td class="pdef">{esc(param.default)}</td>'
        direction = (
            f'<span class="badge">{esc(param.direction)}</span> ' if param.direction else ""
        )
        cells += f'<td class="pdesc">{direction}{md_inline(param.description)}</td>'
        rows.append(f"<tr>{cells}</tr>")

    return (
        f'<div class="field-head">{esc(title)}</div>'
        f'<div class="table-wrap"><table class="params"><thead><tr>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _returns_html(doc: Doc) -> str:
    if not (doc.returns_type or doc.returns_description or doc.returns_items):
        return ""
    parts = ['<div class="field-head">Returns</div>']
    if doc.returns_type:
        parts.append(f'<div class="signature">{highlight(doc.returns_type, "python")}</div>')
    if doc.returns_description:
        parts.append(f'<div class="prose">{md_block(doc.returns_description)}</div>')
    if doc.returns_items:
        rows = "".join(
            f'<tr><td class="pname">{esc(item.name)}</td>'
            f'<td class="ptype">{esc(item.type)}</td>'
            f'<td class="pdesc">{md_inline(item.description)}</td></tr>'
            for item in doc.returns_items
        )
        parts.append(
            '<div class="table-wrap"><table class="params"><thead><tr>'
            "<th>Element</th><th>Type</th><th>Description</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    return "".join(parts)


def _admonitions(doc: Doc) -> str:
    out = []
    for note in doc.notes:
        out.append(f'<div class="adm adm-note"><div class="adm-title">Note</div>{md_block(note)}</div>')
    for warning in doc.warnings:
        out.append(
            f'<div class="adm adm-warn"><div class="adm-title">Warning</div>{md_block(warning)}</div>'
        )
    return "".join(out)


def render_symbol(symbol: Symbol, *, base: str, depth: int = 0) -> str:
    """Render one symbol card, recursing into class members."""
    doc = symbol.doc
    kind_label = _KIND_LABEL.get(symbol.kind, symbol.kind)
    badges = [f'<span class="badge">{esc(kind_label)}</span>']
    for flag in symbol.flags[:3]:
        # The kind label is already derived from the qualifier for kernels and
        # device helpers, so re-badging it adds noise.
        if flag and len(flag) < 26 and flag != kind_label:
            badges.append(f'<span class="badge">{esc(flag)}</span>')
    if not symbol.documented:
        badges.append('<span class="badge badge-undoc">undocumented</span>')

    source = ""
    if symbol.source_file:
        line = f"#L{symbol.source_line}" if symbol.source_line else ""
        source = (
            f'<a class="sym-src" href="{REPO}/blob/main/{symbol.source_file}{line}">'
            f"{esc(symbol.source_file)}{esc(':' + str(symbol.source_line) if symbol.source_line else '')}</a>"
        )

    name = symbol.qualname or symbol.name
    if "." in name and symbol.kind in ("method", "property"):
        prefix, _, short = name.rpartition(".")
        display = f'<span class="prefix">{esc(prefix)}.</span>{esc(short)}'
    else:
        display = esc(name)

    body = [_signature_html(symbol)]
    if doc.summary:
        body.append(f'<p class="summary">{md_inline(doc.summary)}</p>')
    if doc.description:
        body.append(f'<div class="prose">{md_block(doc.description)}</div>')
    if symbol.bases:
        bases = ", ".join(esc(b) for b in symbol.bases)
        body.append(f'<div class="field-head">Inherits</div><div class="signature">{bases}</div>')

    body.append(_params_table("Parameters", doc.params))
    body.append(_params_table("Attributes", doc.attributes, with_default=False))
    body.append(_returns_html(doc))
    body.append(_params_table("Raises", doc.raises, with_default=False))
    body.append(_admonitions(doc))

    for example in doc.examples:
        lang = "cpp" if symbol.language in ("cuda", "cpp") else "python"
        body.append(f'<div class="field-head">Example</div>{code_block(example, lang)}')

    for reference in doc.references:
        body.append(f'<div class="field-head">References</div><div class="prose">{md_block(reference)}</div>')

    if not doc.summary and not doc.description and not doc.params:
        body.append(
            '<div class="adm adm-warn"><div class="adm-title">Undocumented</div>'
            "<p>This symbol has no docstring yet. Its signature is shown above.</p></div>"
        )

    children = "".join(render_symbol(c, base=base, depth=depth + 1) for c in symbol.children)

    return (
        f'<section class="sym" id="{symbol.anchor}">'
        f'<div class="sym-head">'
        f'<span class="sym-name">{display}</span>'
        f'{"".join(badges)}'
        f'<a class="sym-anchor" href="#{symbol.anchor}">#</a>'
        f"{source}</div>"
        f'<div class="sym-body">{"".join(p for p in body if p)}{children}</div>'
        f"</section>"
    )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #


def build_sidebar(groups: List[Group], base: str, current: str) -> str:
    """Sidebar grouping every API page under its tier."""
    by_tier: Dict[str, List[Group]] = {}
    for group in groups:
        by_tier.setdefault(group.tier, []).append(group)

    blocks = []
    for tier in TIER_ORDER:
        if tier not in by_tier:
            continue
        meta = TIERS[tier]
        links = []
        for group in by_tier[tier]:
            documented, total = group.counts
            active = " active" if group.slug == current else ""
            href = f"{base}api/{TIERS[group.tier]['slug']}-{group.slug}.html"
            links.append(
                f'<a class="side-link{active}" href="{href}">{esc(group.title)}'
                f'<span class="count">{total}</span></a>'
            )
        blocks.append(
            f'<div class="side-group">'
            f'<div class="side-head"><span class="tier-dot" style="background:var(--a-{_tier_colour(tier)})"></span>'
            f"{esc(tier)} · {esc(meta['name'])}</div>{''.join(links)}</div>"
        )
    return "".join(blocks)


_TIER_COLOURS = {
    "T1": "green", "T2": "cyan", "T3": "violet",
    "T4": "amber", "T5": "rose", "T6": "cyan", "T7": "green",
}


def _tier_colour(tier: str) -> str:
    return _TIER_COLOURS.get(tier, "cyan")


def build_toc(group: Group) -> str:
    """Right-hand 'on this page' index."""
    links = []
    for symbol in group.symbols:
        links.append(f'<a href="#{symbol.anchor}">{esc(symbol.name)}</a>')
        for child in symbol.children:
            links.append(f'<a class="lvl-2" href="#{child.anchor}">{esc(child.name)}</a>')
    if not links:
        return ""
    return f'<div class="toc-head">On this page</div>{"".join(links[:400])}'
