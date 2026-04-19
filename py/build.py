#!/usr/bin/env python3
"""
build.py — Static site generator for krishnaShreedhar.github.io
================================================================

Converts Markdown files in markdowns/ into finished HTML pages using
the CSS / template system in the repository.

Usage
-----
    # Build everything
    python py/build.py

    # Build one file
    python py/build.py markdowns/technical/intro-to-transformers.md

    # Watch for changes (requires watchdog)
    python py/build.py --watch

Requirements
------------
    pip install -r py/requirements.txt

How it works
------------
1.  Reads YAML frontmatter from a .md file.
2.  Converts the Markdown body to HTML using mistune.
3.  Applies post-processing:
    - Wraps ``` code blocks with .code-wrapper + .code-lang label
    - Converts [^N] footnotes to numbered reference list
    - Converts > [!NOTE/TIP/WARN] callout syntax
    - Wraps $$...$$ in .math-block divs
    - Wraps ```mermaid blocks in .mermaid-wrap divs
4.  Injects into the appropriate HTML template based on category.
5.  Writes the output HTML file.

Adding a new blog post
----------------------
1.  Create a Markdown file in markdowns/<category>/<slug>.md
    with the YAML frontmatter block (see existing files for reference).
2.  Run: python py/build.py markdowns/<category>/<slug>.md
3.  The output HTML is written to the path specified by `output:` in frontmatter.
4.  Add a post-card entry to blogs/<category>/index.html pointing to the new file.
"""

import argparse
import os
import re
import sys
import textwrap
from datetime import datetime
from pathlib import Path

# ── optional dependencies — install via requirements.txt ──────────────────────
try:
    import yaml
except ImportError:
    sys.exit("PyYAML not found. Run: pip install pyyaml")

try:
    import mistune
except ImportError:
    sys.exit("mistune not found. Run: pip install mistune")

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "content" / "config.yaml"
MARKDOWNS_DIR = REPO_ROOT / "markdowns"

# ── load site config ──────────────────────────────────────────────────────────
def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

SITE_CONFIG = load_config()

# ── frontmatter parsing ───────────────────────────────────────────────────────
FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

def parse_file(path: Path):
    """Return (meta: dict, body: str) for a markdown file."""
    text = path.read_text(encoding="utf-8")
    m = FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) or {}
    body = text[m.end():]
    return meta, body


# ── markdown post-processing ──────────────────────────────────────────────────

def process_callouts(html: str) -> str:
    """Convert GitHub-style > [!NOTE], > [!TIP], > [!WARN] blockquotes."""
    def replace_callout(m):
        kind = m.group(1).lower()   # note | tip | warn
        content = m.group(2).strip()
        label_map = {"note": "Note", "tip": "Tip", "warn": "Warning"}
        label = label_map.get(kind, kind.capitalize())
        return (
            f'<div class="callout callout--{kind}">'
            f'<span class="callout__label">{label}</span>'
            f'{content}</div>'
        )
    # Pattern: <blockquote>\n<p>[!KIND]\ncontent</p>\n</blockquote>
    pattern = re.compile(
        r'<blockquote>\s*<p>\[!(NOTE|TIP|WARN)\]\s*(.*?)</p>\s*</blockquote>',
        re.DOTALL | re.IGNORECASE
    )
    return pattern.sub(replace_callout, html)


def process_math(html: str) -> str:
    """Wrap display math $$ ... $$ in .math-block divs."""
    # mistune doesn't handle math — we do it as raw HTML
    def wrap_display(m):
        expr = m.group(1)
        return f'<div class="math-block">\\[{expr}\\]</div>'
    html = re.sub(r'\$\$(.*?)\$\$', wrap_display, html, flags=re.DOTALL)
    return html


def process_mermaid(html: str) -> str:
    """Wrap <pre><code class="language-mermaid"> in .mermaid-wrap."""
    def wrap_mermaid(m):
        code = m.group(1)
        return (
            '<div class="mermaid-wrap">'
            '<div class="mermaid">' + code + '</div>'
            '</div>'
        )
    pattern = re.compile(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        re.DOTALL
    )
    return pattern.sub(wrap_mermaid, html)


def process_code_wrappers(html: str) -> str:
    """Add .code-wrapper + language label to fenced code blocks."""
    def wrap_code(m):
        lang = m.group(1) or ""
        code_block = m.group(0)
        if lang:
            return (
                f'<div class="code-wrapper">'
                f'<span class="code-lang">{lang}</span>'
                f'{code_block}'
                f'</div>'
            )
        return code_block
    pattern = re.compile(
        r'<pre><code class="language-(\w+)">(.*?)</code></pre>',
        re.DOTALL
    )
    return pattern.sub(wrap_code, html)


def process_footnotes(body_md: str, html: str):
    """
    Extract [^N]: ... footnote definitions from markdown body,
    replace [^N] inline citations with <a class='cite'>[N]</a>,
    and append a <section class='post-references'> to the HTML.

    Returns (processed_html, refs_html).
    """
    # Extract definitions
    defs = {}
    def_pattern = re.compile(r'^\[\^(\w+)\]:\s*(.+(?:\n(?!  ?\[).+)*)',
                              re.MULTILINE)
    for m in def_pattern.finditer(body_md):
        key, text = m.group(1), m.group(2).strip()
        defs[key] = text

    if not defs:
        return html, ""

    # Assign numeric IDs in order of appearance in body
    ordered = []
    for key in defs:
        ordered.append(key)

    # Replace inline citations [^key] -> <a class="cite" href="#ref-N">[N]</a>
    for i, key in enumerate(ordered, 1):
        html = html.replace(
            f'[^{key}]',
            f'<a class="cite" href="#ref-{i}">[{i}]</a>'
        )

    # Build references section
    items = ""
    for i, key in enumerate(ordered, 1):
        items += f'<li id="ref-{i}">{defs[key]}</li>\n'

    refs_html = (
        '<section class="post-references">'
        '<h2 class="refs-title">References</h2>'
        f'<ol class="references-list">{items}</ol>'
        '</section>'
    )
    return html, refs_html


# ── relative path helper ──────────────────────────────────────────────────────

def depth_prefix(output_path: Path) -> str:
    """Return '../' repeated by directory depth relative to repo root."""
    rel = output_path.relative_to(REPO_ROOT)
    depth = len(rel.parts) - 1   # exclude the filename itself
    return "../" * depth


# ── CDN includes ──────────────────────────────────────────────────────────────

def cdn_includes(meta: dict, prefix: str) -> dict:
    """Return dict of CDN <link> and <script> tags based on post requirements."""
    cdn = SITE_CONFIG.get("cdn", {})
    requires = meta.get("requires", [])

    head_extras = ""
    body_extras = ""

    if "highlight" in requires or meta.get("category") == "technical":
        head_extras += (
            f'<link rel="stylesheet" href="{cdn.get("highlight_css", "")}">\n'
        )
        body_extras += (
            f'<script src="{cdn.get("highlight_js", "")}"></script>\n'
            '<script>document.addEventListener("DOMContentLoaded",function(){'
            'document.querySelectorAll("pre code").forEach(function(el){'
            'hljs.highlightElement(el);});});</script>\n'
        )

    if "katex" in requires or "$$" in meta.get("subtitle", "") or True:
        # Always include KaTeX on technical posts; it's lightweight when unused
        if meta.get("category") in ("technical", "growth"):
            head_extras += (
                f'<link rel="stylesheet" href="{cdn.get("katex_css", "")}">\n'
            )
            body_extras += (
                f'<script defer src="{cdn.get("katex_js", "")}"></script>\n'
                f'<script defer src="{cdn.get("katex_auto", "")}" '
                'onload="renderMathInElement(document.body,{delimiters:['
                '{left:\'\\\\[\',right:\'\\\\]\',display:true},'
                '{left:\'\\\\(\',right:\'\\\\)\',display:false},'
                '{left:\'$$\',right:\'$$\',display:true},'
                '{left:\'$\',right:\'$\',display:false}]});"></script>\n'
            )

    if "mermaid" in requires or meta.get("category") == "technical":
        body_extras += (
            f'<script src="{cdn.get("mermaid_js", "")}"></script>\n'
            '<script>mermaid.initialize({startOnLoad:true,theme:"dark"});</script>\n'
        )

    return {"head": head_extras, "body": body_extras}


# ── HTML template ─────────────────────────────────────────────────────────────

def render_post(meta: dict, content_html: str, refs_html: str,
                output_path: Path) -> str:
    """Assemble the full HTML page for a blog post."""
    prefix = depth_prefix(output_path)
    cdns = cdn_includes(meta, prefix)

    category = meta.get("category", "technical")
    title = meta.get("title", "Post")
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", SITE_CONFIG.get("site", {}).get("author", ""))
    date_val = meta.get("date", "")
    if isinstance(date_val, datetime):
        date_str = date_val.strftime("%d %B %Y")
    else:
        date_str = str(date_val)
    reading_time = meta.get("reading_time", "?")
    tags = meta.get("tags", [])

    # Breadcrumb + nav depth
    nav = {
        "home":       f'{prefix}index.html',
        "experience": f'{prefix}experience/',
        "blog":       f'{prefix}blogs/',
        "category":   f'{prefix}blogs/{category}/',
        "resources":  f'{prefix}resources.html',
        "css_base":   f'{prefix}css/base.css',
        "css_main":   f'{prefix}css/main.css',
        "css_blog":   f'{prefix}css/blog.css',
        "js_mod":     f'{prefix}js/modernizr.js',
        "js_blog":    f'{prefix}js/blog.js',
        "favicon":    f'{prefix}favicon.png',
    }

    tag_html = "".join(
        f'<span class="tag tag--{category}">{t}</span>' for t in tags
    )

    cat_labels = {
        "technical": "Technical",
        "leadership": "Leadership",
        "community": "Community",
        "growth": "Growth",
    }
    cat_label = cat_labels.get(category, category.capitalize())

    return f"""<!DOCTYPE html>
<html class="no-js" lang="en">
<head>
    <meta charset="utf-8">
    <title>{title} &middot; {author}</title>
    <meta name="description" content="{subtitle[:160]}">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <link rel="stylesheet" href="{nav['css_base']}">
    <link rel="stylesheet" href="{nav['css_main']}">
    <link rel="stylesheet" href="{nav['css_blog']}">
    {cdns['head']}
    <script src="{nav['js_mod']}"></script>
    <link rel="icon" type="image/png" href="{nav['favicon']}">
</head>
<body id="top" class="sk-page">

<div class="reading-progress"></div>

<button class="theme-toggle" aria-label="Toggle theme">
    <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
    <svg class="icon-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"/>
        <line x1="12" y1="1" x2="12" y2="3"/>
        <line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>
        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
        <line x1="1" y1="12" x2="3" y2="12"/>
        <line x1="21" y1="12" x2="23" y2="12"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/>
        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
    </svg>
</button>

<nav class="site-ribbon" aria-label="Page sections">
    <ul>
        <li><a class="ribbon-dot" href="#top" data-section="top">
            <span class="ribbon-label">Top</span>
        </a></li>
        <li><a class="ribbon-dot" href="#post-article" data-section="post-article">
            <span class="ribbon-label">Article</span>
        </a></li>
    </ul>
</nav>

<header>
    <div class="row">
        <div class="top-bar">
            <div class="logo"><a href="{nav['home']}" aria-label="Home">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>
                    <polyline points="9 22 9 12 15 12 15 22"/>
                </svg>
            </a></div>
            <nav id="main-nav-wrap">
                <ul class="main-navigation">
                    <li><a href="{nav['home']}">Home</a></li>
                    <li><a href="{nav['experience']}">Experience</a></li>
                    <li class="current"><a href="{nav['blog']}">Blog</a></li>
                    <li><a href="{nav['resources']}">Resources</a></li>
                </ul>
            </nav>
        </div>
    </div>
</header>

<div class="post-header">
    <div class="row">
        <div class="post-breadcrumb">
            <a href="{nav['blog']}">Blog</a>
            <span class="post-breadcrumb__sep">/</span>
            <a href="{nav['category']}">{cat_label}</a>
            <span class="post-breadcrumb__sep">/</span>
            <span class="post-breadcrumb__current">{title[:50]}</span>
        </div>
        <h1 class="post-title">{title}</h1>
        <p class="post-subtitle">{subtitle}</p>
        <div class="post-meta">
            <span class="post-meta__author">{author}</span>
            <span class="post-meta__sep">&middot;</span>
            <span>{date_str}</span>
            <span class="post-meta__sep">&middot;</span>
            <span>{reading_time} min read</span>
        </div>
        <div class="tags-row">{tag_html}</div>
    </div>
</div>

<div class="row">
<div class="post-layout">
    <aside class="post-toc">
        <div class="post-toc__title">Contents</div>
        <nav class="toc-nav"><ul data-auto></ul></nav>
    </aside>
    <article class="post-body" id="post-article">
        {content_html}
        {refs_html}
        <nav class="post-nav">
            <a href="{nav['category']}" class="post-nav__item post-nav__item--prev">
                <span class="post-nav__label">&larr; Back</span>
                <span class="post-nav__title">{cat_label} Blog</span>
            </a>
            <span></span>
        </nav>
    </article>
</div>
</div>

<footer class="sk-footer">
    <div class="sk-row">
        <div class="sk-footer__inner">
            <span class="sk-footer__copy">&copy; {author}</span>
            <div class="sk-footer__links">
                <a href="{nav['category']}">{cat_label}</a>
                <a href="{nav['blog']}">All Blogs</a>
                <a href="{nav['home']}">Home</a>
            </div>
        </div>
    </div>
</footer>

{cdns['body']}
<script src="{nav['js_blog']}"></script>
</body>
</html>"""


# ── main build function ───────────────────────────────────────────────────────

def build_file(md_path: Path, verbose: bool = True):
    """Build a single markdown file to HTML."""
    if not md_path.exists():
        print(f"ERROR: file not found: {md_path}")
        return False

    meta, body = parse_file(md_path)

    if not meta.get("output"):
        print(f"SKIP: no 'output' in frontmatter for {md_path.name}")
        return False

    output_path = REPO_ROOT / meta["output"]

    # Convert markdown → HTML
    md = mistune.create_markdown(
        plugins=["strikethrough", "table", "url"],
        escape=False
    )
    html = md(body)

    # Post-process
    html = process_callouts(html)
    html = process_mermaid(html)
    html = process_code_wrappers(html)
    html = process_math(html)
    html, refs_html = process_footnotes(body, html)

    # Render full page
    page_html = render_post(meta, html, refs_html, output_path)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page_html, encoding="utf-8")

    if verbose:
        print(f"  ✓  {md_path.relative_to(REPO_ROOT)}"
              f"  →  {output_path.relative_to(REPO_ROOT)}")
    return True


def build_all(verbose: bool = True):
    """Build every markdown file with an 'output' frontmatter key."""
    md_files = list(MARKDOWNS_DIR.rglob("*.md"))
    if not md_files:
        print("No markdown files found in markdowns/")
        return

    success = 0
    for f in sorted(md_files):
        if build_file(f, verbose):
            success += 1

    print(f"\nBuilt {success}/{len(md_files)} files.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build HTML from Markdown for krishnaShreedhar.github.io"
    )
    parser.add_argument(
        "files", nargs="*",
        help="Specific markdown file(s) to build. Omit to build all."
    )
    parser.add_argument(
        "--watch", action="store_true",
        help="Watch markdowns/ for changes and rebuild automatically."
    )
    args = parser.parse_args()

    if args.watch:
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
        except ImportError:
            sys.exit("watchdog not installed. Run: pip install watchdog")

        class Handler(FileSystemEventHandler):
            def on_modified(self, event):
                if event.src_path.endswith(".md"):
                    build_file(Path(event.src_path))

        print(f"Watching {MARKDOWNS_DIR} for changes…  (Ctrl+C to stop)")
        observer = Observer()
        observer.schedule(Handler(), str(MARKDOWNS_DIR), recursive=True)
        observer.start()
        try:
            import time
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()
        return

    if args.files:
        for f in args.files:
            build_file(Path(f))
    else:
        build_all()


if __name__ == "__main__":
    main()
