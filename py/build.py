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
1.  Create a M  arkdown file in markdowns/<category>/<slug>.md
    with the YAML frontmatter block (see existing files for reference).
2.  Run: python py/build.py markdowns/<category>/<slug>.md
3.  The output HTML is written to the path specified by `output:` in frontmatter.
4.  Add a post-card entry to blogs/<category>/index.html pointing to the new file.
"""

import argparse
import os
import re
import subprocess
import sys
import textwrap
from datetime import date, datetime
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
PROJECTS_ROOT = MARKDOWNS_DIR / "technical" / "projects"

# Anything that can change what build_file()/render_post() emit for a page
# that isn't the source .md itself — a template edit or a config change
# (author name, CDN URL, category color, ...) must invalidate every output,
# even though no markdown file touched. Bump GLOBAL_MTIME by touching either.
BUILD_PY_MTIME = Path(__file__).resolve().stat().st_mtime
CONFIG_MTIME = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
GLOBAL_MTIME = max(BUILD_PY_MTIME, CONFIG_MTIME)

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


# ── nested project docs (markdowns/technical/projects/<slug>/**) ──────────────
#
# These are multi-file technical write-ups (READMEs + docs/ trees) rather than
# single flat posts. Files here usually arrive without frontmatter. build_all()
# fills in sensible defaults on first run (idempotent — files that already have
# a frontmatter block are left untouched) and wires every project into a
# sidebar nav + prev/next chain so the nested structure stays navigable.

ACRONYMS = {
    "cuda", "cudnn", "gpu", "gpus", "ai", "ml", "aiml", "llm", "llmops",
    "mlops", "rl", "onnx", "api", "apis", "sre", "cicd", "ci", "cd", "iac",
    "sac", "ddpg", "td3", "dqn", "ppo", "a2c", "grpo", "kv", "http", "https",
    "rest", "graphql", "sql", "nosql", "json", "yaml", "ptx", "sass", "wmma",
    "ebpf", "soa", "solid", "dry", "kiss", "yagni",
}


def humanize_slug(slug: str) -> str:
    """'large_scale_aiml_systems' -> 'Large Scale AIML Systems'."""
    words = re.split(r"[_\-]+", slug)
    out = []
    for w in words:
        if not w:
            continue
        out.append(w.upper() if w.lower() in ACRONYMS else w.capitalize())
    return " ".join(out)


def yaml_dquote(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def extract_leading_h1(text: str):
    """If the first non-blank line is an H1, return (title, text-without-it)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and lines[i].strip() == "":
        i += 1
    if i < len(lines) and lines[i].startswith("# "):
        title = lines[i][2:].strip()
        rest = "\n".join(lines[:i] + lines[i + 1:])
        return title, rest.lstrip("\n")
    return None, text


def extract_subtitle(body: str) -> str:
    """Best-effort one-liner: first paragraph in the body, with hand-wrapped
    lines joined back together before truncating."""
    lines = body.splitlines()
    n = len(lines)
    i = 0
    while i < n:
        s = lines[i].strip()
        if not s or s.startswith(("#", ">", "```", "-", "*", "|", "$$")):
            i += 1
            continue
        para = [s]
        j = i + 1
        while j < n:
            nxt = lines[j].strip()
            if not nxt or nxt.startswith(("#", ">", "```", "-", "*", "|", "$$")):
                break
            para.append(nxt)
            j += 1
        text = " ".join(para)
        if len(text) > 220:
            text = text[:217].rsplit(" ", 1)[0] + "..."
        return text
    return ""


def committed_date(path: Path) -> date:
    """The date a markdown source file was actually added to the repo, so
    auto-injected frontmatter dates reflect real publish history instead of
    an arbitrary/random value. Falls back to today for files not yet
    committed (the common case: this runs as part of authoring a new post,
    before it's committed) or when git isn't available."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "--diff-filter=A",
             "--format=%as", "--follow", "--", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        lines = [l for l in out.stdout.strip().splitlines() if l]
        if lines:
            return datetime.strptime(lines[-1], "%Y-%m-%d").date()
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    return date.today()


def ordered_project_files(project_dir: Path):
    """Depth-first file order for a project: README.md first in every
    directory, then the rest alphabetically, descending into subdirectories
    in alphabetical order. Drives both frontmatter dates and the sidebar."""
    def walk(d: Path):
        entries = list(d.iterdir())
        files = sorted(
            (p for p in entries if p.is_file() and p.suffix == ".md"),
            key=lambda p: (p.name != "README.md", p.name.lower()),
        )
        dirs = sorted((p for p in entries if p.is_dir()), key=lambda p: p.name.lower())
        result = list(files)
        for sub in dirs:
            result.extend(walk(sub))
        return result
    return walk(project_dir)


def inject_project_frontmatter(verbose: bool = True):
    """Add missing YAML frontmatter to every markdown file under
    markdowns/technical/projects/. Idempotent — a file that already starts
    with a '---' frontmatter block is left completely untouched."""
    if not PROJECTS_ROOT.exists():
        return

    author = SITE_CONFIG.get("site", {}).get("author", "Shreedhar Kodate")
    added = 0

    for project_dir in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        slug = project_dir.name
        root_readme = project_dir / "README.md"
        project_title = humanize_slug(slug)
        if root_readme.exists():
            raw = root_readme.read_text(encoding="utf-8")
            if not FRONTMATTER_RE.match(raw):
                h1, _ = extract_leading_h1(raw)
                if h1:
                    project_title = h1

        for f in ordered_project_files(project_dir):
            text = f.read_text(encoding="utf-8")
            if FRONTMATTER_RE.match(text):
                continue

            rel = f.relative_to(project_dir)
            title, body = extract_leading_h1(text)
            if not title:
                title = humanize_slug(f.stem)
            subtitle = extract_subtitle(body)

            if f.name == "README.md":
                if rel.parent == Path("."):
                    out_rel = f"blogs/technical/posts/{slug}/index.html"
                else:
                    out_rel = f"blogs/technical/posts/{slug}/{rel.parent.as_posix()}/index.html"
            else:
                out_rel = f"blogs/technical/posts/{slug}/{rel.with_suffix('.html').as_posix()}"

            word_count = len(re.findall(r"\S+", body))
            reading_time = max(1, round(word_count / 220))
            date_val = committed_date(f)

            tags = [slug.replace("_", "-")]
            if rel.parent != Path("."):
                second = rel.parts[0].split("_", 1)[-1].replace("_", "-")[:24]
                if second and second not in tags:
                    tags.append(second)

            meta_lines = ["---", f'title: "{yaml_dquote(title)}"']
            if subtitle:
                meta_lines.append(f'subtitle: "{yaml_dquote(subtitle)}"')
            meta_lines += [
                "category: technical",
                f"project: {slug}",
                f'project_title: "{yaml_dquote(project_title)}"',
                f'date: {date_val.strftime("%Y-%m-%d")}',
                f"reading_time: {reading_time}",
                "tags:",
            ]
            meta_lines += [f"  - {t}" for t in tags]
            meta_lines += [
                f'author: "{yaml_dquote(author)}"',
                f'output: "{out_rel}"',
                "---",
                "",
            ]
            f.write_text("\n".join(meta_lines) + body, encoding="utf-8")
            added += 1
            if verbose:
                print(f"  +  frontmatter: {f.relative_to(REPO_ROOT)}")

    if verbose and added:
        print(f"\nAdded frontmatter to {added} project doc(s).")


def build_project_index():
    """{slug: [(md_path, meta), ...]} in sidebar/prev-next order, built from
    files that now have an 'output' key (after inject_project_frontmatter)."""
    index = {}
    if not PROJECTS_ROOT.exists():
        return index
    for project_dir in sorted(p for p in PROJECTS_ROOT.iterdir() if p.is_dir()):
        entries = []
        for f in ordered_project_files(project_dir):
            meta, _ = parse_file(f)
            if meta.get("output"):
                entries.append((f, meta))
        index[project_dir.name] = entries
    return index


def render_project_nav(entries, project_dir: Path, current_path: Path,
                        current_output: Path) -> str:
    """Sidebar <ul> linking every page in this project, indented by depth,
    with the current page marked active and subfolder READMEs styled as
    section headers."""
    items = []
    for f, meta in entries:
        rel = f.relative_to(project_dir)
        depth = len(rel.parts) - 1
        classes = [f"project-nav__lvl-{min(depth, 2)}"]
        if f.name == "README.md" and depth >= 1:
            classes.append("project-nav__section")
        if f == current_path:
            classes.append("active")
        href = os.path.relpath(REPO_ROOT / meta["output"], start=current_output.parent)
        title = meta.get("title", f.stem)
        items.append(f'<li class="{" ".join(classes)}"><a href="{href}">{title}</a></li>')
    return '<ul class="project-nav__list">' + "".join(items) + "</ul>"


def render_prev_next(entries, current_path: Path, current_output: Path):
    """Return (prev_html, next_html) anchors for sequential reading through a
    project's docs, or (None, None) entries where there's no sibling."""
    idx = next((i for i, (f, _) in enumerate(entries) if f == current_path), None)
    prev_html = next_html = None
    if idx is None:
        return prev_html, next_html
    if idx > 0:
        f, m = entries[idx - 1]
        href = os.path.relpath(REPO_ROOT / m["output"], start=current_output.parent)
        prev_html = (
            f'<a href="{href}" class="post-nav__item post-nav__item--prev">'
            f'<span class="post-nav__label">&larr; Prev</span>'
            f'<span class="post-nav__title">{m.get("title", "")}</span></a>'
        )
    if idx < len(entries) - 1:
        f, m = entries[idx + 1]
        href = os.path.relpath(REPO_ROOT / m["output"], start=current_output.parent)
        next_html = (
            f'<a href="{href}" class="post-nav__item post-nav__item--next">'
            f'<span class="post-nav__label">Next &rarr;</span>'
            f'<span class="post-nav__title">{m.get("title", "")}</span></a>'
        )
    return prev_html, next_html


# ── series (sub-category) docs (markdowns/<category>/<series_slug>/*.md) ──────
#
# A "series" is a flat folder of sibling posts under any category — e.g.
# markdowns/leadership/learning_from_sandeep_swadia/ — that should be
# presented on the site as one grouped sub-collection (a single card on the
# category index pointing to a series landing page, with sidebar nav +
# prev/next between its posts). This mirrors the existing technical/projects
# mechanism but works for any category. technical/projects keeps its own
# (nested, README-driven) handling and is excluded here.

def discover_series_dirs():
    """{series_slug: (category, [md_path, ...])} for every flat sub-folder of
    a category directory that contains markdown files directly. Files are
    kept in filename order (series posts are numbered, e.g. 01_, 02_, ...)."""
    series = {}
    if not MARKDOWNS_DIR.exists():
        return series
    for category_dir in sorted(p for p in MARKDOWNS_DIR.iterdir() if p.is_dir()):
        for sub in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            if category_dir.name == "technical" and sub.name == "projects":
                continue
            md_files = sorted(p for p in sub.iterdir() if p.suffix == ".md")
            if md_files:
                series[sub.name] = (category_dir.name, md_files)
    return series


def series_output_slug(path: Path) -> str:
    """'01_Systems_Thinking.md' -> 'systems-thinking'."""
    stem = re.sub(r'^\d+[_\-]*', '', path.stem)
    return re.sub(r'[_\s]+', '-', stem).strip('-').lower()


def inject_series_frontmatter(verbose: bool = True):
    """Add missing YAML frontmatter to every markdown file inside a series
    folder. Idempotent — files that already start with a '---' frontmatter
    block are left completely untouched."""
    author = SITE_CONFIG.get("site", {}).get("author", "Shreedhar Kodate")
    added = 0

    for series_slug, (category, md_files) in discover_series_dirs().items():
        series_title = humanize_slug(series_slug)

        for f in md_files:
            text = f.read_text(encoding="utf-8")
            if FRONTMATTER_RE.match(text):
                continue

            title, body = extract_leading_h1(text)
            if not title:
                title = humanize_slug(f.stem)
            subtitle = extract_subtitle(body)

            word_count = len(re.findall(r"\S+", body))
            reading_time = max(1, round(word_count / 220))
            date_val = committed_date(f)
            out_rel = f"blogs/{category}/posts/{series_slug}/{series_output_slug(f)}.html"

            tags = [category, series_slug.replace("_", "-")[:24]]

            meta_lines = ["---", f'title: "{yaml_dquote(title)}"']
            if subtitle:
                meta_lines.append(f'subtitle: "{yaml_dquote(subtitle)}"')
            meta_lines += [
                f"category: {category}",
                f"series: {series_slug}",
                f'series_title: "{yaml_dquote(series_title)}"',
                f'date: {date_val.strftime("%Y-%m-%d")}',
                f"reading_time: {reading_time}",
                "tags:",
            ]
            meta_lines += [f"  - {t}" for t in tags]
            meta_lines += [
                f'author: "{yaml_dquote(author)}"',
                f'output: "{out_rel}"',
                "---",
                "",
            ]
            f.write_text("\n".join(meta_lines) + body, encoding="utf-8")
            added += 1
            if verbose:
                print(f"  +  frontmatter: {f.relative_to(REPO_ROOT)}")

    if verbose and added:
        print(f"\nAdded frontmatter to {added} series doc(s).")


def build_series_index():
    """{series_slug: [(md_path, meta), ...]} in reading order, built from
    files that now have an 'output' key (after inject_series_frontmatter)."""
    index = {}
    for series_slug, (category, md_files) in discover_series_dirs().items():
        entries = []
        for f in md_files:
            meta, _ = parse_file(f)
            if meta.get("output"):
                entries.append((f, meta))
        index[series_slug] = entries
    return index


def render_series_index_page(series_slug: str, category: str, entries) -> str:
    """Build the series landing page: a scoped listing of every post in the
    series, styled like a category index page."""
    cat_cfg = next(
        (c for c in SITE_CONFIG.get("categories", []) if c.get("id") == category),
        {},
    )
    cat_name = cat_cfg.get("name", category.capitalize())
    series_title = humanize_slug(series_slug)

    cards = []
    for f, meta in entries:
        output_rel = Path(meta["output"])
        href = output_rel.name  # sibling file inside the same posts/<series>/ dir
        date_val = meta.get("date", "")
        date_str = date_val.strftime("%d %b %Y") if isinstance(date_val, date) else str(date_val)
        day = date_val.strftime("%d") if isinstance(date_val, date) else ""
        month = date_val.strftime("%b&nbsp;%Y") if isinstance(date_val, date) else date_str
        tag_html = "".join(f'<span class="tag tag--{category}">{t}</span>' for t in meta.get("tags", []))
        cards.append(f"""
            <a href="{href}" class="post-card post-card--{category} anim-in">
                <div class="post-card__date">
                    <span class="post-card__date-day">{day}</span>
                    <span class="post-card__date-month">{month}</span>
                </div>
                <div class="post-card__body">
                    <h2 class="post-card__title">{meta.get('title', '')}</h2>
                    <p class="post-card__excerpt">{meta.get('subtitle', '')}</p>
                    <div class="post-card__footer">
                        {tag_html}
                        <span class="post-card__readtime">&#9711;&nbsp;{meta.get('reading_time', '?')} min</span>
                    </div>
                </div>
            </a>""")

    author = SITE_CONFIG.get("site", {}).get("author", "Shreedhar Kodate")

    return f"""<!DOCTYPE html>
<html class="no-js" lang="en">
<head>
    <meta charset="utf-8">
    <title>{series_title} &middot; {author}</title>
    <meta name="description" content="A series of posts on {series_title}.">
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
    <link rel="stylesheet" href="../../../../css/base.css">
    <link rel="stylesheet" href="../../../../css/main.css">
    <link rel="stylesheet" href="../../../../css/blog.css">
    <script src="../../../../js/modernizr.js"></script>
    <link rel="icon" type="image/png" href="../../../../favicon.png">
</head>
<body id="top" class="sk-page">

<div class="reading-progress"></div>

<button class="theme-toggle" aria-label="Toggle theme">
    <svg class="icon-moon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
    <svg class="icon-sun"  viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>
</button>

<header>
    <div class="row">
        <div class="top-bar">
            <div class="logo"><a href="../../../../index.html" aria-label="Home"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg></a></div>
            <nav id="main-nav-wrap">
                <ul class="main-navigation">
                    <li><a href="../../../../index.html">Home</a></li>
                    <li><a href="../../../../experience/">Experience</a></li>
                    <li class="current"><a href="../../../">Blog</a></li>
                    <li><a href="../../../../resources/">Resources</a></li>
                </ul>
            </nav>
        </div>
    </div>
</header>

<section class="page-hero page-hero--{category}">
    <div class="row">
        <span class="page-hero__eyebrow">{cat_name} &middot; Series</span>
        <h1>{series_title}</h1>
        <p class="page-hero__desc">
            A grouped series of posts within {cat_name}.
        </p>
    </div>
</section>

<section class="blog-listing">
    <div class="row">
        <div class="post-grid">
{"".join(cards)}
        </div>
    </div>
</section>

<footer class="sk-footer">
    <div class="sk-row">
        <div class="sk-footer__inner">
            <span class="sk-footer__copy">&copy; {author}</span>
            <div class="sk-footer__links">
                <a href="../../../">All Blogs</a>
                <a href="../../../../index.html">Home</a>
            </div>
        </div>
    </div>
</footer>

<script src="../../../../js/blog.js"></script>
</body>
</html>
"""


def build_series_landing_pages(series_index: dict, verbose: bool = True, force: bool = False):
    """Write the grouped landing page for every discovered series. Skipped
    when the existing landing page is already newer than every member post
    and the template/config (same freshness rule as build_file)."""
    dirs = discover_series_dirs()
    for series_slug, entries in series_index.items():
        if not entries:
            continue
        category = dirs[series_slug][0]
        out_dir = REPO_ROOT / f"blogs/{category}/posts/{series_slug}"
        out_path = out_dir / "index.html"

        if not force and out_path.exists():
            source_mtime = max(f.stat().st_mtime for f, _ in entries)
            if out_path.stat().st_mtime >= max(source_mtime, GLOBAL_MTIME):
                if verbose:
                    print(f"  ·  up to date: series index  {out_path.relative_to(REPO_ROOT)}")
                continue

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(render_series_index_page(series_slug, category, entries), encoding="utf-8")
        if verbose:
            print(f"  ✓  series index  →  {out_path.relative_to(REPO_ROOT)}")


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


def process_relative_md_links(html: str) -> str:
    """Rewrite in-body links to sibling markdown docs (e.g. '01_foo.md',
    'sub/README.md') to their built .html equivalents, so cross-references
    between project docs keep working after the static build."""
    def fix(m):
        href = m.group(1)
        if re.match(r'^[a-zA-Z][a-zA-Z0-9+.\-]*:', href):  # scheme:// or mailto:
            return m.group(0)
        path, _, frag = href.partition('#')
        if not path.endswith('.md'):
            return m.group(0)
        if path == 'README.md' or path.endswith('/README.md'):
            new_path = path[:-len('README.md')] + 'index.html'
        else:
            new_path = path[:-3] + '.html'
        new_href = new_path + (('#' + frag) if frag else '')
        return f'<a href="{new_href}"'
    return re.sub(r'<a href="([^"]+)"', fix, html)


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
    """Return dict of CDN <link> and <script> tags based on post requirements.

    Library inclusion is driven primarily by what's actually present in the
    rendered post (meta['_has_mermaid'] / meta['_has_math'], set by
    build_file() from the post-processed HTML) so any category — not just
    "technical" — gets working diagrams/math when it uses them. The
    'requires' frontmatter key and legacy per-category defaults still work
    as an override/fallback.
    """
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

    needs_math = "katex" in requires or meta.get("_has_math") or meta.get("category") in ("technical", "growth")
    if needs_math:
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

    needs_mermaid = "mermaid" in requires or meta.get("_has_mermaid") or meta.get("category") == "technical"
    if needs_mermaid:
        body_extras += (
            f'<script src="{cdn.get("mermaid_js", "")}"></script>\n'
            '<script>(function(){'
            'var t=document.body.dataset.theme==="light"?"default":"dark";'
            'mermaid.initialize({startOnLoad:true,theme:t});'
            '})();</script>\n'
        )

    return {"head": head_extras, "body": body_extras}


# ── HTML template ─────────────────────────────────────────────────────────────

def render_post(meta: dict, content_html: str, refs_html: str,
                output_path: Path, project_nav_html: str = "",
                prev_html: str = None, next_html: str = None) -> str:
    """Assemble the full HTML page for a blog post."""
    prefix = depth_prefix(output_path)
    cdns = cdn_includes(meta, prefix)

    category = meta.get("category", "technical")
    title = meta.get("title", "Post")
    subtitle = meta.get("subtitle", "")
    author = meta.get("author", SITE_CONFIG.get("site", {}).get("author", ""))
    date_val = meta.get("date", "")
    if isinstance(date_val, date):
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
        "resources":  f'{prefix}resources/',
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

    if project_nav_html:
        series_label = meta.get('project_title') or meta.get('series_title') or 'Contents'
        toc_aside = f"""<div class="post-toc__title">{series_label}</div>
        <nav class="project-nav">{project_nav_html}</nav>
        <hr class="project-nav__divider">
        <div class="post-toc__title">On This Page</div>
        <nav class="toc-nav"><ul data-auto></ul></nav>"""
    else:
        toc_aside = """<div class="post-toc__title">Contents</div>
        <nav class="toc-nav"><ul data-auto></ul></nav>"""

    prev_block = prev_html or (
        f'<a href="{nav["category"]}" class="post-nav__item post-nav__item--prev">'
        f'<span class="post-nav__label">&larr; Back</span>'
        f'<span class="post-nav__title">{cat_label} Blog</span></a>'
    )
    next_block = next_html or "<span></span>"

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
        {toc_aside}
    </aside>
    <article class="post-body" id="post-article">
        {content_html}
        {refs_html}
        <nav class="post-nav">
            {prev_block}
            {next_block}
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


<script src="{nav['js_blog']}"></script>
{cdns['body']}
</body>
</html>"""


# ── main build function ───────────────────────────────────────────────────────

def build_file(md_path: Path, verbose: bool = True, project_index: dict = None,
               series_index: dict = None, force: bool = False):
    """Build a single markdown file to HTML.

    Skips the rebuild (mistune conversion + write) when the existing output
    is already newer than every input that could affect it: the source .md,
    build.py / config.yaml (template or site-wide settings), and — for
    project/series members — every sibling post's .md (their titles feed
    this page's sidebar nav). Pass force=True to always rebuild.
    """
    md_path = md_path.resolve()
    if not md_path.exists():
        print(f"ERROR: file not found: {md_path}")
        return False

    meta, body = parse_file(md_path)

    if not meta.get("output"):
        print(f"SKIP: no 'output' in frontmatter for {md_path.name}")
        return False

    output_path = REPO_ROOT / meta["output"]

    # Freshness check — bail out before any conversion work if nothing that
    # could affect this page's output has changed since it was last built.
    if not force and output_path.exists():
        source_mtime = md_path.stat().st_mtime
        slug = meta.get("project")
        if slug and project_index and slug in project_index:
            source_mtime = max(source_mtime, *(f.stat().st_mtime for f, _ in project_index[slug]))
        series_slug = meta.get("series")
        if series_slug and series_index and series_slug in series_index:
            source_mtime = max(source_mtime, *(f.stat().st_mtime for f, _ in series_index[series_slug]))

        if output_path.stat().st_mtime >= max(source_mtime, GLOBAL_MTIME):
            if verbose:
                print(f"  ·  up to date: {md_path.relative_to(REPO_ROOT)}")
            return True

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
    html = process_relative_md_links(html)
    html, refs_html = process_footnotes(body, html)

    # Drive CDN inclusion (mermaid/katex) off what's actually in the post,
    # not just its category, so any category renders diagrams/math correctly.
    meta["_has_mermaid"] = "mermaid-wrap" in html
    meta["_has_math"] = "math-block" in html

    # Nested project docs: sidebar nav + prev/next through the project
    project_nav_html = ""
    prev_html = next_html = None
    slug = meta.get("project")
    if slug and project_index and slug in project_index:
        entries = project_index[slug]
        project_dir = PROJECTS_ROOT / slug
        project_nav_html = render_project_nav(entries, project_dir, md_path, output_path)
        prev_html, next_html = render_prev_next(entries, md_path, output_path)

    # Sub-category "series" (e.g. leadership/learning_from_sandeep_swadia):
    # sidebar nav + prev/next through the series, same mechanism as projects.
    series_slug = meta.get("series")
    if series_slug and series_index and series_slug in series_index:
        entries = series_index[series_slug]
        series_dir = MARKDOWNS_DIR / meta.get("category", "") / series_slug
        project_nav_html = render_project_nav(entries, series_dir, md_path, output_path)
        prev_html, next_html = render_prev_next(entries, md_path, output_path)

    # Render full page
    page_html = render_post(meta, html, refs_html, output_path,
                             project_nav_html=project_nav_html,
                             prev_html=prev_html, next_html=next_html)

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page_html, encoding="utf-8")

    if verbose:
        print(f"  ✓  {md_path.relative_to(REPO_ROOT)}"
              f"  →  {output_path.relative_to(REPO_ROOT)}")
    return True


def build_all(verbose: bool = True, force: bool = False):
    """Build every markdown file with an 'output' frontmatter key. Files
    whose output is already up to date are skipped — see build_file()."""
    inject_project_frontmatter(verbose)
    inject_series_frontmatter(verbose)
    project_index = build_project_index()
    series_index = build_series_index()

    md_files = list(MARKDOWNS_DIR.rglob("*.md"))
    if not md_files:
        print("No markdown files found in markdowns/")
        return

    success = 0
    for f in sorted(md_files):
        if build_file(f, verbose, project_index=project_index, series_index=series_index, force=force):
            success += 1

    build_series_landing_pages(series_index, verbose, force=force)

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
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild every page even if its output is already up to date."
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
                    build_file(Path(event.src_path), project_index=build_project_index(),
                               series_index=build_series_index())

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
        inject_project_frontmatter(verbose=False)
        inject_series_frontmatter(verbose=False)
        project_index = build_project_index()
        series_index = build_series_index()
        for f in args.files:
            build_file(Path(f), project_index=project_index, series_index=series_index)
        build_series_landing_pages(series_index, verbose=False)
    else:
        build_all()


if __name__ == "__main__":
    main()
