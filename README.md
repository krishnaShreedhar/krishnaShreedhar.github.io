# krishnaShreedhar.github.io

Personal site of **Shreedhar Kodate** — ML Engineer, learner, seeker.

Live at [krishnashreedhar.github.io](https://krishnashreedhar.github.io)

---

## Site Structure

```
.
├── index.html              # Home page (hero, about, experience, contact)
├── experience/index.html   # Detailed experience page
├── resources.html          # Curated reading / resource list
├── blogs/
│   ├── index.html          # Blog hub
│   ├── technical/          # ML, transformers, paper reviews
│   ├── growth/             # Learning, attention, personal development
│   ├── leadership/         # Engineering culture, team dynamics
│   └── community/          # Gratitude, reflections
├── css/
│   ├── base.css            # Reset / typography base
│   ├── main.css            # Core layout and components
│   ├── blog.css            # Blog design tokens and post styles
│   └── home.css            # Home page cosmic/warm redesign
├── js/                     # Vendor and custom scripts
├── fonts/                  # Local web fonts
├── images/                 # Site imagery
├── markdowns/              # Source Markdown files for blog posts
├── templates/              # HTML template for new posts (new-post.md)
├── py/
│   ├── build.py            # Static site generator
│   └── requirements.txt    # Python dependencies
└── data/                   # Structured content data
```

---

## Blog Post Workflow

Blog posts are written in Markdown and compiled to HTML via `py/build.py`.

**1. Create a Markdown file**

```
markdowns/<category>/<slug>.md
```

Include a YAML frontmatter block (see existing files for reference). The `output:` field specifies the destination HTML path.

**2. Build the post**

```bash
pip install -r py/requirements.txt

# Build a single post
python py/build.py markdowns/<category>/<slug>.md

# Build everything
python py/build.py

# Watch for changes
python py/build.py --watch
```

**3. Link the post**

Add a post-card entry to `blogs/<category>/index.html` pointing to the new file.

---

## CSS Architecture

| File | Purpose |
|------|---------|
| `base.css` | Reset, global typography |
| `main.css` | Core layout, nav, shared components |
| `blog.css` | Design tokens, reading styles, code blocks, callouts |
| `home.css` | Home page sections — hero, about, contact |

Dark/light theme is toggled via the moon/sun button; state persists in `localStorage`.

---

## Features

- Dark / light theme toggle
- Section ribbon (right-side nav dots on home page)
- Reading progress bar on blog posts
- Markdown → HTML build pipeline with:
  - Syntax-highlighted code blocks with language labels
  - Footnotes (`[^N]`)
  - Callout blocks (`> [!NOTE]`, `> [!TIP]`, `> [!WARN]`)
  - Math blocks (`$$...$$`)
  - Mermaid diagram support

---

## License

Built on the [Kards](http://www.styleshout.com/) template (CC BY 3.0). Original template credits in `readme.txt`.
