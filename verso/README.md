# VeriTile architecture overview — Verso source

Renders the architecture overview as a self-contained HTML "deck" using
[Verso](https://github.com/leanprover/verso). Each top-level `# Heading`
inside `VeriTileOverview.lean` becomes a slide in the rendered output;
arrow keys / `PageUp` / `PageDown` / `Space` navigate, `f` toggles
fullscreen.

This is an **isolated sub-project** that pins its own Lean toolchain
(`v4.30.0-rc2`, matching Verso `main`) — independent of the parent
VeriTile project's `v4.29.0`. It does **not** import any VeriTile Lean
code; the architecture content is plain Verso markdown with custom
block components.

## Layout

```text
docs/verso/
├── lakefile.toml                 ← Lake project + verso dep
├── lean-toolchain                ← v4.30.0-rc2
├── VeriTileOverview.lean         ← #doc (Page) "..." => …  (the deck content)
├── VeriTileSite.lean             ← Site + slide-style theme
├── Main.lean                     ← `blogMain` entrypoint
└── static_files/
    ├── slides.css                ← Slide presentation CSS
    └── slides.js                 ← Arrow-key navigation
```

## Build

From `docs/verso/`:

```bash
# First time: fetch Verso (large download, takes a few minutes)
lake update
lake build

# Render the site to ./_site/
lake exe veritile-overview --output _site
```

Then open `_site/index.html` in a browser. Arrow keys / `PageUp` /
`PageDown` / `Space` advance slides; press `f` to toggle fullscreen.

## Iterating

Edit `VeriTileOverview.lean` for content changes — section-level
structure (each `# Heading`) controls slide breaks. For visual /
layout tweaks, edit `static_files/slides.css`. For navigation
behaviour, edit `static_files/slides.js`. Re-run `lake build` after
changing the `.lean` files; static files are copied at site-render
time, so you only need to re-run the renderer.

## Custom block components

Defined at the top of `VeriTileOverview.lean`:

| Directive       | Renders to                              |
|-----------------|-----------------------------------------|
| `:::cardBlue`   | `<div class="card card-blue">…</div>`   |
| `:::cardOrange` | `<div class="card card-orange">…</div>` |
| `:::cardGreen`  | `<div class="card card-green">…</div>`  |
| `:::cardPurple` | `<div class="card card-purple">…</div>` |
| `:::cardMuted`  | `<div class="card card-muted">…</div>`  |
| `:::cols`       | `<div class="cols cols-2">…</div>`      |
| `:::cols3`      | `<div class="cols cols-3">…</div>`      |
| `:::pipeline`   | `<div class="pipeline">…</div>`         |
| `:::numgrid`    | `<div class="numgrid">…</div>`          |

Inside a card, the first `**bold sentence**` becomes the labelled cap
("INPUT" / "WHAT WE DID" / "OUTPUT") via CSS.

## Notes

- Verso's `lean4:autoprove` style highlighter requires the snippet to
  be a real Lean program. The snippets in this deck are illustrative
  and use unfenced ` ``` ` code blocks (no language tag) so Verso
  emits them as plain `<pre><code>`.
- If Verso's site theme ever changes its top-level wrapper class, the
  CSS rule `header, footer { display: none; }` may need to be adjusted.
- The earlier remark.js single-file deck lives at
  `docs/VeriTile_Architecture_Overview.html` — same content, no Lean
  toolchain required.
