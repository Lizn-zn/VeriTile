/-
VeriTile architecture overview — site definition + theme.

Renders `VeriTileOverview` as the front page, with a slide-style theme that
hides the default chrome and loads `static/slides.css` + `static/slides.js`.
The JS post-processes the rendered HTML so each top-level `<h1>` introduces a
full-bleed slide with arrow-key navigation.
-/

import VersoBlog
import VeriTileOverview

open Verso Genre Blog Site Syntax

open Output Html Template Theme in
def theme : Theme := { Theme.default with
  primaryTemplate := do
    return {{
      <html lang="en">
        <head>
          <meta charset="utf-8"/>
          <meta name="viewport" content="width=device-width, initial-scale=1"/>
          <meta name="color-scheme" content="light"/>
          <title>{{ (← param (α := String) "title") }}</title>
          {{← builtinHeader }}
          <link rel="preconnect" href="https://fonts.googleapis.com"/>
          <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin="anonymous"/>
          <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"/>
          <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/themes/prism.min.css"/>
          <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-core.min.js"></script>
          <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-clike.min.js"></script>
          <script src="https://cdn.jsdelivr.net/npm/prismjs@1.29.0/components/prism-python.min.js"></script>
          <link rel="stylesheet" href="static/slides.css"/>
        </head>
        <body class="deck">
          <main id="deck-root">
            {{ (← param "content") }}
          </main>
          <nav id="deck-nav">
            <button id="prev" aria-label="Previous slide">"◀"</button>
            <span id="counter">""</span>
            <button id="next" aria-label="Next slide">"▶"</button>
          </nav>
          <script src="static/slides.js"></script>
        </body>
      </html>
    }}
  }

def veritileSite : Site := site VeriTileOverview /
  static "static" ← "static_files"
