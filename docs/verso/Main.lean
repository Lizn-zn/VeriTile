/-
VeriTile architecture overview — Verso build entrypoint.

Build:
    lake build
    ./.lake/build/bin/veritile-overview --output _site

Then open `_site/index.html` (or `_site/VeriTile-Architecture-Overview/index.html`,
depending on Verso routing).
-/

import VersoBlog
import VeriTileSite

open Verso Genre Blog

def linkTargets : Code.LinkTargets TraverseContext := {}

def main := blogMain theme veritileSite (linkTargets := linkTargets)
