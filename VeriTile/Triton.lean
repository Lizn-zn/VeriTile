/-
VeriTile.Triton

Umbrella prelude for the whole Triton subset: the AST/core, operational
semantics, memory layer, kernel-proof lemmas, the correctness/refinement
surfaces, the floating-dtype and rounding-model layer, the DSL front-end, the
pure-math operator library, and the grid-launch / concurrency layers.

`import VeriTile.Triton` pulls the entire library. Kernel transcriptions and
worked examples import this single module in place of the individual
`VeriTile.Triton.*` sub-umbrellas.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory
import VeriTile.Triton.Kernel
import VeriTile.Triton.Correctness
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.Math
import VeriTile.Triton.Launch
import VeriTile.Triton.Concurrency
