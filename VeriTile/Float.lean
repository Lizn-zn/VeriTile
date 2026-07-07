/-
VeriTile.Float

Floating-dtype support for the Triton subset. This parent module re-exports
the erasure layer, the float-facing correctness bridge, and the abstract
rounding-model layer (#447: parametric semantics whose trivial instance
recovers the exact semantics).
-/

import VeriTile.Float.Erasure
import VeriTile.Float.StateErasure
import VeriTile.Float.Correctness
import VeriTile.Float.RoundingModel
import VeriTile.Float.EvalOpR
import VeriTile.Float.StepR
import VeriTile.Float.Refine
