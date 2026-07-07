/-
VeriTile.Triton.Float

Floating-dtype support for the Triton subset. This parent module re-exports
the erasure layer, the float-facing correctness bridge, and the abstract
rounding-model layer (#447: parametric semantics whose trivial instance
recovers the exact semantics).
-/

import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.StateErasure
import VeriTile.Triton.Float.Correctness
import VeriTile.Triton.Float.RoundingModel
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.StepR
import VeriTile.Triton.Float.Refine
import VeriTile.Triton.Float.Pipeline
