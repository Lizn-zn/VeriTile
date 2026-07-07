/-
# `VeriTile.Triton.Float` — floating-dtype layer

Floating-dtype support for the Triton subset:

* **dtype erasure** (`Erasure`, `StateErasure`) — project annotated float
  kernels onto the Real channel so correctness reuses the Real-valued proofs;
* the **abstract rounding model** (`RoundingModel`, `EvalOpR`, `StepR`,
  `Refine`, `Pipeline`) — #447 parametric semantics (`round_real = id` the sole
  axiom, theorems ∀R) whose trivial instance recovers the exact semantics.

The correctness/refinement surfaces themselves now live one level up in
`VeriTile.Triton.Correctness`; this module re-exports them for convenience.
-/

import VeriTile.Triton.Float.Erasure
import VeriTile.Triton.Float.StateErasure
import VeriTile.Triton.Correctness
import VeriTile.Triton.Float.RoundingModel
import VeriTile.Triton.Float.EvalOpR
import VeriTile.Triton.Float.StepR
import VeriTile.Triton.Float.Refine
import VeriTile.Triton.Float.Pipeline
