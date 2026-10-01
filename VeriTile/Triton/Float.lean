/-
# `VeriTile.Triton.Float` — floating-dtype layer

Floating-dtype support for the Triton subset:

* **dtype erasure** (`Erasure`, `StateErasure`) — project annotated float
  kernels onto the Real channel so correctness reuses the Real-valued proofs;
* the **abstract rounding model** (`RoundingModel`, `EvalOpR`, `StepR`,
  `Refine`, `Pipeline`) — parametric semantics with identity on reals and
  idempotent rounding, whose trivial instance recovers the exact semantics;
* **concrete software values** (`BitValue`, `ScalarOps`) — format-indexed
  bf16/fp32 bits and explicitly rounded scalar operations, separate from the
  Real-valued kernel evaluators. No hardware conformance is asserted.

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
import VeriTile.Triton.Float.BitValue
import VeriTile.Triton.Float.ScalarOps
import VeriTile.Triton.Float.Equivalence
