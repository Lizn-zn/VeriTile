/-
HELD-OUT BENCHMARK FILE — Phase A LLM tool eval target.

Replicates `VeriTile.Examples.softmax_naive_correct` from
`VeriTile/Examples/SoftmaxEq.lean` but with the proof body replaced
by `sorry`. The MAIN LIBRARY copy stays intact and proven; this file
is the LLM proof-drafting evaluation target.

DO NOT add this file to the main `VeriTile.lean` import graph.
DO NOT modify the proof body during development; the LLM tool runs
end-to-end on this file with no human edit (per PLAN.md §LLM
benchmark protocol).

To run the eval: scripts/prove.sh bench/llm_eval/softmax_naive_correct_held_out.lean
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples.HeldOut

open VeriTile.Triton VeriTile.Examples

/-- Held-out copy of `softmax_naive_correct` for LLM benchmark.

    The original is in `VeriTile/Examples/SoftmaxEq.lean` and is fully
    proven. This copy has the proof body replaced by `sorry`. -/
theorem softmax_naive_correct_held_out
    (N : Nat) (_hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (naiveSoftmaxKernel N) s) N s.pid i
        = some (naiveSpec xs i) := by
  sorry

end VeriTile.Examples.HeldOut
