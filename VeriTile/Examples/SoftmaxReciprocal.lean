/-
VeriTile.Examples.SoftmaxReciprocal

Real Triton optimization: replace `y = e/s` (per-element division) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ since e/s = e * (1/s) when s ≠ 0.

The "div" side of the comparison is `stableSoftmaxKernel` (already defined
in `SoftmaxEq.lean` and proven correct via `softmax_stable_correct`).
This file adds only the reciprocal-form kernel and the equivalence theorem.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Stable softmax with precomputed reciprocal. Saves N-1 divisions vs
    the per-element-divide form (`stableSoftmaxKernel`). -/
def softmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := tl.load(tl.ptr($(xReg)) + offs)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store(tl.ptr($(yReg)) + offs, y)
}

/-- Closed-form spec for `softmaxRecipKernel`'s `Y[pid*N+i]` cell. -/
noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

end VeriTile.Examples
