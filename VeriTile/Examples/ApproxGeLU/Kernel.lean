/-
VeriTile.Examples.ApproxGeLU.Kernel

Split-out support for the ApproxGeLU example.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.Complex.ExponentialBounds
import Mathlib.Analysis.Real.Pi.Bounds
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import VeriTile.Math.GeluTaylor20Cert
import VeriTile.Math.RealErf
import VeriTile.Math.Tanh
import VeriTile.Core
import VeriTile.Semantics
import VeriTile.Float
import VeriTile.Frontend.Triton.DSL
import VeriTile.Examples.Common

namespace VeriTile.Examples

open VeriTile VeriTile.Math

/-! ## Embedded Triton AST -/

/-- Approximate GeLU, aligned single-block version.

Reads `blockSize` values from `xReg`, computes the tanh-style approximate
GeLU using `2 * sigmoid(2u) - 1` for `tanh(u)`, and writes the result to
`outReg`. -/
def approxGeLUKernel (xReg outReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid     := tl.program_id(0)
  offsets := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x       := tl.load($(xReg) + offsets)
  x3      := x * x * x
  c       := 7978845608028654 / 10000000000000000
  k       := 44715 / 1000000
  u       := c * (x + k * x3)
  tanh_u  := 2 * tl.sigmoid(2 * u) - 1
  y       := (1 / 2) * x * (1 + tanh_u)
  tl.store($(outReg) + offsets, y)
}

/-! ## Math Denotations -/

/-- Exact mathematical GeLU scalar function. -/
noncomputable def exactGeLUScalar (x : ℝ) : ℝ :=
  (1 / 2) * x * (1 + realErf (x / Real.sqrt 2))

/-- Tanh-style approximate GeLU scalar function implemented by the Triton kernel.

The `tanh` is represented as `2 * sigmoid(2u) - 1`, matching the embedded
DSL implementation. -/
noncomputable def approxGeLUScalar (x : ℝ) : ℝ :=
  let x3 := x * x * x
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * x3)
  (1 / 2) * x * (1 + (2 * Real.sigmoid (2 * u) - 1))

/-- Math-level denotation of the embedded approximate GeLU expression. -/
noncomputable def approxGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  approxGeLUScalar (xs i)

/-- Exact erf-based GeLU spec used as the mathematical reference. -/
noncomputable def exactGeLUSpec {N : Nat} (xs : Fin N → ℝ) (i : Fin N) : ℝ :=
  exactGeLUScalar (xs i)

/-! ## Algebraic Decomposition and Trivial Bound

Two helper lemmas that factor the approximation error into the gap between
`tanh` and `realErf`, and a uniform `|x|` bound that follows from `|tanh| ≤ 1`
and `|realErf| ≤ 1`. -/


end VeriTile.Examples
