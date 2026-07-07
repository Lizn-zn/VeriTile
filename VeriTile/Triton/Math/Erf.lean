/-
VeriTile.Triton.Math.Erf

Lightweight home of the Gauss error function used by the core semantics
(`VeriTile.Triton.Semantics.TileOps` lifts it to `WithBot ℝ` for `Op.erf`).

Only the *definitions* live here — `gaussianKernel` and `realErf` — together
with their definitional unfolding lemmas. The definition needs interval
integrals (`Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic`) but none
of the heavy analysis (Gaussian integral, FTC, improper integrals) that the
theorems about `realErf` require. Those theorems stay in
`VeriTile.Math.RealErf`, which imports this file and proves properties of the
*same* definition — there is exactly one `realErf` in the codebase.

Namespace note: unlike the `VeriTile.Triton.*` siblings in this directory,
these declarations live in namespace `VeriTile.Math`. The names
`VeriTile.Math.realErf` / `VeriTile.Math.gaussianKernel` predate this split
and are referenced throughout `bench/` and `VeriTile.Examples.ApproxGeLU`;
keeping the constant names unchanged keeps every downstream proof intact.
-/

import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

namespace VeriTile.Math

/-- Pointwise integrand of the Gauss error function. We use `t * t` rather
than `t ^ 2` so the algebraic shape lines up with the kernel that appears in
`VeriTile.Examples.ApproxGeLU`. -/
noncomputable def gaussianKernel (t : ℝ) : ℝ := Real.exp (-(t * t))

lemma gaussianKernel_def (t : ℝ) :
    gaussianKernel t = Real.exp (-(t * t)) := rfl

/-- The Gauss error function:
`realErf x = (2 / √π) * ∫_0^x exp(-(t*t)) dt`. -/
noncomputable def realErf (x : ℝ) : ℝ :=
  (2 / Real.sqrt Real.pi) * ∫ t in (0)..x, gaussianKernel t

lemma realErf_def (x : ℝ) :
    realErf x = (2 / Real.sqrt Real.pi) *
      ∫ t in (0)..x, Real.exp (-(t * t)) := rfl

end VeriTile.Math
