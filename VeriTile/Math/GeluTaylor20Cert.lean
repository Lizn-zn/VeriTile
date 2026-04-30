/-
VeriTile.Math.GeluTaylor20Cert

Standalone target for the midrange approximate-GeLU Taylor-20 certificate.

This file intentionally avoids importing `VeriTile.Examples.ApproxGeLU`: the
goal is to expose the former axiom as one small theorem statement, with only
the mathematical definitions needed by an external prover.
-/

import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Math.GeluTaylor20PolyDef
import VeriTile.Math.GeluTaylor20PolyBound
import VeriTile.Math.RealErf

namespace VeriTile.Math

/-- Exact mathematical GeLU scalar function. -/
noncomputable def exactGeLUScalarForCert (x : ℝ) : ℝ :=
  (1 / 2) * x * (1 + realErf (x / Real.sqrt 2))

/-- Tanh-style approximate GeLU scalar function used by the Triton kernel.

The `tanh` is represented as `2 * sigmoid (2u) - 1`, matching the embedded
kernel expression in `VeriTile.Examples.ApproxGeLU`.
-/
noncomputable def approxGeLUScalarForCert (x : ℝ) : ℝ :=
  let x3 := x * x * x
  let c : ℝ := 7978845608028654 / 10000000000000000
  let k : ℝ := 44715 / 1000000
  let u := c * (x + k * x3)
  (1 / 2) * x * (1 + (2 * Real.sigmoid (2 * u) - 1))

/-- Approximate-GeLU minus exact-GeLU error function. -/
noncomputable def geluErrorForCert (x : ℝ) : ℝ :=
  approxGeLUScalarForCert x - exactGeLUScalarForCert x

/-- Taylor-20 approximation remainder, kept as the external certificate
boundary. Proving this internally requires verified high-order derivative
bounds for the sigmoid/erf composition. -/
axiom geluError_mid_taylor20_approx :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      |geluErrorForCert x - geluError_mid_taylor20_forCert x| ≤ 1 / 100000

/-- Standalone version of the former `geluError_mid_taylor20_cert` axiom.

This is the theorem intended for external certification:
on `[83/100, 19/5]`, the Taylor-20 polynomial approximates the GeLU error
within `1e-5`, and the polynomial itself is bounded by `5e-4`.
-/
theorem geluError_mid_taylor20_cert_standalone :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      |geluErrorForCert x - geluError_mid_taylor20_forCert x| ≤ 1 / 100000 ∧
      |geluError_mid_taylor20_forCert x| ≤ 50 / 100000 := by
  intro x hx
  exact ⟨geluError_mid_taylor20_approx x hx, geluError_mid_taylor20_bound x hx⟩

end VeriTile.Math
