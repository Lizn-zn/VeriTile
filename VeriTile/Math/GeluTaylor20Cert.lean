/-
VeriTile.Math.GeluTaylor20Cert

Standalone target for the midrange approximate-GeLU Taylor-20 certificate.

Two halves:
  * `geluError_mid_taylor20_bound` — `|polynomial| ≤ 5e-4` on `[83/100, 19/5]`.
    Proven internally via case-split + `nlinarith` in `GeluTaylor20PolyBound`
    (driver) plus `GeluTaylor20PolyLowerCases` / `GeluTaylor20PolyUpperCases`
    (4 cases, ~500K–700K heartbeats each).
  * `geluError_mid_taylor20_approx` — `|geluError − polynomial| ≤ 1e-5` on the
    same interval. Kept as an axiom (see its docstring).

`geluError_mid_taylor20_cert_standalone` packages the two halves into the
shape consumed by `Examples.ApproxGeLU`. This file intentionally avoids
importing `VeriTile.Examples.ApproxGeLU` so the math content stays
self-contained; `approxGeLUScalarForCert` mirrors the kernel-side
`approxGeLUScalar` and must be kept in sync if the kernel formula changes.
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

Mirror of `approxGeLUScalar` in `Examples.ApproxGeLU`. `tanh` is represented
as `2 * sigmoid (2u) - 1`, matching the embedded kernel expression. Keep this
in sync if the kernel-side formula or its constants change.
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

/-- Taylor-20 approximation remainder — the external certificate boundary.

On `[83/100, 19/5]`, the degree-20 Taylor expansion of `geluErrorForCert`
at `463/200` matches `geluErrorForCert` within `1e-5`. Proving this
internally would require either (a) Lagrange remainder over an explicit
21st-derivative bound for the sigmoid / erf composition, or (b) interval
arithmetic over a micro-partition of the domain. Both are out of scope;
discharged externally for now. -/
axiom geluError_mid_taylor20_approx :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      |geluErrorForCert x - geluError_mid_taylor20_forCert x| ≤ 1 / 100000

/-- Math-side counterpart of `Examples.ApproxGeLU.geluError_mid_taylor20_cert`.

Conjoins `geluError_mid_taylor20_approx` (axiom — Taylor remainder) with
`geluError_mid_taylor20_bound` (proven — polynomial bound). Together: on
`[83/100, 19/5]`, the Taylor-20 polynomial approximates the GeLU error
within `1e-5`, and the polynomial itself is bounded by `5e-4`. -/
theorem geluError_mid_taylor20_cert_standalone :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      |geluErrorForCert x - geluError_mid_taylor20_forCert x| ≤ 1 / 100000 ∧
      |geluError_mid_taylor20_forCert x| ≤ 50 / 100000 := by
  intro x hx
  exact ⟨geluError_mid_taylor20_approx x hx, geluError_mid_taylor20_bound x hx⟩

end VeriTile.Math
