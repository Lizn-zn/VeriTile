/-
VeriTile.Math.GeluTaylor20PolyBound

Verified polynomial half of the midrange approximate-GeLU Taylor-20
certificate (`GeluTaylor20Cert`). Driver file: dispatches on
`x ≤ 463/200` and delegates to the four `nlinarith` case lemmas in
`GeluTaylor20PolyLowerCases` / `GeluTaylor20PolyUpperCases`.
-/

import VeriTile.Math.GeluTaylor20PolyLowerCases
import VeriTile.Math.GeluTaylor20PolyUpperCases

namespace VeriTile.Math

/-- Lower half of the Taylor polynomial bound. -/
theorem geluError_mid_taylor20_lower :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      -(50 / 100000) ≤ geluError_mid_taylor20_forCert x := by
  intro x hx
  by_cases h2 : x ≤ 463 / 200
  · exact geluError_mid_taylor20_lower_left x hx h2
  · exact geluError_mid_taylor20_lower_right x hx h2

/-- Upper half of the Taylor polynomial bound. -/
theorem geluError_mid_taylor20_upper :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      geluError_mid_taylor20_forCert x ≤ 50 / 100000 := by
  intro x hx
  by_cases h2 : x ≤ 463 / 200
  · exact geluError_mid_taylor20_upper_left x hx h2
  · exact geluError_mid_taylor20_upper_right x hx h2

/-- The Taylor polynomial itself is bounded by `5e-4` on `[83/100, 19/5]`. -/
theorem geluError_mid_taylor20_bound :
    ∀ x ∈ Set.Icc (83 / 100 : ℝ) (19 / 5),
      |geluError_mid_taylor20_forCert x| ≤ 50 / 100000 := by
  intro x hx
  exact abs_le.mpr
    ⟨geluError_mid_taylor20_lower x hx, geluError_mid_taylor20_upper x hx⟩

end VeriTile.Math
