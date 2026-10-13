import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormTriton.Mathematics

/-- Independent full-row scalar RMS normalization formula. -/
noncomputable def rms (N : Nat) (eps : ℝ) (x w : Fin N → ℝ) (k : Fin N) : ℝ :=
  x k * (Real.sqrt ((∑ j, (x j) ^ 2) / (N : ℝ) + eps))⁻¹ * w k

/-- The integer exponent's real interpretation is squared magnitude. This is
an exact-real identity, not an FP intrinsic-to-multiplication rewrite. -/
theorem powi_two (x : ℝ) : x ^ (2 : Int) = x * x := by
  simp [zpow_ofNat, pow_two]

/-- Positive epsilon makes the denominator positive, including the empty-row
convention. No IEEE or approximate-rsqrt law is asserted. -/
theorem denominator_pos (N : Nat) (eps : ℝ) (x : Fin N → ℝ) (heps : 0 < eps) :
    0 < Real.sqrt ((∑ j, (x j) ^ 2) / (N : ℝ) + eps) := by
  apply Real.sqrt_pos.2
  have hsum : 0 ≤ ∑ j, (x j) ^ 2 := Finset.sum_nonneg (fun j _ => sq_nonneg (x j))
  exact add_pos_of_nonneg_of_pos (div_nonneg hsum (Nat.cast_nonneg N)) heps

end VeriTile.Bench.TritonBenchG.RmsnormTriton.Mathematics

#guard_msgs (drop info) in
#auditModuleAxioms
