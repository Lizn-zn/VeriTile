import VeriTile.Triton.Math.Softmax
import Mathlib.Tactic.Ring
import VeriTile.Meta.StatementAudit

/-! Independent scalar formulas. No interpreter, memory state or tile reduction
occurs in these definitions. The source-to-formula bridge is separate. -/
namespace VeriTile.Bench.TritonBenchG.LogSoftmax.Math
open scoped BigOperators

noncomputable def forward {N : Nat} (x : Fin N → ℝ) (i : Fin N) : ℝ :=
  Real.log (Real.exp (x i) / ∑ j, Real.exp (x j))

theorem denominator_positive {N : Nat} (x : Fin N → ℝ) (i : Fin N) :
    0 < ∑ j, Real.exp (x j) :=
  Finset.sum_pos (fun _ _ ↦ Real.exp_pos _) ⟨i, Finset.mem_univ i⟩

/-- Stable evaluation has the same real result for every finite shift. -/
theorem forward_shift {N : Nat} (x : Fin N → ℝ) (m : ℝ) (i : Fin N) :
    Real.log (Real.exp (x i - m) / ∑ j, Real.exp (x j - m)) = forward x i := by
  unfold forward
  congr 1
  exact congrFun (VeriTile.Triton.TiledSoftmax.naive_eq_stable x m).symm i

theorem forward_logsumexp {N : Nat} (x : Fin N → ℝ) (i : Fin N) :
    forward x i = x i - Real.log (∑ j, Real.exp (x j)) := by
  rw [forward, Real.log_div (Real.exp_ne_zero _) (ne_of_gt (denominator_positive x i)),
    Real.log_exp]

/-- The scalar expression written by the backward kernel. -/
noncomputable def backward (N : Nat) (y dy : Nat → ℝ) (i : Nat) : ℝ :=
  dy i - Real.exp (y i) * ∑ j ∈ Finset.range N, dy j

/-- Padded reduction length is explicit because the source loads omit `other`. -/
noncomputable def backwardPadded (P : Nat) (y dy : Nat → ℝ) (i : Nat) : ℝ :=
  dy i - Real.exp (y i) * ∑ j ∈ Finset.range P, dy j

theorem backwardPadded_eq (N tail : Nat) (y dy : Nat → ℝ) (i : Nat) :
    backwardPadded (N + tail) y dy i = backward N y dy i -
      Real.exp (y i) * ∑ j ∈ Finset.range tail, dy (N + j) := by
  unfold backwardPadded backward
  rw [Finset.sum_range_add]
  ring

theorem backwardPadded_eq_of_zero (N tail : Nat) (y dy : Nat → ℝ) (i : Nat)
    (hpad : ∀ j < tail, dy (N + j) = 0) :
    backwardPadded (N + tail) y dy i = backward N y dy i := by
  rw [backwardPadded_eq]
  have hz : (∑ j ∈ Finset.range tail, dy (N + j)) = 0 :=
    Finset.sum_eq_zero (fun j hj ↦ hpad j (Finset.mem_range.mp hj))
  rw [hz, mul_zero, sub_zero]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.LogSoftmax.Math
