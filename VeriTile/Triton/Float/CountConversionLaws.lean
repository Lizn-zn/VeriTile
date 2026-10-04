import VeriTile.Triton.Float.CountConversion
import VeriTile.Triton.Float.WelfordInduction

/-! Instantiate the selected count candidates for the recurrence proof.
The integer bound remains explicit in every successor application. -/
namespace VeriTile.Triton.FP.CountConversion
open Structural Guarded ScalarArithmetic

set_option maxHeartbeats 1600000 in
theorem zero {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α) :
    M.fromNat (some .fp32) 0 = ScalarArithmetic.zero M := by
  have h := hM (lhs .zero) (rhs .zero) (admitted R .zero (by decide)) s
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ComputeDType.eraseDType] at h
  simp [State.setReg, ScalarArithmetic.zero] at h ⊢
  exact congrFun h PUnit.unit

set_option maxHeartbeats 1600000 in
theorem successor {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (i : Nat) (hi : i < limit) :
    M.fromNat (some .fp32) (i + 1) = add M (M.fromNat (some .fp32) i) (one M) := by
  let t := s.setReg "i" .nat [] (fun _ => i)
  have h := hM (lhs .successor) (rhs .successor) (admitted R .successor (by decide)) t
    (by simp [ScalarDomain, lhs]) (by simp [ScalarDomain, rhs])
  simp only [lhs, rhs, lhsCode, rhsCode, fragment, bounded, index, plus, run, step,
    evalExpr, evalComputeOp, evalOp_unfold, ComputeDType.eraseDType, t, State.setReg_same] at h
  simp [State.setReg, numeric, natLt, bop, hi, ScalarArithmetic.add, ScalarArithmetic.one] at h ⊢
  exact congrFun h PUnit.unit

/-- Every conversion needed by the original N-step recurrence, with the
admitted upper bound retained. This is derived from two scalar atoms. -/
theorem conversion {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D) (s : State α)
    (N : Nat) (hN : N ≤ limit) : WelfordInduction.CountConversion M N :=
  ⟨zero R M D hM s, fun i hi => successor R M D hM s i (lt_of_lt_of_le hi hN)⟩

end VeriTile.Triton.FP.CountConversion
