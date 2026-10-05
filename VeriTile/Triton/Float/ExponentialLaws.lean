import VeriTile.Triton.Float.Exponential
import VeriTile.Triton.Float.SoftmaxShift

/-! The stable-softmax adapter uses the admitted libdevice candidate.
The standalone Exponential catalog does not depend on this application. -/
namespace VeriTile.Triton.FP.Exponential
open Structural Guarded ScalarArithmetic

set_option maxHeartbeats 1600000 in
theorem exp_sub {α : Type} [Inhabited α] (R : Rules)
    (M : Algebra α) (D : Domain α) (hM : Models R.assumptions M D)
    (s : State α) : SoftmaxShift.LibdeviceExpSub M D := by
  constructor
  intro a b ha hb
  let t := (s.setReg "a" .real [] (fun _ => a)).setReg "b" .real [] (fun _ => b)
  have hg : ScalarDomain D guards t := by
    intro g hg
    simp [guards] at hg
    rcases hg with rfl | rfl
    · exact ⟨fun _ => a, by simp [t], ha⟩
    · exact ⟨fun _ => b, by simp [t], hb⟩
  have h := hM lhs rhs (admitted R (.exp_sub .libdevice) (by decide)) t hg hg
  simp only [lhs, rhs, Atom.lhs, Atom.rhs, Backend.exp, fragment, run, step, evalExpr,
    evalComputeOp, evalOp_unfold, ref, minus, divide, ComputeDType.eraseDType,
    numeric, State.setReg_same, t] at h
  simp [State.setReg, SoftmaxShift.exp, sub, div] at h ⊢
  exact congrFun h PUnit.unit

end VeriTile.Triton.FP.Exponential
