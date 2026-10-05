import VeriTile.Triton.Float.LogExp
import VeriTile.Triton.DSL
import Mathlib.Tactic.NormNum

namespace FPLogBackendsTests
open VeriTile Triton FP FP.Structural FP.LogExp
open scoped VeriTile.Spec

/-- Separate source expressions ensure that changing log does not change exp. -/
def references : ComputeKernel := triton {
  a := tl.load($(("x" : RegionName)) + 0, dtype=tl.float32)
  out := tl.log(tl.exp(a))
  out := libdevice.log(tl.exp(a))
  out := tl.log(libdevice.exp(a))
  out := libdevice.log(libdevice.exp(a))
}

theorem reference_matrix :
    Atom.log_exp.lhs.code = references.surfaceBody[1]?.toList ∧
    Atom.log_exp_log_libdevice.lhs.code = references.surfaceBody[2]?.toList ∧
    Atom.log_exp_libdevice.lhs.code = references.surfaceBody[3]?.toList ∧
    Atom.log_exp_full_libdevice.lhs.code = references.surfaceBody[4]?.toList :=
  ⟨rfl, rfl, rfl, rfl⟩

theorem each_guarded_reference_is_unchanged :
    Atom.log_exp_elim_intrinsic.lhs = Atom.log_exp_libdevice.lhs ∧
    Atom.log_exp_elim.lhs = Atom.log_exp_full_libdevice.lhs ∧
    Atom.log_mul_split_intrinsic.lhs = Atom.log_mul.lhs ∧
    Atom.log_mul_split.lhs = Atom.log_mul_libdevice.lhs := ⟨rfl, rfl, rfl, rfl⟩

/-- Equal measured results must never let one backend's report select the other. -/
theorem reports_cannot_cross_backends :
    Atom.log_exp_elim_intrinsic.matches LogAdmission.fp32_log_exp_guarded = Bool.false ∧
    Atom.log_exp_elim.matches LogAdmission.fp32_log_exp_guarded_intrinsic = Bool.false ∧
    Atom.log_mul_split_intrinsic.matches LogAdmission.fp32_log_mul_guarded = Bool.false ∧
    Atom.log_mul_split.matches LogAdmission.fp32_log_mul_guarded_intrinsic = Bool.false := by decide

theorem selected_intrinsic_elimination (R : Rules) :
    [Atom.log_exp_elim_intrinsic.lhs] ≡[R] [Atom.log_exp_elim_intrinsic.rhs] :=
  FP.LogExp.rewrite R .log_exp_elim_intrinsic (by decide)

theorem selected_intrinsic_product (R : Rules) :
    [Atom.log_mul_split_intrinsic.lhs] ≡[R] [Atom.log_mul_split_intrinsic.rhs] :=
  FP.LogExp.rewrite R .log_mul_split_intrinsic (by decide)

noncomputable section

/-- A routing fixture, not a floating-point accuracy model. Distinct log values
make accidental use of libdevice.log on the intrinsic path observable. -/
private def model : Algebra ℚ where
  literal := fun _ _ r => if r = 0 then 0 else if r = 1 / 2 then 1 / 2
    else if r = 1 then 1 else if r = 2 then 2 else 80
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .mul => a * b
    | .add => a + b
    | _ => a - b
  unary := fun _ op a => match op with
    | .log => a + 10
    | .libdeviceLog => a + 100
    | .libdeviceExp => a + 1000
    | _ => a + 2000
  compareLt := fun _ _ => some (fun a b => decide (a < b))
  compareLe := fun _ _ => some (fun a b => decide (a ≤ b))
  cast := fun _ _ _ a => a
  fromNat := fun _ n => n
  fromInt := fun _ n => n
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

private def state : State ℚ where
  mem := fun _ _ => .mk .real 0
  regs := fun _ _ _ => none
  pids := fun _ => 0
  numPids := fun _ => 1
  undef := fun d _ _ => defaultValue d

private def evaluate (e : Op .real []) : Option ℚ :=
  (evalOp model (some .fp32) e state).map (fun t => t PUnit.unit)

set_option maxHeartbeats 1600000 in
theorem intrinsic_elimination_routes :
    evaluate (intrinsicExpression (.const 1)) = some 1 ∧
    evaluate (intrinsicExpression (.const (1 / 2))) = some (2021 / 2) ∧
    evaluate (expression (.const (1 / 2))) = some (2201 / 2) := by
  norm_num [evaluate, intrinsicExpression, expression, useIdentity, absolute,
    evalOp_unfold, numeric, numericLt, numericLe, bop, model]

set_option maxHeartbeats 1600000 in
theorem intrinsic_product_routes :
    evaluate (splitProductIntrinsic (.const 1) (.const 1)) = some 11 ∧
    evaluate (splitProduct (.const 1) (.const 1)) = some 101 ∧
    evaluate (splitProductIntrinsic (.const 2) (.const 2)) = some 24 ∧
    evaluate (splitProduct (.const 2) (.const 2)) = some 204 := by
  norm_num [evaluate, splitProductIntrinsic, splitProduct, keepProduct,
    evalOp_unfold, numeric, numericLe, bop, model]

end

#print_fp_assumptions selected_intrinsic_elimination
#print_fp_assumptions selected_intrinsic_product
#axiomsClean selected_intrinsic_elimination
#axiomsClean selected_intrinsic_product
#guard_msgs (drop info) in
#auditModuleAxioms

end FPLogBackendsTests
