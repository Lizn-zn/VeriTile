/- Scalar-derived normalization and reduction boundaries. These checks do not
assert a GPU result or manufacture an EvidenceValidated proof. -/
import VeriTile.Triton.Float.ScalarReduction
import VeriTile.Meta.StatementAudit

namespace FPScalarArithmeticTests
open VeriTile Triton FP.Structural FP.Guarded
open FP.ScalarArithmetic
open FP.Equational (ReductionTree ReductionPlan)

-- Ordinary division is bound to the same tested fragments as the existing
-- reciprocal example, including its nonzero denominator guard.
example : entry .divMulRcp (by decide) = FP.Reciprocal.entry .fp32 (by decide) := rfl
example : (report .mulDistribute (by decide)).report.compute = "fp32" := rfl
example : (report .mulDistribute (by decide)).report.output = "fp32" := rfl
example : (report .mulDistribute (by decide)).report.key = FP.ReportedAdmission.fp32_mul_distrib.key := rfl
example : (report .mulRcpCancel (by decide)).guards = [⟨"a", .finite⟩, ⟨"a", .nonzero⟩] := rfl

theorem used_distribution (R : Rules) :
    Spec.Derivation R.assumptions [lhs .mulDistribute] [rhs .mulDistribute] :=
  admitted R .mulDistribute (by decide)

#print_fp_assumptions used_distribution

private def D : Domain Nat
  | .finite => fun x => x ≤ 7
  | .nonzero => fun x => x ≠ 0
  | .positive => fun x => 0 < x

example : ¬ Inputs D 2 0 0 .divMulRcp := by simp [Inputs, D]
example : ¬ Inputs D 0 0 0 .mulRcpCancel := by simp [Inputs, D]
example : Inputs D 2 1 0 .divMulRcp := by simp [Inputs, D]

private def M : Algebra Nat where
  literal := fun _ _ _ => 0
  negInf := 0
  binary := fun precision _ op a b => match op with
    | .add => a + b + if precision = some .fp64 then 100 else 0
    | .mul => a + b -- Deliberately fails distribution; no Models premise.
    | .div => if b = 0 then 17 else a / b
    | .sub => a - b
    | _ => a
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a + 1
  fromNat := fun _ a => a
  fromInt := fun _ a => a.toNat
  fp32Bits := fun a => a.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 19
  reduceSum := fun _ _ _ _ _ => 29

private def pair : ReductionTree 2 := .add (.input 0) (.input 1)

-- Finite leaves are insufficient when a partial sum overflows its domain.
example : (∀ i : Fin 2, D .finite ((fun _ => 4) i)) ∧
    ¬ FP.ScalarReduction.FiniteTree M D (fun _ => 4) 0 pair := by
  norm_num [D, pair, FP.ScalarReduction.FiniteTree, FP.ScalarReduction.value, add, M]
  decide

-- Arbitrary numerical operations do not get distribution or padding removal
-- merely because they have the same types as fp32 arithmetic.
example : FP.ScalarReduction.value M (fun _ : Fin 2 => mul M 2 3) 0 pair ≠
    mul M 2 (FP.ScalarReduction.value M (fun _ => 3) 0 pair) := by decide
example : mul M 2 (zero M) ≠ zero M := by decide

private def scheduled := FP.ScalarReduction.algebra M FP.Equational.seededSchedules

example : scheduled.reduceSum (some .fp32) (shape := [2]) ⟨0, by simp⟩ false
    (fun _ => 3) PUnit.unit = 6 := by decide
example : scheduled.reduceSum (some .fp64) (shape := [2]) ⟨0, by simp⟩ false
    (fun _ => 3) PUnit.unit = 29 := by decide
example : scheduled.reduceMax (some .fp32) (shape := [2]) ⟨0, by simp⟩ false
    (fun _ => 3) PUnit.unit = 19 := rfl
example : scheduled.cast (some .fp32) .real .bf16 3 = 4 := rfl

-- No correctness example or Real algebra supplies these numerical identities.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for n in [`VeriTile.Bench.Examples.SoftmaxStableCorrect.naiveSoftmaxKernel,
            `VeriTile.Bench.Examples.WelfordCorrect.twopassWelfordKernel] do
    if env.contains n then throwError "Scalar FP proofs imported a real-correctness example"

#axiomsClean FP.ScalarArithmetic.apply_atom
#axiomsClean FP.ScalarArithmetic.mul_zero
#axiomsClean FP.ScalarArithmetic.normalize_scale
#axiomsClean FP.ScalarReduction.factor_right
#axiomsClean FP.ScalarReduction.reduce_scale
#axiomsClean FP.ScalarReduction.normalize_reduction

end FPScalarArithmeticTests
