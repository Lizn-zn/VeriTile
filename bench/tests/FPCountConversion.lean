/- Range boundaries are logical regressions, not new numerical evidence. -/
import bench.examples.WelfordFPEquiv
import bench.examples.FusedLayerNormFPEquiv
import Mathlib.Tactic.NormNum

namespace FPCountConversionTests
open VeriTile Triton FP.Structural FP.CountConversion

-- This model obeys the count laws up to limit and deliberately breaks them
-- beyond it. The admitted scalar syntax must make no claim there.
def value (n : Nat) : Nat := if n ≤ limit then n else 0

private noncomputable def model : Algebra Nat where
  literal := fun _ _ r => if r = 0 then 0 else 1
  negInf := 0
  binary := fun _ _ op a b => match op with
    | .add => a + b
    | .sub => a - b
    | .mul => a * b
    | .div => a / b
    | .max => max a b
    | .pow => a
  unary := fun _ _ a => a
  cast := fun _ _ _ a => a
  fromNat := fun _ => value
  fromInt := fun _ n => value n.toNat
  fp32Bits := fun b => b.bits.toNat
  fp32Load := id
  reduceMax := fun _ _ _ _ _ => 0
  reduceSum := fun _ _ _ _ _ => 0

theorem count_zero : value 0 = 0 := by decide

theorem in_range (i : Nat) (hi : i < limit) : value (i + 1) = value i + 1 := by
  simp [value, Nat.le_of_lt hi, Nat.succ_le_of_lt hi]

theorem last_accepted_successor : value limit = value (limit - 1) + 1 := by decide

theorem out_of_range_successor_fails : value (limit + 2) ≠ value (limit + 1) + 1 := by decide

-- Execute both actual bound fragments on the failing input. Their out-of-domain
-- result is the common zero, so the assumption does not enforce the false law.
theorem out_of_range_fragments_agree (s : State Nat) :
    run model (lhsCode .successor) (s.setReg "i" .nat [] (fun _ => limit + 1)) =
      run model (rhsCode .successor) (s.setReg "i" .nat [] (fun _ => limit + 1)) := by
  simp [lhsCode, rhsCode, FP.ScalarArithmetic.fragment, bounded, index, FP.ScalarArithmetic.plus,
    run, step, evalExpr, evalComputeOp, evalOp_unfold, numeric, natLt, bop, model,
    value, limit, FP.CountAdmission.upperExclusive]
  rfl

-- Correct counterparts must not be imported to establish the FP claims.
open Lean Elab Command in
run_cmd do
  let env ← getEnv
  for name in [
      `VeriTile.Bench.Examples.WelfordCorrect.twopassWelfordKernel,
      `VeriTile.Bench.Examples.FusedLayerNormCorrect.twoPassLayerNormKernel] do
    if env.contains name then throwError "FP proof imported Correct: {name}"

#axiomsClean FP.CountConversion.conversion
#axiomsClean VeriTile.Bench.Examples.WelfordFPEquiv.welford_equiv
#axiomsClean VeriTile.Bench.Examples.FusedLayerNormFPEquiv.layernorm_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end FPCountConversionTests
