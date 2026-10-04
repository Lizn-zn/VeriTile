import bench.examples.Welford.Kernels
/- Original online versus two-pass Welford, under admitted scalar FP atoms.
The count bound is the integer conversion rule's domain, not the GPU probe's
shape. Empty Welford rows divide by zero and are outside this FP contract. -/
import bench.examples.Welford.Contract
import VeriTile.Triton.Float.CountConversionLaws
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.WelfordFPEquiv
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton FP.Structural FP.CountConversion
open WelfordFPContract WelfordFPExecution
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec

specification welford_equiv (N stride : Nat) (hN : 0 < N) (hBound : N ≤ limit)
    (empty : ReductionPlan 0) (R : Rules) :
    online "x" "mean" "variance" N stride empty ≡[R]
      twopass "x" "mean" "variance" N stride empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := online "x" "mean" "variance" N stride empty)
    (rhs := twopass "x" "mean" "variance" N stride empty)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [online, IO₁ₓ₂PrivateScratch, onlineIO],
    by simp [twopass, IO₁ₓ₂PrivateScratch, twopassIO, onlineIO], ?_⟩
  intro α _ M D hM plans s hd
  exact (original_runs_under_count R.arithmetic M D (arithmetic_models R M D hM) s plans
    "x" "mean" "variance" N stride hN (by decide) empty
    (conversion R M D hM s N hBound) hd).2.2

#print_fp_assumptions welford_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.WelfordFPEquiv
