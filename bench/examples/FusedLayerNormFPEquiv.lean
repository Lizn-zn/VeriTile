/- Original online-statistics versus two-pass LayerNorm. The shared affine
suffix retains its sqrt, epsilon, gamma/beta and bf16 cast. Only the admitted
scalar arithmetic and bounded count atoms are used. Empty output rows remain
covered because both sources preserve every memory cell at N = 0. -/
import bench.examples.support.LayerNormContract
import VeriTile.Triton.Float.CountConversion
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FusedLayerNormFPEquiv
open VeriTile Triton FP.Structural FP.CountConversion
open LayerNormFPContract LayerNormFPExecution
open _root_.VeriTile.Triton.FP.Equational (ReductionPlan)
open scoped VeriTile.Spec

specification layernorm_equiv (N stride : Nat) (hBound : N ≤ limit) (ε : ℝ)
    (empty : ReductionPlan 0) (R : Rules) :
    fused "x" "gamma" "beta" "y" N stride ε empty ≡[R]
      twopass "x" "gamma" "beta" "y" N stride ε empty := by
  apply Spec.FloatingPoint.ofNumerical
    (lhs := fused "x" "gamma" "beta" "y" N stride ε empty)
    (rhs := twopass "x" "gamma" "beta" "y" N stride ε empty)
    (structural := fun _ _ => False) rfl rfl
  refine ⟨by simp [fused, PrivateScratch, fusedIO, twopassIO],
    by simp [twopass, PrivateScratch, twopassIO], ?_⟩
  intro α _ M D hM plans s hd
  exact (original_runs_under_count R.arithmetic M D (arithmetic_models R M D hM) s plans
    "x" "gamma" "beta" "y" N stride ε empty (conversion R M D hM s N hBound) hd).2.2

#print_fp_assumptions layernorm_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.Examples.FusedLayerNormFPEquiv
