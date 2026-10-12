import bench.tritonbench_g.bgmv_shrink_kernel.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Full typed-source equivalence at final fp32 scaling, including store and
atomic-add tails. The actual accumulator and scaling must be finite after the
complete non-sentinel prefix and K loop. No rule rewrites a reduction or atomic
addition. Successful execution and model witnesses are separate obligations. -/

namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 2200000
set_option maxRecDepth 10000

def requirementsFor (kernel : ComputeKernel) (BN : Nat) (scaling : ℝ) : List Site :=
  (do
    let active ← Source.conditional? kernel.surfaceBody
      (fun body => decide ((Source.prefixBefore body "accumulator" 1).length < body.length))
    return Site.branch active.before active.condition Bool.true
      (.mulScalarFP32 (Source.prefixBefore active.yes "accumulator" 1)
        BN [] (.ref .real [BN] "accumulator") (.const scaling))).toList

section
variable (storage : TypedSource.Storage) (out_dtype : FloatDType)
    (x w out : RegionName) (N K : Nat) (indices : Region .int) (scaling : ℝ)
    (sx sk sw0 swn swk sy syn BN BK splits : Nat)

def original : Program where
  kernel := TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits
  domain := requirementsFor (TypedSource.original storage out_dtype x w out N K indices scaling
    sx sk sw0 swn swk sy syn BN BK splits) BN scaling

def scalarLeft : Program :=
  { original storage out_dtype x w out N K indices scaling sx sk sw0 swn swk sy syn BN BK splits with
    kernel := TypedSource.scalarLeft storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits }

theorem requirements_present :
    (original storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits).domain.length = 1 := by
  cases storage <;> rfl

theorem scaling_equiv (R : FP.ScalarArithmetic.Rules) :
    original storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits ≡[R]
    scalarLeft storage out_dtype x w out N K indices scaling
      sx sk sw0 swn swk sy syn BN BK splits := by
  cases storage
  all_goals equiv_decompose
  all_goals fp_prove

end
#print_fp_assumptions scaling_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.FPEquivProofs
