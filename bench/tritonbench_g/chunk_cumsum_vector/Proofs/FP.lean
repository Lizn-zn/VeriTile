import bench.tritonbench_g.chunk_cumsum_vector.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! The finite domain follows the actual range, store and branch before each carry addition. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumVector.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

def requirementsFor (kernel : ComputeKernel) (BT BS : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "i_t"
    let branch ← Source.conditional? loop.body (fun _ => Bool.true)
    return Site.range loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.branch branch.before branch.condition Bool.true
        (.addFP32 (Source.prefixBefore branch.yes "b_z") [BS]
          (.ref .real [BS] "b_z")
          (.reduceSum ⟨0, by simp⟩ Bool.false (.ref .real [BT, BS] "b_s"))))).toList
section
variable (dtype out_dtype : FloatDType) (s z : RegionName)
    (s_s_h s_s_t s_s_d T S BT BS : Nat)
def original : Program where
  kernel := TypedSource.original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS
  domain := requirementsFor (TypedSource.original dtype out_dtype s z
    s_s_h s_s_t s_s_d T S BT BS) BT BS
def sumLeft : Program :=
  { original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS with
    kernel := TypedSource.sumLeft dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS }
theorem requirements_present :
    (original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS).domain.length = 1 := rfl

theorem sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS ≡[R]
    sumLeft dtype out_dtype s z s_s_h s_s_t s_s_d T S BT BS := by
  equiv_decompose
  fp_prove
end
#print_fp_assumptions sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumVector.FPEquivProofs
