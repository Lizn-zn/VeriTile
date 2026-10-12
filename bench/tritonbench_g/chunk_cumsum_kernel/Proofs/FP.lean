import bench.tritonbench_g.chunk_cumsum_kernel.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Only the reached fp32 carry update changes. Scans and block sums retain
all backend choices; no scan decomposition or reduction reassociation is used. -/
namespace VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 3000000
set_option maxRecDepth 12000

/-- At each reached iteration, execute the loads, scan and sum before checking
that the actual carry and block total are finite. No output equation is assumed. -/
def requirementsFor (kernel : ComputeKernel) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "i_t"
    return Site.range loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "b_z") []
        (.ref .real [] "b_z") (.ref .real [] "b_zz"))).toList

section
variable (dtype out_dtype : FloatDType) (s o : RegionName) (T BT : Nat)

def original : Program where
  kernel := TypedSource.original dtype out_dtype s o T BT
  domain := requirementsFor (TypedSource.original dtype out_dtype s o T BT)

def sumLeft : Program :=
  { original dtype out_dtype s o T BT with
    kernel := TypedSource.sumLeft dtype out_dtype s o T BT }

theorem requirements_present :
    (original dtype out_dtype s o T BT).domain.length = 1 := rfl

theorem sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original dtype out_dtype s o T BT ≡[R] sumLeft dtype out_dtype s o T BT := by
  equiv_decompose
  fp_prove
end
#print_fp_assumptions sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkCumsumKernel.FPEquivProofs
