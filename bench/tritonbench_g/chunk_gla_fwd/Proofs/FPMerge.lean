import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete merge equivalence, using add_commute only when the source branch
is enabled and at every reached K iteration. The finite checks evaluate the
actual carry and checked load; padding, addresses and store remain unchanged. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPMerge
open VeriTile.Triton
open VeriTile FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 16000
set_option maxHeartbeats 4000000

def requirements (kernel : ComputeKernel) (BC : Nat) : List Site :=
  (do
    let branch ← Source.conditional? kernel.surfaceBody (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? branch.yes "i_k"
    return Site.branch branch.before branch.condition Bool.true
      (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
        (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
          (.ref .real [BC,BC] "b_A")
          (.load .real (.blockPtr (.ref .blockPtr [BC,BC] "p_A") [0,1]) .none .unspecified)))).toList

def originalProgram (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program where
  kernel := TypedMerge.original dtype A A2 T BT BC NK
  domain := requirements (TypedMerge.original dtype A A2 T BT BC NK) BC

def loadFirstProgram (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat) : Program :=
  { originalProgram dtype A A2 T BT BC NK with kernel := TypedMerge.loadFirst dtype A A2 T BT BC NK }

theorem merge_equiv (dtype : FloatDType) (A A2 : RegionName) (T BT BC NK : Nat)
    (R : FP.ScalarArithmetic.Rules) :
    originalProgram dtype A A2 T BT BC NK ≡[R] loadFirstProgram dtype A A2 T BT BC NK := by
  equiv_decompose; fp_prove

#print_fp_assumptions merge_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPMerge
