import bench.tritonbench_g.chunk_gla_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Complete inter-subblock equivalence. Finite checks apply to the actual
 carry and dot result after the whole iteration prefix, inside both enabled
 branches and the full K loop. Only add_commute is used; no exponential
 identity, reassociation, projection or output equation is assumed. -/

namespace VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPInter
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 18000
set_option maxHeartbeats 8000000

def requirements (kernel : ComputeKernel) (BC BK : Nat) : List Site :=
  (do
    let outer ← Source.conditional? kernel.surfaceBody (fun code =>
      (Source.conditional? code (fun body => (Source.dynamicRange? body "i_k").isSome)).isSome)
    let inner ← Source.conditional? outer.yes (fun body => (Source.dynamicRange? body "i_k").isSome)
    let loop ← Source.dynamicRange? inner.yes "i_k"
    return Site.branch outer.before outer.condition Bool.true
      (.branch inner.before inner.condition Bool.true
        (.range loop.before "i_k" loop.start loop.stop loop.stride loop.body
          (.addFP32 (Source.prefixBefore loop.body "b_A") [BC,BC]
            (.ref .real [BC,BC] "b_A")
            (.dot (batch := []) (.ref .real [BC,BK] "b_qg") (.ref .real [BK,BC] "b_kg") {}))))).toList
section
variable (qs ks gs : TypedInter.Storage) (out : FloatDType) (q k g A : RegionName)
  (s_k_h s_k_t : Nat) (scale : ℝ) (T K BT BC BK NC : Nat)
def original : Program where
  kernel := TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC
  domain := requirements (TypedInter.original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC) BC BK
def dotFirst : Program :=
  { original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC with
    kernel := TypedInter.dotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC }
theorem inter_equiv (R : FP.ScalarArithmetic.Rules) :
    original qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC ≡[R]
    dotFirst qs ks gs out q k g A s_k_h s_k_t scale T K BT BC BK NC := by
  cases qs <;> cases ks <;> cases gs <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions inter_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGlaFwd.FPInter
