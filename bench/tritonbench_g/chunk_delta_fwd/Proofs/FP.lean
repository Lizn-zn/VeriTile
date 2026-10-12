import bench.tritonbench_g.chunk_delta_fwd.Kernels
import VeriTile.Triton.Float.RewriteTactics
import VeriTile.Meta.StatementAudit

/-! Commute only the reached outer carry update. The prefix runs the history
store and every inner corrected-value update before checking finite operands.
Dots, casts, subtractions and inner accumulation remain opaque and unchanged. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.FPEquivProofs
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 4000000
set_option maxRecDepth 16000

def requirementsFor (kernel : ComputeKernel) (BK BV : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "i_t"
    return Site.range loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "b_h") [BK, BV]
        (.ref .real [BK, BV] "b_h")
        (.ref .real [BK, BV] "b_h_cumsum"))).toList

section
variable (kd value initial final : TypedSource.Storage)
    (k v d v_new h initial_state final_state : RegionName)
    (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT : Nat)
    (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)

def original : Program where
  kernel := TypedSource.original kd value initial final k v d v_new h initial_state final_state
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
    H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
  domain := requirementsFor
    (TypedSource.original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) BK BV

def sumLeft : Program :=
  { original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE with
    kernel := TypedSource.sumLeft kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE }

theorem requirements_present :
    (original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE).domain.length = 1 := by
  cases value <;> cases final <;> rfl

theorem sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R]
    sumLeft kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE := by
  cases value <;> cases final <;> equiv_decompose <;> fp_prove
end
#print_fp_assumptions sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.FPEquivProofs
