import bench.tritonbench_g.chunk_delta_fwd.Kernels
import bench.tritonbench_g.chunk_delta_fwd.Proofs.FP

/-! FP equivalence of the complete kernels in Kernels. Only the outer carry
add is commuted. Its fp32 operands must be finite at each reached update,
after the history store and ALL inner-loop iterations. The source's dots,
casts, subtractions, padding and stores stay in context. This is conditional
equivalence under add_commute, not concrete IEEE bit equality. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

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
  domain := FPEquivProofs.requirementsFor
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

specification chunk_sum_left_equiv (R : FP.ScalarArithmetic.Rules) :
    original kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R]
    sumLeft kd value initial final k v d v_new h initial_state final_state
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
      H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE := by
  exact FPEquivProofs.sum_left_equiv kd value initial final k v d v_new h initial_state final_state
    s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t
    H T K V BT BC BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE R
end
#print_fp_assumptions chunk_sum_left_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.Equivalence
