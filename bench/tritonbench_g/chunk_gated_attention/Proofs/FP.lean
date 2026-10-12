import bench.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.cache_transform_triton.Proofs.IndexRewrite
import VeriTile.Meta.StatementAudit

/-! Complete source equivalence. Cumulative preprocessing changes only exact
integer address multiplication, so it uses no floating atom. The recurrent
candidate commutes the final fp32 carry/dot addition at every reached chunk.
Its finite-operand domain is checked after the history store, K/V loads and
the actual gate branch. Intrinsic tl.exp, casts and dot order remain opaque.
Neither proof assumes a final-output equation or a supplied carried state. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.FPEquivProofs
section Cumulative
open VeriTile Triton FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxRecDepth 12000
set_option maxHeartbeats 2000000
variable (dtype out_dtype : FloatDType) (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)
def cumulative : Program where
  kernel := TypedSource.cumOriginal dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := []
def cumulativeIndexFirst : Program where
  kernel := TypedSource.cumIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := []
theorem cumulative_equiv :
    cumulative dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    cumulativeIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS := by
  equiv_decompose <;> fp_prove
end Cumulative

open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec
set_option maxHeartbeats 8000000
set_option maxRecDepth 18000

section Recurrent
def blockDot (storage : TypedSource.Storage) (BK BT BV : Nat) : Op .real [BK,BV] :=
  match storage with
  | .fp32 => .dot (batch := []) (.ref .real [BK,BT] "b_k") (.ref .real [BT,BV] "b_v")
      { allowTF32 := some Bool.false }
  | .fp16 => .castFloat .fp32 .real (.dotTyped .fp16 .fp16 .fp32 (batch := [])
      (.ref .fp16 [BK,BT] "b_k") (.ref .fp16 [BT,BV] "b_v") { allowTF32 := some Bool.false })
  | .bf16 => .castFloat .fp32 .real (.dotTyped .bf16 .bf16 .fp32 (batch := [])
      (.ref .bf16 [BK,BT] "b_k") (.ref .bf16 [BT,BV] "b_v") { allowTF32 := some Bool.false })

def requirements (kernel : ComputeKernel) (storage : TypedSource.Storage) (BK BT BV : Nat) : List Site :=
  (do
    let loop ← Source.dynamicRange? kernel.surfaceBody "i_t"
    return Site.range loop.before "i_t" loop.start loop.stop loop.stride loop.body
      (.addFP32 (Source.prefixBefore loop.body "b_h") [BK,BV]
        (.ref .real [BK,BV] "b_h") (blockDot storage BK BT BV))).toList

variable (dotStorage gateStorage historyStorage : TypedSource.Storage)
    (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT : Nat)
    (gateK useSeed storeFinal : Bool)

def recurrent : Program where
  kernel := TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
    s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal
  domain := requirements
    (TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal)
    dotStorage BK BT BV

def recurrentSumLeft : Program :=
  { recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal with
    kernel := TypedSource.hSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal }

theorem domain_present :
    (recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal).domain.length = 1 := by
  cases dotStorage <;> cases gateStorage <;> cases historyStorage <;> rfl

theorem recurrent_equiv (R : FP.ScalarArithmetic.Rules) :
    recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal ≡[R]
    recurrentSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK useSeed storeFinal := by
  cases dotStorage <;> cases gateStorage <;> cases historyStorage <;> equiv_decompose <;> fp_prove

end Recurrent

#print_fp_assumptions cumulative_equiv
#print_fp_assumptions recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.FPEquivProofs
