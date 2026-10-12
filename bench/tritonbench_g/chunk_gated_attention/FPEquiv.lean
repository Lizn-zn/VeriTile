import bench.tritonbench_g.chunk_gated_attention.Kernels
import bench.tritonbench_g.chunk_gated_attention.Proofs.FP

/-! FP equivalence of the two complete source/candidate pairs in Kernels.
The cumulative candidate changes exact natural-number pointer arithmetic and
needs no FP atom. H commutes only its final fp32 carry/dot addition, using
add_commute after checking both actual operands are finite on every reached
iteration. Every load, cast, tl.exp, dot option, branch and store is retained.
These are conditional program equivalences, not a real or bitwise FP claim. -/
namespace VeriTile.Bench.TritonBenchG.ChunkGatedAttention.Equivalence
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section Cumulative
variable (dtype out_dtype : FloatDType) (s o : RegionName) (s_s_h s_s_t s_s_d T S BT BS : Nat)

def cumulative : Program where
  kernel := TypedSource.cumOriginal dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
  domain := []

def cumulativeIndexFirst : Program :=
  { cumulative dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS with
    kernel := TypedSource.cumIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS }

specification cumulative_equiv :
    cumulative dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS ≡[([] : Spec.Assumptions FP.GuardedFragment)]
    cumulativeIndexFirst dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS := by
  exact FPEquivProofs.cumulative_equiv dtype out_dtype s o s_s_h s_s_t s_s_d T S BT BS
end Cumulative

section Recurrent
variable (dotStorage gateStorage historyStorage : TypedSource.Storage)
    (dtype out_dtype : FloatDType) (k v g h h0 ht : RegionName)
    (s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT : Nat)
    (gateK hasSeed storeFinal : Bool)

/-- Every reached iteration executes the actual history/K/V/gate prefix before
checking the fp32 carry and TF32-disabled dot result. Intrinsic tl.exp remains
unchanged, so no exponential rewrite atom is used. -/
def recurrent : Program where
  kernel := TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
    s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal
  domain := FPEquivProofs.requirements
    (TypedSource.hOriginal dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal)
    dotStorage BK BT BV

def recurrentSumLeft : Program :=
  { recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal with
    kernel := TypedSource.hSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal }

specification recurrent_equiv (R : FP.ScalarArithmetic.Rules) :
    recurrent dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal ≡[R]
    recurrentSumLeft dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
      s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal := by
  exact FPEquivProofs.recurrent_equiv dotStorage gateStorage historyStorage dtype out_dtype k v g h h0 ht
    s_k_h s_k_t s_k_d s_v_h s_v_t s_v_d s_h_h s_h_t s_h_d T K V BT BK BV NT gateK hasSeed storeFinal R
end Recurrent

#print_fp_assumptions cumulative_equiv
#print_fp_assumptions recurrent_equiv
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkGatedAttention.Equivalence
