import bench.tritonbench_g.chunk_retention_ops.Proofs.FP

/-! Complete typed-source equivalences. Forward output commutes the final
fp32 scale multiplication; the other three candidates commute one fp32
carry-plus-dot addition. No precision erasure is used in this file.

Requirements inspect the actual operands after the reached source prefix:
forward state checks the decayed carry and weighted dot at each iteration;
reverse state checks its decayed carry and query/weighted-gradient dot;
forward output checks its final total and scale; backward gradients check the
final row-weighted dQ carry and decayed-score dot. They require the admitted finite operands,
with all source casts and TF32-disabled dot options retained. Experimental
shape chooses assumptions and does not restrict these symbolic dimensions.
Proofs/TypedExecution provides nonempty mixed-storage executions for all four. -/
namespace VeriTile.Bench.TritonBenchG.ChunkRetentionOps.FPEquiv
open VeriTile Triton FP.Structural FP.ProfiledRewrite
open scoped VeriTile.Spec

section ForwardState
section
variable (dotStorage : Storage) (dtype out_dtype : FloatDType) (k v h h0 ht : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (H T K V BT BK BV NT : Nat) (USE_INITIAL_STATE STORE_FINAL_STATE : Bool)
def forwardState : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE
  domain := FPProof.forwardHRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardH dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE) dotStorage BK BV BT
def forwardStateDotLeft : Program :=
  { forwardState dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE with
    kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardHDotLeft dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE }
specification forwardState_equiv (R : FP.ScalarArithmetic.Rules) :
    forwardState dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE ≡[R] forwardStateDotLeft dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE := by
  exact FPProof.forwardH.equivalent dotStorage dtype out_dtype k v h h0 ht
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t H T K V BT BK BV NT USE_INITIAL_STATE STORE_FINAL_STATE R
end
#print_fp_assumptions forwardState_equiv
end ForwardState

section ForwardOutput
section
variable (dotStorage valueStorage : Storage) (out_dtype : FloatDType) (q k v h o : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV : Nat)
def forwardOutput : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV
  domain := FPProof.forwardORequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardO dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV ) valueStorage BT BV scale
def forwardOutputScaleFirst : Program :=
  { forwardOutput dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  with
    kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.forwardOScaleFirst dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  }
specification forwardOutput_equiv (R : FP.ScalarArithmetic.Rules) :
    forwardOutput dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  ≡[R] forwardOutputScaleFirst dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV  := by
  exact FPProof.forwardO.equivalent dotStorage valueStorage out_dtype q k v h o
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV R
end
#print_fp_assumptions forwardOutput_equiv
end ForwardOutput

section BackwardState
section
variable (dotStorage : Storage) (out_dtype : FloatDType) (q do_ dh : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat)
def backwardState : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT
  domain := FPProof.backwardHRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardH dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT ) dotStorage BK BV BT
def backwardStateDotLeft : Program :=
  { backwardState dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  with
    kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardHDotLeft dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  }
specification backwardState_equiv (R : FP.ScalarArithmetic.Rules) :
    backwardState dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  ≡[R] backwardStateDotLeft dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  := by
  exact FPProof.backwardH.equivalent dotStorage out_dtype q do_ dh
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT R
end
#print_fp_assumptions backwardState_equiv
end BackwardState

section BackwardGradients
section
variable (dotStorage : Storage)  (q k v h do_ dh dq dk dv : RegionName)
  (s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t : Nat) (scale : ℝ) (H T K V BT BK BV NT : Nat)
def backwardGradients : Program where
  kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT
  domain := FPProof.backwardQKVRequirements (VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKV dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT ) dotStorage BT BK
def backwardGradientsDotLeft : Program :=
  { backwardGradients dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  with
    kernel := VeriTile.Bench.TritonBenchG.ChunkRetentionOps.backwardQKVDotLeft dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  }
specification backwardGradients_equiv (R : FP.ScalarArithmetic.Rules) :
    backwardGradients dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  ≡[R] backwardGradientsDotLeft dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT  := by
  exact FPProof.backwardQKV.equivalent dotStorage  q k v h do_ dh dq dk dv
      s_qk_h s_qk_t s_qk_d s_vo_h s_vo_t s_vo_d s_h_h s_h_t scale H T K V BT BK BV NT R
end
#print_fp_assumptions backwardGradients_equiv
end BackwardGradients

end VeriTile.Bench.TritonBenchG.ChunkRetentionOps.FPEquiv

#guard_msgs (drop info) in
#auditModuleAxioms
