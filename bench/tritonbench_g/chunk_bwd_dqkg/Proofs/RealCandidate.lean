import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealPost
import VeriTile.Triton.Math.MatrixRewrite

/-! Transfer real correctness to the dot-left candidate. Both complete typed
sources are connected to the same statement context; only the final dq add
is reversed. Real addition then gives the identical final state, preserving
all readback and frame conclusions once established for the original.
This proof uses real arithmetic, independently of the FP atom assumptions. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealCandidate
open VeriTile.Triton
set_option maxHeartbeats 6000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

section
variable (qk value hist dhist : TypedSource.Storage)
    (q k v h g dout dh dq dk dg : RegionName)
    (s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t : Nat)
    (scale : ℝ) (T K V BT BK BV NT : Nat)

/-- The real context before the final dq addition. The source split theorems
check these internal statement positions against both complete programs. -/
def before : List Stmt :=
  cbdPreLoop g T BT BK ++
    [.forRangeDyn "i_v" (.constNat 0) (cbdStopOp V BV) (.constNat 1)
      (cbdLoopBody v h dout dh s_v_h s_v_t s_h_h s_h_t T K V BT BK BV NT)] ++
    (cbdPostLoop q k dq dk dg s_k_h s_k_t scale T K BT BK).take 10

def after : List Stmt :=
  (cbdPostLoop q k dq dk dg s_k_h s_k_t scale T K BT BK).drop 11

theorem original_split :
    (RealSource.source qk value hist dhist Bool.false q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    before q k v h g dout dh dq dk dg s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT ++
      .assign .real [BT,BK] "b_dq"
        (.add .real (.consSame (.consSame .nil)) (.ref .real [BT,BK] "b_dq")
          (.dot (batch := []) (.ref .real [BT,BT] "b_ds") (.ref .real [BT,BK] "b_k"))) ::
      after q k dq dk dg s_k_h s_k_t scale T K BT BK := by
  rw [RealSource.real_body_matches, cbd_body_eq]
  rfl

theorem candidate_split :
    (RealSource.source qk value hist dhist Bool.true q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel.body =
    before q k v h g dout dh dq dk dg s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT ++
      .assign .real [BT,BK] "b_dq"
        (.add .real (.consSame (.consSame .nil))
          (.dot (batch := []) (.ref .real [BT,BT] "b_ds") (.ref .real [BT,BK] "b_k"))
          (.ref .real [BT,BK] "b_dq")) ::
      after q k dq dk dg s_k_h s_k_t scale T K BT BK := by
  cases hist <;> cases dhist
  all_goals
    simp only [RealSource.source, TypedSource.specialize, before, after,
        cbdPreLoop, cbdLoopBody, cbdPostLoop, cbdStopOp, List.take, List.drop, List.append, TypedSource.Storage.dtype,
        VeriTile.Triton.eraseDType, Region.cast, ComputeKernel.surfaceBody,
        ComputeKernel.eraseDType, chunk_bwd_dqkg_surface,
        ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
        ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
        ComputeOp.toAlgorithm?, ComputeDType.eraseDType, List.map_cons, List.map_nil,
        Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
        Op.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType,
        NumericDType.eraseDType, ComparableDType.eraseDType,
        IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]
    all_goals simp [FloatDType.toTileDType, Op.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      VeriTile.Triton.eraseDType, MemAccess.eraseDType, MaskOpt.eraseDType, List.map_cons, List.map_nil]
    all_goals cases qk <;> cases value
    all_goals simp [TypedSource.Storage.dtype, Op.eraseDType.eq_def,
      FloatDType.toTileDType, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType]

theorem candidate_executes_like (s t : BlockState)
    (he : exec (RealSource.source qk value hist dhist Bool.false q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel s = some t) :
    exec (RealSource.source qk value hist dhist Bool.true q k v h g dout dh dq dk dg
      s_k_h s_k_t s_v_h s_v_t s_h_h s_h_t scale T K V BT BK BV NT).eraseDType.toAlgKernel s = some t := by
  rw [exec, original_split] at he
  rw [exec, candidate_split]
  exact MatrixRewrite.add_commute _ _ _ _ _ s t he
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealCandidate
