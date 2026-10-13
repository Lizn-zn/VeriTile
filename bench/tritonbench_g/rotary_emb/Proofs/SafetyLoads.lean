import bench.tritonbench_g.rotary_emb.Proofs.SafetyPure
import bench.tritonbench_g.rotary_emb.Proofs.KStore

namespace VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
open VeriTile.Triton
set_option maxHeartbeats 5000000
set_option maxRecDepth 12000
set_option linter.unusedSimpArgs false

macro "solveRotaryAccess" "using" h:term : tactic => `(tactic|
  simpa [Stmt.TraceSafe, Op.SafeAt.eq_def, MaskOpt.SafeAt, MaskOpt.Active,
    MaskOpt.MemorySafe, MemAccess.SafeAt, MemAccess.MemorySafe, memAccessMemorySafe,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe, Op.PointerAddressesSafeOn,
    Op.MemorySafe, evalOp.eq_def, qEvenState, kEvenState, kArithmeticState, kLoadState,
    kIndexState, qStoreState, qArithmeticState, qLoadState, qIndexState, prefixState,
    Parameters.dataOffsets, Parameters.trigOffsets, Parameters.qAddress,
    Parameters.kAddress, Parameters.activeQ, Parameters.activeK, address, trigAddress,
    sequence, head, dimension, active, range, dims, tile_elementwise, Bool.and_eq_true]
    using $h)

/-- The four trigonometric loads use the sequence mask alone; their bounds
quantify `[BLOCK_SEQ,1,D/2]`, independently of the number of data heads. -/
theorem q_loads_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) (hb : p.Bounds s bounds Q K Cos Sin) :
    Stmt.TraceSafeList bounds (qLoadBody p Q K Cos Sin) (qIndexState p s) := by
  change Stmt.TraceSafeList bounds [_,_,_,_,_,_] (qIndexState p s)
  apply TraceProofs.assign_cons
  · solveRotaryAccess using hb.1 Bool.false
  · intro v0
    apply TraceProofs.assign_cons
    · solveRotaryAccess using hb.1 Bool.true
    · intro v1
      apply TraceProofs.assign_cons
      · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.false i hi).1)
      · intro c0
        apply TraceProofs.assign_cons
        · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.false i hi).2)
        · intro s0
          apply TraceProofs.assign_cons
          · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.true i hi).1)
          · intro c1
            apply TraceProofs.assign_cons
            · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.true i hi).2)
            · intro s1
              exact Stmt.TraceSafeList.nil_intro

theorem k_loads_safe (bounds : RegionBounds) (p : Parameters)
    (Q K Cos Sin : RegionName) (s : BlockState) (hb : p.Bounds s bounds Q K Cos Sin) :
    Stmt.TraceSafeList bounds (kLoadBody p Q K Cos Sin) (kIndexState p s Q Cos Sin) := by
  change Stmt.TraceSafeList bounds [_,_,_,_,_,_] (kIndexState p s Q Cos Sin)
  apply TraceProofs.assign_cons
  · solveRotaryAccess using hb.2.1 Bool.false
  · intro v0
    apply TraceProofs.assign_cons
    · solveRotaryAccess using hb.2.1 Bool.true
    · intro v1
      apply TraceProofs.assign_cons
      · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.false i hi).1)
      · intro c0
        apply TraceProofs.assign_cons
        · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.false i hi).2)
        · intro s0
          apply TraceProofs.assign_cons
          · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.true i hi).1)
          · intro c1
            apply TraceProofs.assign_cons
            · solveRotaryAccess using (fun i hi ↦ (hb.2.2 Bool.true i hi).2)
            · intro s1
              exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RotaryEmb.FullMath
