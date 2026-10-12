import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealLoopSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealFinishSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealMemory

/-! Whole-source safety. Bounds apply only to the selected causal tile and,
for metadata, the reached sequence-ID arm. Each loop and final load/store
retains its actual source mask; skipped tiles require no data addresses. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  input : RealSetup.active s BM BN p → RealLoopSafety.Bounds s BM BN p bounds
  rowIDs : p.hasSeq → RealSetup.active s BM BN p → ∀ i,
    RealSetup.rows s BM BN p i < RealSetup.limit s p →
    RealSetup.seqBase s p + RealSetup.rows s BM BN p i * p.sRow < bounds (Region.cast p.Seq)
  columnIDs : p.hasSeq → RealSetup.active s BM BN p → ∀ j,
    RealSetup.columns s BN p j < RealSetup.limit s p →
    RealSetup.seqBase s p + RealSetup.columns s BN p j * p.sRow < bounds (Region.cast p.Seq)
  output : ∀ i, RealMemory.active s BM BN p i → RealMemory.address s BM BN p i < bounds p.Out

theorem active_safe (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hbounds : Bounds s BM BN p bounds)
    (hactive : RealSetup.active s BM BN p = Bool.true) :
    Stmt.TraceSafeList bounds (RealExecution.activeCode swapped BM BN BK p) (RealSetup.prologueResult s BM BN p) := by
  obtain ⟨t, hp, h⟩ := RealLoop.prefix_executes swapped s BM BN BK p
  have hb : t.regs .nat [] "pid_b" = some (Tile.scalar (s.pids 1)) := by
    have hv := h.fixed .nat [] "pid_b" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hc : t.regs .nat [] "pid_c" = some (Tile.scalar (RealSetup.chunk s p)) := by
    have hv := h.fixed .nat [] "pid_c" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hh : t.regs .nat [] "pid_h" = some (Tile.scalar (RealSetup.head s p)) := by
    have hv := h.fixed .nat [] "pid_h" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hm : t.regs .nat [] "pid_m" = some (Tile.scalar (RealSetup.tileM s BN p)) := by
    have hv := h.fixed .nat [] "pid_m" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hn : t.regs .nat [] "pid_n" = some (Tile.scalar (RealSetup.tileN s BN p)) := by
    have hv := h.fixed .nat [] "pid_n" (by decide)
    cases hs : p.hasSeq <;> cases hc : p.causal <;>
      simpa [RealSetup.result, RealSetup.prologueResult, hs, hc] using hv
  have hseq : p.hasSeq = Bool.true → t.regs .ptr [] "seq_idx_ptr" =
      some (Tile.scalar (Region.cast p.Seq, RealSetup.seqBase s p)) := by
    intro hs
    simpa [RealSetup.result, hs] using h.fixed .ptr [] "seq_idx_ptr" (by decide)
  have hs : Stmt.TraceSafeList bounds
      (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).before ++
        [.forRangeDyn "k" (.constNat 0) (RealProgram.loopBound p.K BK) (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BK p).body)]) (RealSetup.prologueResult s BM BN p) := by
    rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
    apply TraceProofs.append_safe _ _ (RealPureSafety.setup_safe BM BN BK p bounds _)
    intro u he
    rw [RealSetup.executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro (RealLoopSafety.loop_safe swapped s BM BN BK p bounds (hbounds.input hactive))
    intro _ _
    exact Stmt.TraceSafeList.nil_intro
  rw [RealExecution.activeCode]
  apply TraceProofs.append_safe _ _ hs
  intro u he
  rw [hp] at he
  cases he
  rw [RealPostlude.matches_source]
  exact RealFinishSafety.safe t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
    (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.seqBase s p)
    (RealLoop.carry s BM BN BK p (BmmChunkBwd.RealMath.blocks p.K BK)) bounds hb hc hh hm hn hseq h.acc
    (fun hs => hbounds.rowIDs hs hactive) (fun hs => hbounds.columnIDs hs hactive)
    (fun i j hi hj => hbounds.output (i,j,PUnit.unit) ⟨hactive, hi, hj⟩)

theorem safe (swapped : Bool) (s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hbounds : Bounds s BM BN p bounds) :
    (RealProgram.kernel swapped BM BN BK p).toAlgKernel.TraceSafe bounds s := by
  unfold Kernel.TraceSafe
  rw [RealExecution.kernel_split]
  apply TraceProofs.append_safe _ _ (RealPureSafety.prologue_safe BM BN BK p bounds s)
  intro t he
  rw [RealSetup.prologue_executes] at he
  cases he
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    simp only [Op.SafeAt.eq_def, RealExecution.prologue_active, Tile.scalar, true_and]
    cases hactive : RealSetup.active s BM BN p
    · trivial
    · exact active_safe swapped s BM BN BK p bounds hbounds hactive
  · intro _ _
    exact Stmt.TraceSafeList.nil_intro

/-- All operations in the actual complete source belong to the shared
region-to-flat-memory bridge's supported fragment. -/
theorem flatten_ok (swapped : Bool) (BM BN BK : Nat) (p : RealProgram.Parameters) :
    (RealProgram.kernel swapped BM BN BK p).toAlgKernel.FlattenOk := by
  cases swapped
  all_goals
    simp [RealProgram.kernel, RealProgram.source, TypedSource.specialize,
      ComputeKernel.surfaceBody, ComputeKernel.eraseDType, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?_map_alg, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float,
      Kernel.FlattenOk, StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSafety
