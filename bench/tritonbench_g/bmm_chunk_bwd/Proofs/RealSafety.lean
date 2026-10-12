import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealLoopSafety
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealFinishSafety
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealMemory

/-! Whole-program address safety for the complete real source. Input/output
bounds concern initial-array windows at all three actual program IDs; setup,
the entire dynamic loop and the optional residual/store execute inside them. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSafety
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 18000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop
    extends RealLoopSafety.Bounds s BM BN p bounds where
  residual : p.hasResidual → ∀ i, RealMemory.active s BM BN p i →
    RealMemory.address s BM BN p p.rBatch p.rRow p.rHead p.rColumn i < bounds p.Res
  output : ∀ i, RealMemory.active s BM BN p i →
    RealMemory.address s BM BN p p.oBatch p.oRow p.oHead p.oColumn i < bounds p.Db

theorem safe (swapped : Bool) (s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hbounds : Bounds s BM BN p bounds) :
    (RealProgram.kernel swapped BM BN BCS p).toAlgKernel.TraceSafe bounds s := by
  obtain ⟨t, hp, h⟩ := RealLoop.prefix_executes swapped s BM BN BCS p
  have hb : t.regs .nat [] "pid_b" = some (Tile.scalar (s.pids 1)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "pid_b" (by decide)
  have hc : t.regs .nat [] "pid_c" = some (Tile.scalar (RealSetup.chunk s p)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "pid_c" (by decide)
  have hh : t.regs .nat [] "pid_h" = some (Tile.scalar (RealSetup.head s p)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "pid_h" (by decide)
  have hm : t.regs .nat [] "pid_m" = some (Tile.scalar (RealSetup.tileM s BN p)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "pid_m" (by decide)
  have hn : t.regs .nat [] "pid_n" = some (Tile.scalar (RealSetup.tileN s BN p)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "pid_n" (by decide)
  have hl : t.regs .nat [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limit s p)) := by
    simpa [RealSetup.result] using h.fixed .nat [] "chunk_size_limit" (by decide)
  have hs : Stmt.TraceSafeList bounds
      (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).before ++
        [.forRangeDyn "cs" (.constNat 0) (RealProgram.loopBound BCS) (.constNat 1)
          (RealProgram.realPart (RealProgram.parts swapped BM BN BCS p).body)]) s := by
    rw [RealSetup.original_candidate_setup, RealLoads.matches_source]
    apply TraceProofs.append_safe _ s (RealPureSafety.setup_safe BM BN BCS p bounds s)
    intro u he
    rw [RealSetup.executes] at he
    cases he
    apply Stmt.TraceSafeList.cons_intro (RealLoopSafety.loop_safe s BM BN BCS p bounds hbounds.toBounds)
    intro _ _
    exact Stmt.TraceSafeList.nil_intro
  unfold Kernel.TraceSafe
  rw [RealProgram.body_split]
  apply TraceProofs.append_safe _ s hs
  intro u he
  rw [hp] at he
  cases he
  rw [RealResidual.matches_source]
  exact RealFinishSafety.safe swapped t BM BN p (s.pids 1) (RealSetup.chunk s p) (RealSetup.head s p)
    (RealSetup.tileM s BN p) (RealSetup.tileN s BN p) (RealSetup.limit s p)
    (RealLoop.carry s BM BN BCS p (RealMath.blocks (RealSetup.limit s p) BCS)) bounds
    hb hc hh hm hn hl h.acc
    (fun hr i j hi hj => hbounds.residual hr (i,j,PUnit.unit) ⟨hi,hj⟩)
    (fun i j hi hj => hbounds.output (i,j,PUnit.unit) ⟨hi,hj⟩)

/-- All operations in the actual complete source belong to the shared
region-to-flat-memory bridge's supported fragment. -/
theorem flatten_ok (swapped : Bool) (BM BN BCS : Nat) (p : RealProgram.Parameters) :
    (RealProgram.kernel swapped BM BN BCS p).toAlgKernel.FlattenOk := by
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
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealSafety
