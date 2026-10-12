import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealLoop
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealAccess
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety follows the same full dynamic-loop invariant as execution.
The hypotheses bound initial-array windows, not prepared pointers or a supplied
loop run. They include the source's Dout rows up to chunk_size. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoopSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  dout : ∀ (i : Fin BM) (k : Nat), RealSetup.rows s BM BN p i < p.chunk → k < RealSetup.limit s p →
    RealSetup.dBase s p + (RealSetup.rows s BM BN p i * p.dColumn + k * p.dRow) < bounds p.Dout
  a : ∀ (k : Nat) (j : Fin BN), k < RealSetup.limit s p → RealSetup.columns s BN p j < p.K →
    RealSetup.aBase s p + (k * p.aRow + RealSetup.columns s BN p j * p.aColumn) < bounds p.A

theorem iteration_safe (initial s : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds)
    (c : Nat) (h : RealLoop.Holds initial BM BN BCS p c s) :
    Stmt.TraceSafeList bounds (RealLoads.code BM BN BCS p) (s.setReg "cs" .nat [] (Tile.scalar c)) := by
  let entry := s.setReg "cs" .nat [] (Tile.scalar c)
  have hr : entry.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows initial BM BN p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BM] "offs_m" (by decide)
  have hn : entry.regs .nat [BN] "offs_n" = some (Tile.vec (RealSetup.columns initial BN p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BN] "offs_n" (by decide)
  have hk : entry.regs .nat [BCS] "offs_cs" = some (Tile.vec Fin.val) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BCS] "offs_cs" (by decide)
  have hl : entry.regs .nat [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limit initial p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [] "chunk_size_limit" (by decide)
  have hc : entry.regs .nat [] "cs" = some (Tile.scalar c) := by simp [entry]
  have hd : entry.regs .ptr [BM, BCS] "dout_ptrs" = some (RealLoop.dPointers initial BM BN BCS p c) := by
    simpa [entry] using h.dout
  have ha : entry.regs .ptr [BCS, BN] "a_ptrs" = some (RealLoop.aPointers initial BN BCS p c) := by
    simpa [entry] using h.a
  have hsafe := RealAccess.dout_safe bounds entry BM BCS p.chunk (RealSetup.limit initial p) c
    (RealSetup.rows initial BM BN p) (RealLoop.dPointers initial BM BN BCS p c) hd hr hk hl hc
    (fun i k hi hk => hb.dout i (c * BCS + k.val) hi ((RealMath.tail_mask _ _ _ _).mp hk))
  rw [RealLoads.code]
  apply TraceProofs.assign_cons hsafe
  intro v
  apply TraceProofs.assign_cons
  · apply RealAccess.a_safe bounds _ BN BCS p.K (RealSetup.limit initial p) c
      (RealSetup.columns initial BN p) (RealLoop.aPointers initial BN BCS p c)
    · simpa using ha
    · simpa using hn
    · simpa using hk
    · simpa using hl
    · simpa using hc
    · intro k j hk hj
      exact hb.a (c * BCS + k.val) j ((RealMath.tail_mask _ _ _ _).mp hk) hj
  · intro _
    apply TraceProofs.all_safe
    simp only [List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true]
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, and_self, implies_true]

theorem loop_safe (initial : BlockState) (BM BN BCS : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds) :
    Stmt.TraceSafe bounds (RealLoop.statement BM BN BCS p) (RealSetup.result initial BM BN BCS p) := by
  rw [RealLoop.statement, Stmt.TraceSafe]
  simp only [RealProgram.loopBound, Op.SafeAt.eq_def, true_and, evalOp.eq_def,
    RealSetup.chunk_limit, bind, Option.bind_some, Tile.scalar, Tile.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub, NumericDType.div]
  apply TraceProofs.range_safe (P := RealLoop.Holds initial BM BN BCS p)
  · intro c s _ h
    exact iteration_safe initial s BM BN BCS p bounds hb c h
  · intro c s t hc h he
    obtain ⟨u, hu, hf⟩ := RealLoop.iteration initial BM BN BCS p c s hc h
    rw [he] at hu
    cases hu
    exact hf
  · exact RealLoop.initial initial BM BN BCS p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealLoopSafety
