import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealLoop
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealAccess
import VeriTile.Triton.Memory.TraceProofs

/-! Address safety follows the same full dynamic-loop invariant as execution.
The hypotheses bound initial-array windows, not prepared pointers or a supplied
loop run. Both input windows stop at the actual shortened chunk length. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoopSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (bounds : RegionBounds) : Prop where
  a : ∀ (i : Fin BM) (k : Nat), RealSetup.rows s BM BN p i < RealSetup.limit s p → k < p.K →
    RealSetup.aBase s p + (RealSetup.rows s BM BN p i * p.aRow + k * p.aColumn) < bounds p.A
  b : ∀ (k : Nat) (j : Fin BN), k < p.K → RealSetup.columns s BN p j < RealSetup.limit s p →
    RealSetup.bBase s p + (k * p.bColumn + RealSetup.columns s BN p j * p.bRow) < bounds p.B

theorem iteration_safe (swapped : Bool) (initial s : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds)
    (c : Nat) (h : RealLoop.Holds initial BM BN BK p c s) :
    Stmt.TraceSafeList bounds (RealLoads.code swapped BM BN BK p) (s.setReg "k" .nat [] (Tile.scalar c)) := by
  let entry := s.setReg "k" .nat [] (Tile.scalar c)
  have hr : entry.regs .nat [BM] "offs_m" = some (Tile.vec (RealSetup.rows initial BM BN p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BM] "offs_m" (by decide)
  have hn : entry.regs .nat [BN] "offs_n" = some (Tile.vec (RealSetup.columns initial BN p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BN] "offs_n" (by decide)
  have hk : entry.regs .nat [BK] "offs_k" = some (Tile.vec Fin.val) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [BK] "offs_k" (by decide)
  have hl : entry.regs .nat [] "chunk_size_limit" = some (Tile.scalar (RealSetup.limit initial p)) := by
    simpa [entry, RealSetup.result] using h.fixed .nat [] "chunk_size_limit" (by decide)
  have hc : entry.regs .nat [] "k" = some (Tile.scalar c) := by simp [entry]
  have hd : entry.regs .ptr [BM, BK] "a_ptrs" = some (RealLoop.aPointers initial BM BN BK p c) := by
    simpa [entry] using h.a
  have ha : entry.regs .ptr [BK, BN] "b_ptrs" = some (RealLoop.bPointers initial BN BK p c) := by
    simpa [entry] using h.b
  have hsafe := RealAccess.a_safe bounds entry BM BK p.K (RealSetup.limit initial p) c
    (RealSetup.rows initial BM BN p) (RealLoop.aPointers initial BM BN BK p c) hd hr hk hl hc
    (fun i k hi hk => hb.a i (c * BK + k.val) hi ((BmmChunkBwd.RealMath.tail_mask _ _ _ _).mp hk))
  rw [RealLoads.code]
  apply TraceProofs.assign_cons hsafe
  intro v
  apply TraceProofs.assign_cons
  · apply RealAccess.b_safe bounds _ BN BK p.K (RealSetup.limit initial p) c
      (RealSetup.columns initial BN p) (RealLoop.bPointers initial BN BK p c)
    · simpa using ha
    · simpa using hn
    · simpa using hk
    · simpa using hl
    · simpa using hc
    · intro k j hk hj
      exact hb.b (c * BK + k.val) j ((BmmChunkBwd.RealMath.tail_mask _ _ _ _).mp hk) hj
  · intro _
    apply TraceProofs.all_safe
    simp only [List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true]
    cases swapped <;> simp only [RealLoads.update, Bool.false_eq_true, ↓reduceIte, Stmt.TraceSafe, Op.SafeAt.eq_def, and_self, implies_true]

theorem loop_safe (swapped : Bool) (initial : BlockState) (BM BN BK : Nat) (p : RealProgram.Parameters)
    (bounds : RegionBounds) (hb : Bounds initial BM BN p bounds) :
    Stmt.TraceSafe bounds (RealLoop.statement swapped BM BN BK p) (RealSetup.result initial BM BN BK p) := by
  rw [RealLoop.statement, Stmt.TraceSafe]
  simp only [RealProgram.loopBound, Op.SafeAt.eq_def, true_and, evalOp.eq_def,
    RealSetup.chunk_limit, bind, Option.bind_some, Tile.scalar, Tile.bop,
    Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add, NumericDType.sub, NumericDType.div]
  apply TraceProofs.range_safe (P := RealLoop.Holds initial BM BN BK p)
  · intro c s _ h
    exact iteration_safe swapped initial s BM BN BK p bounds hb c h
  · intro c s t hc h he
    obtain ⟨u, hu, hf⟩ := RealLoop.iteration swapped initial BM BN BK p c s hc h
    rw [he] at hu
    cases hu
    exact hf
  · exact RealLoop.initial initial BM BN BK p

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealLoopSafety
