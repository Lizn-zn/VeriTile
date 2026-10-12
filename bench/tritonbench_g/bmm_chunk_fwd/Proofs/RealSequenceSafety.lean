import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSequence
import VeriTile.Triton.Memory.TraceProofs

/-! Signed metadata loads access only the row/column indices selected by the
actual shortened-length masks. The branch's in-body minimum is executed before
checking the loads; no successful metadata load is an assumption. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequenceSafety
open VeriTile.Triton
noncomputable section
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

theorem index_safe (bounds : RegionBounds) (s : BlockState) (region : RegionName)
    (n base stride length : Nat) (offset : RegName) (offsets : Fin n → Nat) (sentinel : Int)
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (region,base)))
    (ho : s.regs .nat [n] offset = some (Tile.vec offsets))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hb : ∀ i, offsets i < length → base + offsets i * stride < bounds region) :
    (RealSequence.indexLoad n stride offset sentinel).SafeAt bounds s := by
  have hpv : evalOp (Op.ptrAdd .scalarL (.ref .ptr [] "seq_idx_ptr")
      (.mul .nat .scalarR (.ref .nat [n] offset) (.constNat stride))) s =
      some (Tile.vec fun i => (region,base + offsets i * stride)) := by
    simp [evalOp.eq_def, hp, ho, bind, Option.bind]
    rfl
  have hmv : evalOp (Op.lt .nat .scalarR (.ref .nat [n] offset) (.ref .nat [] "chunk_size_limit")) s =
      some (Tile.vec fun i => decide (offsets i < length)) := by
    simp [evalOp.eq_def, ho, hl, bind, Option.bind]
    rfl
  rw [RealSequence.indexLoad, Op.SafeAt.eq_def]
  refine ⟨by simp [Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def], ?_⟩
  unfold MemAccess.ActiveAddressSafe memAccessActiveAddressSafe Op.PointerAddressesSafeOn
  refine ⟨by simp [Op.MemorySafe], ?_⟩
  intro ptr he i hi
  rw [hpv] at he
  cases he
  obtain ⟨ms, he, hi⟩ := hi
  rw [hmv] at he
  cases he
  exact hb i.1 (by simpa only [Tile.vec, decide_eq_true_eq] using hi)

theorem safe (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters) (pc base : Nat)
    (rows : Fin BM → Nat) (columns : Fin BN → Nat) (bounds : RegionBounds)
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hp : s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hrow : ∀ i, rows i < min p.chunk (p.seqlen - pc * p.chunk) → base + rows i * p.sRow < bounds (Region.cast p.Seq))
    (hcol : ∀ j, columns j < min p.chunk (p.seqlen - pc * p.chunk) → base + columns j * p.sRow < bounds (Region.cast p.Seq)) :
    Stmt.TraceSafeList bounds (RealSequence.code BM BN p) s := by
  have hlimit := RealSequence.limit_eval s p.seqlen p.chunk pc hc
  rw [RealSequence.code, TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some hlimit)]
  refine ⟨by simp [Stmt.TraceSafe, RealSequence.limitOp, Op.SafeAt.eq_def], ?_⟩
  apply TraceProofs.assign_cons
  · apply index_safe bounds _ (Region.cast p.Seq) BM base p.sRow (min p.chunk (p.seqlen - pc * p.chunk)) "offs_m" rows (-1)
    · simpa using hp
    · simpa using hr
    · simp
    · exact hrow
  · intro _
    apply TraceProofs.assign_cons
    · apply index_safe bounds _ (Region.cast p.Seq) BN base p.sRow (min p.chunk (p.seqlen - pc * p.chunk)) "offs_n" columns (-2)
      · simpa using hp
      · simpa using hn
      · simp
      · exact hcol
    · intro _
      apply TraceProofs.all_safe
      simp [RealSequence.filtered, Stmt.TraceSafe, Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealSequenceSafety
