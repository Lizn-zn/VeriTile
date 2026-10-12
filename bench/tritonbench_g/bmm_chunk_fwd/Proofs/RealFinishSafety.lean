import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealPureSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealSequenceSafety
import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealAccess

/-! Full postlude safety follows actual refresh, optional metadata reads,
output addressing and full-chunk store. Inactive computed pointers need no
bound; active metadata and output masks retain their distinct lengths. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealFinishSafety
open VeriTile.Triton RealPostlude RealFinish
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

theorem store_safe (s : BlockState) (BM BN chunk : Nat) (bounds : RegionBounds)
    (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] "out_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hb : ∀ i j, rows i < chunk → columns j < chunk →
      (ptrs.data (i,j,PUnit.unit)).2 < bounds (ptrs.data (i,j,PUnit.unit)).1) :
    Stmt.TraceSafe bounds (RealStore.code BM BN chunk) s := by
  rw [RealStore.code, Stmt.TraceSafe]
  exact ⟨by simp [MemAccess.SafeAt, Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def],
    by simp [MaskOpt.SafeAt, RealStore.mask, Op.SafeAt.eq_def],
    RealAccess.output_address_safe bounds s BM BN chunk ptrs rows columns hp hr hn hb⟩

theorem safe (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn base : Nat) (carry : Fin BM → Fin BN → ℝ) (bounds : RegionBounds)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hm : s.regs .nat [] "pid_m" = some (Tile.scalar pm))
    (hn : s.regs .nat [] "pid_n" = some (Tile.scalar pn))
    (hp : p.hasSeq = Bool.true → s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hrow : p.hasSeq → ∀ i, rows BM pm i < min p.chunk (p.seqlen - pc * p.chunk) →
      base + rows BM pm i * p.sRow < bounds (Region.cast p.Seq))
    (hcol : p.hasSeq → ∀ j, columns BN pn j < min p.chunk (p.seqlen - pc * p.chunk) →
      base + columns BN pn j * p.sRow < bounds (Region.cast p.Seq))
    (hout : ∀ i j, rows BM pm i < p.chunk → columns BN pn j < p.chunk →
      pb * p.oBatch + pc * p.oChunk + ph * p.oHead +
        (rows BM pm i * p.oRow + columns BN pn j * p.oColumn) < bounds p.Out) :
    Stmt.TraceSafeList bounds (RealPostlude.code BM BN p) s := by
  let r := refreshed s BM BN pm pn
  let v := values r BM BN p pc base (rows BM pm) (columns BN pn) carry
  let t := afterSequence r BM BN p pc base (rows BM pm) (columns BN pn) carry
  let u := outputPrepared t BM BN p pb pc ph (rows BM pm) (columns BN pn) v
  have hb' : r.regs .nat [] "pid_b" = some (Tile.scalar pb) := by
    simpa [r, refreshed, BmmChunkBwd.RealFinish.refreshed] using hb
  have hc' : r.regs .nat [] "pid_c" = some (Tile.scalar pc) := by
    simpa [r, refreshed, BmmChunkBwd.RealFinish.refreshed] using hc
  have hh' : r.regs .nat [] "pid_h" = some (Tile.scalar ph) := by
    simpa [r, refreshed, BmmChunkBwd.RealFinish.refreshed] using hh
  have hp' : p.hasSeq = Bool.true → r.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)) := by
    intro hs
    simpa [r, refreshed, BmmChunkBwd.RealFinish.refreshed] using hp hs
  have ha' : r.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)) := by
    simpa [r, refreshed, BmmChunkBwd.RealFinish.refreshed] using ha
  have hr' : r.regs .nat [BM] "offs_m" = some (Tile.vec (rows BM pm)) := by
    simp [r, refreshed, BmmChunkBwd.RealFinish.refreshed]
  have hn' : r.regs .nat [BN] "offs_n" = some (Tile.vec (columns BN pn)) := by
    simp [r, refreshed, BmmChunkBwd.RealFinish.refreshed]
  have hrefresh := BmmChunkBwd.RealFinish.refresh_executes s BM BN pm pn hm hn
  have hbranch := sequence_branch r BM BN p pc base (rows BM pm) (columns BN pn) carry hc' hp' hr' hn' ha'
  have hfixed (dtype : TileDType) (shape : TileShape) (name : RegName)
      (hnm : name ∉ ["chunk_size_limit", "seq_idx_m", "seq_idx_n", "acc"]) :
      t.regs dtype shape name = r.regs dtype shape name :=
    after_sequence_fixed r BM BN p pc base (rows BM pm) (columns BN pn) carry dtype shape name hnm
  have hv : t.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (v i j)) :=
    after_sequence_acc r BM BN p pc base (rows BM pm) (columns BN pn) carry ha'
  have htb := (hfixed .nat [] "pid_b" (by decide)).trans hb'
  have htc := (hfixed .nat [] "pid_c" (by decide)).trans hc'
  have hth := (hfixed .nat [] "pid_h" (by decide)).trans hh'
  have htr := (hfixed .nat [BM] "offs_m" (by decide)).trans hr'
  have htn := (hfixed .nat [BN] "offs_n" (by decide)).trans hn'
  have heout := output_preparation t BM BN p pb pc ph (rows BM pm) (columns BN pn) v htb htc hth htr htn hv
  have hsbranch : Stmt.TraceSafe bounds (.ifThen (.constBool p.hasSeq) (RealSequence.code BM BN p)) r := by
    rw [Stmt.TraceSafe]
    simp only [Op.SafeAt.eq_def, evalOp.eq_def, Tile.scalar, true_and]
    cases hs : p.hasSeq
    · trivial
    · exact RealSequenceSafety.safe r BM BN p pc base (rows BM pm) (columns BN pn) bounds
        hc' (hp' hs) hr' hn' (hrow hs) (hcol hs)
  have hsstore : Stmt.TraceSafe bounds (RealStore.code BM BN p.chunk) u := by
    apply store_safe u BM BN p.chunk bounds
      (RealStore.pointerValues p.Out BM BN (pb * p.oBatch + pc * p.oChunk + ph * p.oHead)
        p.oRow p.oColumn (rows BM pm) (columns BN pn)) (rows BM pm) (columns BN pn)
    · simp [u, outputPrepared]
    · simpa [u, outputPrepared] using htr
    · simpa [u, outputPrepared] using htn
    · exact hout
  rw [RealPostlude.code]
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  apply TraceProofs.append_safe _ s (RealPureSafety.refresh_safe BM BN bounds s)
  intro r' he
  rw [hrefresh] at he
  cases he
  rw [TraceProofs.cons_of_run_iff hbranch]
  refine ⟨hsbranch, ?_⟩
  apply TraceProofs.append_safe _ t (RealPureSafety.output_preparation_safe BM BN p bounds t)
  intro u' he
  rw [heout] at he
  cases he
  apply Stmt.TraceSafeList.cons_intro hsstore
  intro _ _
  exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealFinishSafety
