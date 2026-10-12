import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealPureSafety
import bench.tritonbench_g.bmm_chunk_bwd.Proofs.RealResidualSafety

/-! Safety of the complete epilogue follows its actual register states.
The optional residual and terminal store use the same shortened row mask.
No claim is made about an inactive address merely computed in a pointer tile. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealFinishSafety
open VeriTile.Triton
noncomputable section
set_option linter.unusedSimpArgs false
set_option maxHeartbeats 3000000

theorem store_safe (s : BlockState) (BM BN length K : Nat) (bounds : RegionBounds)
    (ptrs : Tile .ptr [BM, BN]) (rows : Fin BM → Nat) (columns : Fin BN → Nat)
    (hp : s.regs .ptr [BM, BN] "db_ptrs" = some ptrs)
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rows))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec columns))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (hb : ∀ i j, rows i < length → columns j < K →
      (ptrs.data (i,j,PUnit.unit)).2 < bounds (ptrs.data (i,j,PUnit.unit)).1) :
    Stmt.TraceSafe bounds (RealStore.code BM BN K) s := by
  rw [RealStore.code, Stmt.TraceSafe]
  exact ⟨by simp [MemAccess.SafeAt, Op.SafeAt.eq_def], by simp [Op.SafeAt.eq_def],
    by simp [MaskOpt.SafeAt, RealResidual.mask, Op.SafeAt.eq_def],
    RealAccess.output_address_safe bounds s BM BN K length "db_ptrs" ptrs rows columns hp hr hn hl hb⟩

theorem safe (swapped : Bool) (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn length : Nat) (carry : Fin BM → Fin BN → ℝ) (bounds : RegionBounds)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hm : s.regs .nat [] "pid_m" = some (Tile.scalar pm))
    (hn : s.regs .nat [] "pid_n" = some (Tile.scalar pn))
    (hl : s.regs .nat [] "chunk_size_limit" = some (Tile.scalar length))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)))
    (hres : p.hasResidual → ∀ i j, RealFinish.rows BM pm i < length → RealFinish.columns BN pn j < p.K →
      RealResidual.baseValue p.chunk p.rBatch p.rRow p.rHead pb pc ph +
        (RealFinish.rows BM pm i * p.rRow + RealFinish.columns BN pn j * p.rColumn) < bounds p.Res)
    (hout : ∀ i j, RealFinish.rows BM pm i < length → RealFinish.columns BN pn j < p.K →
      RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph +
        (RealFinish.rows BM pm i * p.oRow + RealFinish.columns BN pn j * p.oColumn) < bounds p.Db) :
    Stmt.TraceSafeList bounds (RealResidual.code swapped BM BN p) s := by
  let r := RealFinish.refreshed s BM BN pm pn
  let v := RealFinish.values r BM BN p pb pc ph length (RealFinish.rows BM pm) (RealFinish.columns BN pn) carry
  let t := RealFinish.afterResidual r BM BN p pb pc ph length (RealFinish.rows BM pm) (RealFinish.columns BN pn) carry
  let u := RealFinish.outputPrepared t BM BN p pb pc ph (RealFinish.rows BM pm) (RealFinish.columns BN pn) v
  have hb' : r.regs .nat [] "pid_b" = some (Tile.scalar pb) := by simpa [r, RealFinish.refreshed] using hb
  have hc' : r.regs .nat [] "pid_c" = some (Tile.scalar pc) := by simpa [r, RealFinish.refreshed] using hc
  have hh' : r.regs .nat [] "pid_h" = some (Tile.scalar ph) := by simpa [r, RealFinish.refreshed] using hh
  have hl' : r.regs .nat [] "chunk_size_limit" = some (Tile.scalar length) := by simpa [r, RealFinish.refreshed] using hl
  have ha' : r.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j)) := by
    simpa [r, RealFinish.refreshed] using ha
  have hr' : r.regs .nat [BM] "offs_m" = some (Tile.vec (RealFinish.rows BM pm)) := by simp [r, RealFinish.refreshed]
  have hn' : r.regs .nat [BN] "offs_n" = some (Tile.vec (RealFinish.columns BN pn)) := by simp [r, RealFinish.refreshed]
  have hrefresh := RealFinish.refresh_executes s BM BN pm pn hm hn
  have hbranch := RealFinish.residual_branch swapped r BM BN p pb pc ph length
    (RealFinish.rows BM pm) (RealFinish.columns BN pn) carry hb' hc' hh' hr' hn' hl' ha'
  have hfixed (dtype : TileDType) (shape : TileShape) (name : RegName)
      (hnm : name ∉ ["res_ptr", "res_ptrs", "res", "acc"]) :
      t.regs dtype shape name = r.regs dtype shape name :=
    RealFinish.after_residual_fixed r BM BN p pb pc ph length
      (RealFinish.rows BM pm) (RealFinish.columns BN pn) carry dtype shape name hnm
  have hv : t.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (v i j)) :=
    RealFinish.after_residual_acc r BM BN p pb pc ph length
      (RealFinish.rows BM pm) (RealFinish.columns BN pn) carry ha'
  have htb := (hfixed .nat [] "pid_b" (by decide)).trans hb'
  have htc := (hfixed .nat [] "pid_c" (by decide)).trans hc'
  have hth := (hfixed .nat [] "pid_h" (by decide)).trans hh'
  have htr := (hfixed .nat [BM] "offs_m" (by decide)).trans hr'
  have htn := (hfixed .nat [BN] "offs_n" (by decide)).trans hn'
  have htl := (hfixed .nat [] "chunk_size_limit" (by decide)).trans hl'
  have heout := RealFinish.output_preparation t BM BN p pb pc ph
    (RealFinish.rows BM pm) (RealFinish.columns BN pn) v htb htc hth htr htn hv
  have hsbranch : Stmt.TraceSafe bounds
      (.ifThen (.constBool p.hasResidual) (RealResidual.residual swapped BM BN p)) r := by
    rw [Stmt.TraceSafe]
    simp only [Op.SafeAt.eq_def, evalOp.eq_def, Tile.scalar, true_and]
    cases hf : p.hasResidual
    · trivial
    · exact RealResidualSafety.safe swapped r BM BN p pb pc ph length
        (RealFinish.rows BM pm) (RealFinish.columns BN pn) bounds hb' hc' hh' hr' hn' hl' (hres hf)
  have hsstore : Stmt.TraceSafe bounds (RealStore.code BM BN p.K) u := by
    apply store_safe u BM BN length p.K bounds
      (RealResidual.pointerValues p.Db BM BN (RealResidual.baseValue p.chunk p.oBatch p.oRow p.oHead pb pc ph)
        p.oRow p.oColumn (RealFinish.rows BM pm) (RealFinish.columns BN pn))
      (RealFinish.rows BM pm) (RealFinish.columns BN pn)
    · simp [u, RealFinish.outputPrepared]
    · simpa [u, RealFinish.outputPrepared] using htr
    · simpa [u, RealFinish.outputPrepared] using htn
    · simpa [u, RealFinish.outputPrepared] using htl
    · exact hout
  rw [RealFinish.source_split]
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
end VeriTile.Bench.TritonBenchG.BmmChunkBwd.RealFinishSafety
