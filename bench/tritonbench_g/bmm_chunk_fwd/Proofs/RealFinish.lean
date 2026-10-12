import bench.tritonbench_g.bmm_chunk_fwd.Proofs.RealPostlude

/-! Execute refresh, optional sequence filtering, output preparation and final
store. The metadata pointer is required only in the branch that reads it. -/
namespace VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealFinish
open VeriTile.Triton RealPostlude
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

def afterSequence (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pc base : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  if p.hasSeq then RealSequence.result s BM BN p pc base rs ns carry else s

def values (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pc base : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat)
    (carry : Fin BM → Fin BN → ℝ) (i : Fin BM) (j : Fin BN) : ℝ :=
  let length := min p.chunk (p.seqlen - pc * p.chunk)
  if p.hasSeq then
    if (RealSequence.indexValues s (Region.cast p.Seq) base p.sRow length rs (-1)).data (i,PUnit.unit) =
       (RealSequence.indexValues s (Region.cast p.Seq) base p.sRow length ns (-2)).data (j,PUnit.unit)
    then carry i j else 0
  else carry i j

theorem sequence_branch (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pc base : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hp : p.hasSeq = Bool.true → s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (hr : s.regs .nat [BM] "offs_m" = some (Tile.vec rs))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec ns))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmt (.ifThen (.constBool p.hasSeq) (RealSequence.code BM BN p)) s =
      some (afterSequence s BM BN p pc base rs ns carry) := by
  cases hs : p.hasSeq
  · simp [stepStmt, evalOp.eq_def, afterSequence, hs, bind, Option.bind]
  · have he := RealSequence.executes s BM BN p pc base rs ns carry hc (hp hs) hr hn ha
    simp [stepStmt, evalOp.eq_def, afterSequence, hs, bind, Option.bind, he]

theorem after_sequence_acc (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pc base : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    (afterSequence s BM BN p pc base rs ns carry).regs .real [BM, BN] "acc" =
      some (Tile.mat fun i j => some (values s BM BN p pc base rs ns carry i j)) := by
  cases hs : p.hasSeq <;> simp [afterSequence, values, hs, RealSequence.result, ha]

theorem after_sequence_fixed (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pc base : Nat) (rs : Fin BM → Nat) (ns : Fin BN → Nat) (carry : Fin BM → Fin BN → ℝ)
    (dtype : TileDType) (shape : TileShape) (name : RegName)
    (hn : name ∉ ["chunk_size_limit", "seq_idx_m", "seq_idx_n", "acc"]) :
    (afterSequence s BM BN p pc base rs ns carry).regs dtype shape name = s.regs dtype shape name := by
  simp only [List.mem_cons, List.not_mem_nil, or_false, not_or] at hn
  rcases hn with ⟨hcl, hm, hn, ha⟩
  cases hs : p.hasSeq <;>
    simp [afterSequence, hs, RealSequence.result, BlockState.setReg, hcl, hm, hn, ha]

def result (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn base : Nat) (carry : Fin BM → Fin BN → ℝ) : BlockState :=
  let r := refreshed s BM BN pm pn
  let v := values r BM BN p pc base (rows BM pm) (columns BN pn) carry
  let t := afterSequence r BM BN p pc base (rows BM pm) (columns BN pn) carry
  let u := outputPrepared t BM BN p pb pc ph (rows BM pm) (columns BN pn) v
  RealStore.write u p.Out BM BN (pb * p.oBatch + pc * p.oChunk + ph * p.oHead)
    p.oRow p.oColumn p.chunk (rows BM pm) (columns BN pn) v

theorem executes (s : BlockState) (BM BN : Nat) (p : RealProgram.Parameters)
    (pb pc ph pm pn base : Nat) (carry : Fin BM → Fin BN → ℝ)
    (hb : s.regs .nat [] "pid_b" = some (Tile.scalar pb))
    (hc : s.regs .nat [] "pid_c" = some (Tile.scalar pc))
    (hh : s.regs .nat [] "pid_h" = some (Tile.scalar ph))
    (hm : s.regs .nat [] "pid_m" = some (Tile.scalar pm))
    (hn : s.regs .nat [] "pid_n" = some (Tile.scalar pn))
    (hp : p.hasSeq = Bool.true → s.regs .ptr [] "seq_idx_ptr" = some (Tile.scalar (Region.cast p.Seq,base)))
    (ha : s.regs .real [BM, BN] "acc" = some (Tile.mat fun i j => some (carry i j))) :
    stepStmts (RealPostlude.code BM BN p) s =
      some (result s BM BN p pb pc ph pm pn base carry) := by
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
  have hout := output_preparation t BM BN p pb pc ph (rows BM pm) (columns BN pn) v htb htc hth htr htn hv
  have hstore := RealStore.executes u p.Out BM BN
    (pb * p.oBatch + pc * p.oChunk + ph * p.oHead) p.oRow p.oColumn p.chunk
    (rows BM pm) (columns BN pn) v (by simp [u, outputPrepared]) (by simp [u, outputPrepared])
    (by simpa [u, outputPrepared] using htr) (by simpa [u, outputPrepared] using htn)
  rw [RealPostlude.code]
  simp only [List.append_assoc, List.cons_append, List.nil_append]
  rw [stepStmts.append_some hrefresh, stepStmts.cons_some hbranch,
    stepStmts.append_some hout, stepStmts.cons_some hstore]
  exact stepStmts.nil

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BmmChunkFwd.RealFinish
