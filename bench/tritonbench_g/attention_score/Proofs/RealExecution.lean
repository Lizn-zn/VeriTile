import bench.tritonbench_g.attention_score.Proofs.RealPrelude
import bench.tritonbench_g.attention_score.Proofs.RealOutput

/-! Compose initialization, the full dynamic query range and the masked store.
The result is an independent formula over initial input memory for both source
versions. Allocation safety and the flat-memory public IO bridge are separate
obligations; this theorem concerns the region-based real execution semantics. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealExecution
open VeriTile.Triton
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false
noncomputable section

theorem correct (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hB : 0 < B) :
    let c := RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate
    let base := s.pids 1 / H * soz + s.pids 1 % H * soh
    ∃ t, stepStmts (RealProgram.program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh
      H HK N RN NK B D offset size scale window complement evenM evenN candidate) s = some t ∧
      (∀ i : Fin B, t.readMem Out (base+s.pids 0*B+i.val) =
        if s.pids 0*B+i.val < NK then c.column s (RealLoop.tripCount RN B) i
        else s.readMem Out (base+s.pids 0*B+i.val)) ∧
      (∀ r a, (r ≠ Out ∨ ∀ i : Fin B, s.pids 0*B+i.val < NK →
        a ≠ base+s.pids 0*B+i.val) → t.mem r a = s.mem r a) := by
  dsimp only
  let c := RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
    offset size scale window complement evenM evenN candidate
  obtain ⟨sp,hp,ip,hlo,hhi,hz,hh,_,fp⟩ := RealPrelude.correct s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk offset size scale window complement evenM evenN candidate
  obtain ⟨sl,hl,il,fl⟩ := RealLoop.correct c s sp RN hB ip
  have hloop : stepStmt (.forRangeDyn "start_m" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat B) c.body) sp = some sl := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_ref, hlo, hhi, evalOp_constNat,
      Option.bind_some, Tile.scalar_data]
    exact hl
  have hzL := (fl.regs .nat [] "off_z" (by simp [RealIteration.writes])).trans hz
  have hhL := (fl.regs .nat [] "off_h" (by simp [RealIteration.writes])).trans hh
  obtain ⟨t,ht,hread,hframe⟩ := RealOutput.executes sl Out B NK soz soh (s.pids 1 / H)
    (s.pids 1 % H) (s.pids 0) (c.column s (RealLoop.tripCount RN B)) hzL hhL il.startN il.output
  refine ⟨t, ?_, ?_, ?_⟩
  · simp only [RealProgram.program, ProgramRewrite.append, hp, Option.bind_some]
    change stepStmt (.forRangeDyn "start_m" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat B) (RealBody.body B D N offset size window complement evenM evenN candidate)) sp = some sl at hloop
    rw [stepStmts.cons_some hloop, stepStmts.nil, Option.bind_some]
    exact ht
  · intro i
    simpa only [BlockState.readMem, fl.mem, fp.mem] using hread i
  · intro r a ha
    exact (hframe r a ha).trans (congrArg (fun mem => mem r a) (fl.mem.trans fp.mem))

/-- The complete typed source, after precision erasure, satisfies the same
initial-memory formula. BM=BN is the source accumulator's typing requirement,
and neither block size nor loop stop is fixed to an experimental shape. -/
theorem source_correct (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hBlock : BM = BN) (hB : 0 < BN) :
    let source := if candidate then TypedSource.scalarLeft Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
      else TypedSource.original Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
    let c := RealPrelude.config s Q K M H HK N RN NK BN BD sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate
    let base := s.pids 1 / H * soz + s.pids 1 % H * soh
    ∃ t, stepStmts source.eraseDType.toAlgKernel.body s = some t ∧
      (∀ i : Fin BN, t.readMem Out (base+s.pids 0*BN+i.val) =
        if s.pids 0*BN+i.val < NK then c.column s (RealLoop.tripCount RN BN) i
        else s.readMem Out (base+s.pids 0*BN+i.val)) ∧
      (∀ r a, (r ≠ Out ∨ ∀ i : Fin BN, s.pids 0*BN+i.val < NK →
        a ≠ base+s.pids 0*BN+i.val) → t.mem r a = s.mem r a) := by
  cases candidate
  all_goals
    dsimp only
    simp only [Bool.false_eq_true, ↓reduceIte, RealProgram.original_source, RealProgram.candidate_source]
    exact correct s Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
      offset size scale window complement evenM evenN _ hB

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealExecution
