import bench.tritonbench_g.attention_score.Proofs.RealSafetyPrelude
import bench.tritonbench_g.attention_score.Proofs.RealSafetyLoop
import bench.tritonbench_g.attention_score.Proofs.RealSafetyOutput

/-! Complete real-source trace safety from actual active-address bounds. The
query and normalizer ranges use the proved trip count, while K is loaded once
even if the query range is empty. Output allocation is required only for the
source store mask. No numerical equation is a safety premise. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealSafety
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
set_option maxHeartbeats 1000000
noncomputable section

structure Bounds (bounds : RegionBounds) (c : RealLoop.Config) (stop : Nat)
    (Out : RegionName) (baseOut : Nat) : Prop where
  query : ∀ j, j < RealLoop.tripCount stop c.B →
    BlockBounds bounds (RealLoads.queryPointer c.Q c.B c.D c.N c.baseQ c.sqm c.sqk (j*c.B))
      (RealLoads.checks c.evenM)
  key : BlockBounds bounds (RealLoads.keyPointer c.K c.B c.D c.NK c.baseK c.skn c.skk c.startN)
    (RealLoads.checks c.evenN)
  normalizer : ∀ row, row < RealLoop.tripCount stop c.B*c.B → c.baseM+row < bounds c.M
  output : ∀ i : Fin c.B, c.startN*c.B+i.val < c.NK → baseOut+c.startN*c.B+i.val < bounds Out

theorem correct (bounds : RegionBounds) (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK B D : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hB : 0 < B)
    (hb : Bounds bounds (RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate) RN Out
      (s.pids 1 / H * soz + s.pids 1 % H * soh)) :
    Stmt.TraceSafeList bounds (RealProgram.program Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh
      H HK N RN NK B D offset size scale window complement evenM evenN candidate) s := by
  let c := RealPrelude.config s Q K M H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk
    offset size scale window complement evenM evenN candidate
  obtain ⟨sp,hp,ip,hlo,hhi,hz,hh,_,_⟩ := RealPrelude.correct s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk offset size scale window complement evenM evenN candidate
  obtain ⟨sl,hl,il,fl⟩ := RealLoop.correct c s sp RN hB ip
  have hloop : stepStmt (.forRangeDyn "start_m" (.ref .nat [] "lo") (.ref .nat [] "hi")
      (.constNat B) c.body) sp = some sl := by
    simp only [stepForRangeAux.forRangeDyn_unfold, evalOp_ref, hlo, hhi, evalOp_constNat,
      Option.bind_some, Tile.scalar_data]
    exact hl
  have safeLoop : Stmt.TraceSafe bounds (.forRangeDyn "start_m" (.ref .nat [] "lo")
      (.ref .nat [] "hi") (.constNat B) c.body) sp := by
    simp only [Stmt.TraceSafe, Op.SafeAt.eq_def, evalOp_ref, hlo, hhi, evalOp_constNat,
      Tile.scalar_data, true_and]
    exact RealSafetyLoop.correct bounds c s sp RN hB ip hb.query hb.normalizer
  have hzL := (fl.regs .nat [] "off_z" (by simp [RealIteration.writes])).trans hz
  have hhL := (fl.regs .nat [] "off_h" (by simp [RealIteration.writes])).trans hh
  simp only [RealProgram.program, List.append_assoc, List.singleton_append]
  apply append_of_run (RealSafetyPrelude.correct bounds s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk scale evenN hb.key) hp
  apply cons_of_run safeLoop hloop
  exact RealSafetyOutput.correct bounds sl Out B NK soz soh (s.pids 1 / H) (s.pids 1 % H)
    (s.pids 0) hzL hhL il.startN hb.output

theorem source_safe (bounds : RegionBounds) (s : BlockState) (Q K M Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN : Nat)
    (offset size : Int) (scale : ℝ) (window complement evenM evenN candidate : Bool)
    (hBlock : BM = BN) (hB : 0 < BN)
    (hb : Bounds bounds (RealPrelude.config s Q K M H HK N RN NK BN BD sqz sqh sqm sqk skz skh skn skk
      offset size scale window complement evenM evenN candidate) RN Out
      (s.pids 1 / H * soz + s.pids 1 % H * soh)) :
    let source := if candidate then TypedSource.scalarLeft Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
      else TypedSource.original Q K M Out
        sqz sqh sqm sqk skz skh skn skk soz soh son Z H HK N RN NK BM BD BN
        offset size scale window complement evenM evenN hBlock
    source.eraseDType.toAlgKernel.TraceSafe bounds s := by
  cases candidate
  all_goals
    dsimp only
    simp only [Bool.false_eq_true, ↓reduceIte, Kernel.TraceSafe,
      RealProgram.original_source, RealProgram.candidate_source]
    exact correct bounds s Q K M Out sqz sqh sqm sqk skz skh skn skk soz soh H HK N RN NK BN BD
      offset size scale window complement evenM evenN _ hB hb

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealSafety
