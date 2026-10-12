import bench.tritonbench_g.attention_score.Proofs.RealSafetyParts
import bench.tritonbench_g.attention_score.Proofs.RealPrelude

/-! The prelude's only memory access is the selected K load. Its pointer is
derived by the already-proved initialization and the frame of the remaining
register assignments. No independently assumed pointer register is required. -/
namespace VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyPrelude
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealAccess
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3.RealRegisterSafety
set_option maxHeartbeats 1000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false
noncomputable section

theorem correct (bounds : RegionBounds) (s : BlockState) (Q K M : RegionName)
    (H HK N RN NK B D sqz sqh sqm sqk skz skh skn skk : Nat) (scale : ℝ) (evenN : Bool)
    (hbK : BlockBounds bounds (RealLoads.keyPointer K B D NK
      (s.pids 1 / H * skz + (s.pids 1 % H) / (H / HK) * skh) skn skk (s.pids 0))
      (RealLoads.checks evenN)) :
    Stmt.TraceSafeList bounds (RealProgram.prelude Q K M scale H HK N RN NK B D
      sqz sqh sqm sqk skz skh skn skk evenN) s := by
  let code := RealProgram.prelude Q K M scale H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk evenN
  let before := code.take 11
  let after := code.drop 11
  have hsplit : code = before ++ after := (List.take_append_drop 11 code).symm
  have hs : Stmt.TraceSafeList bounds before s := by
    apply list_safe
    simp only [before, code, RealProgram.prelude, List.take_succ_cons, List.take_zero,
      List.forall_mem_cons, List.not_mem_nil, false_implies, forall_const, and_true,
      assign_iff, dynamic_offsets_safe]
    simp [Op.SafeAt.eq_def]
  have hf : RealFrame.Code ["k","lo","hi","qk_scale"] after := by
    dsimp [after, code, RealProgram.prelude, RealBody.loadKey]
    repeat' first
      | apply RealFrame.Code.ifThenElse
      | apply RealFrame.Code.assign
      | apply RealFrame.Code.nil
      | simp
  obtain ⟨t,he,_,_,_,_,_,hk,_⟩ := RealPrelude.correct s Q K M H HK N RN NK B D
    sqz sqh sqm sqk skz skh skn skk 0 0 scale Bool.false Bool.false Bool.false evenN Bool.false
  change stepStmts code s = some t at he
  rw [hsplit, ProgramRewrite.append] at he
  cases hp : stepStmts before s with
  | none => simp [hp] at he
  | some u =>
      have ha : stepStmts after u = some t := by simpa only [hp, Option.bind_some] using he
      have frame := hf.frame ha
      have hku := (frame.regs .blockPtr [D,B] "K_block_ptr" (by simp)).symm.trans hk
      change Stmt.TraceSafeList bounds code s
      rw [hsplit]
      apply append_of_run hs hp
      dsimp [after, code, RealProgram.prelude]
      apply Stmt.TraceSafeList.cons_intro
        (RealSafetyParts.load_choice_safe bounds u [D,B] "K_block_ptr" "k" evenN _ hku hbK)
      intro v _
      apply list_safe
      simp [Op.SafeAt.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttentionScore.RealSafetyPrelude
