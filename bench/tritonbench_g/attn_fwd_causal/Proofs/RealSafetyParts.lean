import bench.tritonbench_g.attn_fwd_causal.Proofs.RealAccessSafety
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealRegisterSafety
import bench.tritonbench_g.attention_score.Proofs.RealFrame

/-! Register-only source fragments and the helper load prefix. Safety follows
the actual branch structure; arithmetic outputs need no numerical assumptions.
The frame certificate preserves the pointers needed by the later value load. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyParts
open VeriTile.Triton RealLoads RealMemory RealAccessSafety
open AttentionFwdTriton3.RealAccess AttentionFwdTriton3.RealRegisterSafety
open AttentionScore
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def middle (BM BN BD : Nat) (second diagonal : Bool) : List Stmt :=
  [RealBody.score BM BN BD] ++
    (if second then RealBody.diagonal BM BN "mask" ++ RealBody.center BM BN
     else [.ifThenElse (.constBool diagonal)
       (RealBody.diagonal BM BN "mask_first" ++ RealBody.center BM BN) (RealBody.center BM BN)]) ++
    [RealBody.probability BM BN] ++ RealBody.rescale BM BN BD

def middleWrites : List RegName := ["qk","mask","mask_first","m_ij","p","l_ij","alpha","l_i","acc"]

theorem middle_frame (BM BN BD : Nat) (second diagonal : Bool) :
    RealFrame.Code middleWrites (middle BM BN BD second diagonal) := by
  cases second
  all_goals
    dsimp [middle, RealBody.score, RealBody.diagonal, RealBody.center, RealBody.probability, RealBody.rescale]
    repeat' first
      | apply RealFrame.Code.ifThenElse
      | apply RealFrame.Code.assign
      | apply RealFrame.Code.nil
      | simp [middleWrites]

theorem middle_safe (BM BN BD : Nat) (second diagonal : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (middle BM BN BD second diagonal) s := by
  apply list_safe
  cases second <;> simp [middle, RealBody.score, RealBody.scoreExpr, RealBody.diagonal,
    RealExpressions.maskExpr, RealExpressions.penaltyExpr, RealBody.center, RealBody.centerExpr,
    RealBody.probability, RealBody.rescale, Op.SafeAt.eq_def]

theorem tail_safe (BM BN BD HD : Nat) (second : Bool) (bounds : RegionBounds) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealBody.accumulate BM BN BD ++
      RealBody.advance (keyName second) (scaleName second) (valueName second) BN BD HD) s := by
  apply list_safe
  simp [RealBody.accumulate, RealBody.advance, Op.SafeAt.eq_def]

theorem source_split (BM BN BD N HA HD : Nat) (second diagonal : Bool) :
    RealLoop.sourceBody BM BN BD N HA HD second diagonal =
      RealBody.loadPrefix (keyName second) (scaleName second) BD BN N HA ++
      middle BM BN BD second diagonal ++
      [RealBody.loadValue (valueName second) BN BD N HA] ++
      (RealBody.accumulate BM BN BD ++ RealBody.advance (keyName second) (scaleName second) (valueName second) BN BD HD) := by
  cases second <;> simp [RealLoop.sourceBody, RealBody.first, RealBody.second,
    RealBody.suffix, middle, keyName, scaleName, valueName, List.append_assoc]

theorem loads_safe (bounds : RegionBounds) (s : BlockState) (c : Config) (BN BD start : Nat)
    (second : Bool)
    (hp : s.regs .ptr [BD,BN] (keyName second) = some (keyPointer c BD BN start))
    (hsp : s.regs .ptr [] (scaleName second) = some (keyScalePointer c BN start))
    (hn : s.regs .nat [BN] "offs_n" = some (Tile.vec (fun i => i.val)))
    (hs : s.regs .nat [] "start_n" = some (Tile.scalar start))
    (hbK : ∀ (j : Fin BN) (d : Fin BD), start+j.val < c.context → d.val < c.activeHead →
      c.base+d.val+j.val*c.strideKN+c.headDim*start < bounds c.K)
    (hbS : c.keyScaleBase+start/BN < bounds c.KS) :
    Stmt.TraceSafeList bounds (RealBody.loadPrefix (keyName second) (scaleName second) BD BN c.context c.activeHead) s := by
  let sn := s.setReg "start_n" .nat [] (Tile.scalar start)
  let sm := sn.setReg "k_mask" .bool [BD,BN] (keyMask BD BN c.context c.activeHead start)
  have hm := eval_key_mask sn BD BN c.context c.activeHead start (by simp [sn, hn]) (by simp [sn])
  unfold RealBody.loadPrefix
  apply cons_of_run (by simp [Stmt.TraceSafe, Op.SafeAt.eq_def])
    (stepStmt_assign_eq_some (by simpa only [evalOp_ref] using hs))
  apply cons_of_run (by simp [Stmt.TraceSafe, RealBody.keyMask, Op.SafeAt.eq_def])
    (stepStmt_assign_eq_some hm)
  apply TraceProofs.assign_cons
    (key_safe bounds sm c BN BD start (keyName second)
      (by cases second <;> simpa [sm, sn, keyName] using hp) (by simp [sm]) hbK)
  intro tile
  apply Stmt.TraceSafeList.cons_intro
  · rw [Stmt.TraceSafe]
    exact key_scale_safe bounds (sm.setReg "k" .real [BD,BN] tile) c BN start (scaleName second)
      (by cases second <;> simpa [sm, sn, scaleName] using hsp) hbS
  · intro _ _; exact Stmt.TraceSafeList.nil_intro

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSafetyParts
