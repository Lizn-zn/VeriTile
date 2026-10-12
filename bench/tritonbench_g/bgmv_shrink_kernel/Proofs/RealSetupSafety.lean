import bench.tritonbench_g.bgmv_shrink_kernel.Proofs.RealAccessSafety

/-! Signed-metadata bounds and the register-only setup. -/
namespace VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSafety
open VeriTile.Triton RealSetupSource RealSetup
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

theorem prelude_safe (bounds : RegionBounds) (p : Parameters) (initial : BlockState)
    (hb : initial.pids 1 < bounds (Region.cast p.indices)) :
    Stmt.TraceSafeListR .triv bounds p.prelude initial := by
  unfold Parameters.prelude
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s1 hs1 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 0) initial = some (Tile.scalar (initial.pids 0)))] at hs1
  obtain rfl := Option.some.inj hs1
  refine Stmt.TraceSafeListR.cons_intro (by simp [Stmt.TraceSafeR, Op.SafeAtR.eq_def]) (fun s2 hs2 => ?_)
  rw [stepStmtR_triv, stepStmt_assign_eq_some (by simp :
    evalOp (Op.programId 1) _ = some (Tile.scalar (initial.pids 1)))] at hs2
  obtain rfl := Option.some.inj hs2
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  simp only [Stmt.TraceSafeR, Op.SafeAtR.eq_def, true_and,
    MemAccess.ActiveAddressSafeR, memAccessActiveAddressSafeR, MaskOpt.ActiveR, evalOpR_triv, evalOp_ref]
  intro offsets hoff i _
  simp only [BlockState.setReg_same] at hoff
  obtain rfl := Option.some.inj hoff
  exact hb


theorem setup_safe (bounds : RegionBounds) (p : Parameters) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds p.setup s := by
  apply Stmt.TraceSafeListR.of_forall
  intro st hs t
  simp only [Parameters.setup, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl
  all_goals simp [Stmt.TraceSafeR, Parameters.weightPointer, Op.SafeAtR.eq_def]

theorem sentinel_safe (bounds : RegionBounds) (p : Parameters) (swapped : Bool) (s : BlockState)
    (hm : s.readMemValue .int (Region.cast p.indices) (s.pids 1) = -1)
    (hmeta : s.pids 1 < bounds (Region.cast p.indices)) :
    Stmt.TraceSafeListR .triv bounds (p.body swapped) s := by
  unfold Parameters.body
  refine Stmt.TraceSafeListR.append_intro _ _ (prelude_safe bounds p s hmeta) (fun t ht => ?_)
  rw [stepStmtsR_triv, prelude_exec p s (-1) hm] at ht
  obtain rfl := Option.some.inj ht
  refine Stmt.TraceSafeListR.cons_intro ?_ (fun _ _ => Stmt.TraceSafeListR.nil_intro)
  have hg : evalOp Parameters.guard (afterPrelude s (-1)) = some (Tile.scalar Bool.false) :=
    guard_eval _ (-1) (by simp [afterPrelude])
  simp only [Stmt.TraceSafeR, Parameters.guard, Op.SafeAtR.eq_def, true_and, evalOpR_triv]
  change (match evalOp Parameters.guard (afterPrelude s (-1)) with
    | some v => if v.data PUnit.unit then Stmt.TraceSafeListR .triv bounds
        (p.setup ++ p.loop :: p.tail swapped) (afterPrelude s (-1)) else True
    | none => True)
  rw [hg]
  trivial

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BgmvShrinkKernel.RealSetupSafety
