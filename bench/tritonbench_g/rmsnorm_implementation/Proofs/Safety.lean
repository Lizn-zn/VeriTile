import bench.tritonbench_g.rmsnorm_implementation.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.RmsnormImplementation
open VeriTile.Triton
set_option maxHeartbeats 2000000

namespace Safety

theorem append_iff (bounds : RegionBounds) (before after : List Stmt) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds before s ∧
        ∀ t, stepStmts before s = some t → Stmt.TraceSafeListR .triv bounds after t := by
  induction before generalizing s with
  | nil => simp [Stmt.TraceSafeListR, stepStmts]
  | cons head tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeListR, stepStmtR_triv, stepStmts]
    cases he : stepStmt head s with
    | none => simp
    | some t => simp only [Option.bind_some, ih, and_assoc]

theorem context_congr (bounds : RegionBounds) (before left right after : List Stmt)
    (hs : ∀ s, Stmt.TraceSafeListR .triv bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (he : ∀ s, stepStmts left s = stepStmts right s) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (before ++ left ++ after) s ↔
      Stmt.TraceSafeListR .triv bounds (before ++ right ++ after) s := by
  simp only [List.append_assoc, append_iff, hs, he]

theorem range_congr (bounds : RegionBounds) (idx : RegName) (stop stride : Nat)
    (left right : List Stmt)
    (hs : ∀ s, Stmt.TraceSafeListR .triv bounds left s ↔ Stmt.TraceSafeListR .triv bounds right s)
    (he : ∀ s, stepStmts left s = stepStmts right s) (cur : Nat) (s : BlockState) :
    Stmt.forRangeTraceSafeR .triv bounds idx cur stop stride left s ↔
      Stmt.forRangeTraceSafeR .triv bounds idx cur stop stride right s := by
  rw [Stmt.forRangeTraceSafeR, Stmt.forRangeTraceSafeR]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  simp only [hz, ↓reduceIte]
  by_cases hc : cur < stop
  · simp only [hc, ↓reduceIte, hs, stepStmtsR_triv, he]
    cases stepStmts right (s.setReg idx .nat [] (Tile.scalar cur)) with
    | none => rfl
    | some t => exact and_congr_right (fun _ => range_congr bounds idx stop stride left right hs he (cur + stride) t)
  · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

theorem cast_exec (B : Nat) (s : BlockState) :
    stepStmts [.assign .real [B] "xf" (.ref .real [B] "x")] s =
      stepStmts [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] s := by
  simp [stepStmts, stepStmt, evalOp.eq_def, FloatDType.cast]
  cases s.regs .real [B] "x" <;> rfl

theorem body_exec (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    stepStmts (uncastVarBody X sxk N B) s = stepStmts (rmsVarLoopBody X sxk N B) s := by
  have hb : rmsVarLoopBody X sxk N B =
      (rmsVarLoopBody X sxk N B).take 3 ++
      [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] ++
      (rmsVarLoopBody X sxk N B).drop 4 := rfl
  conv_rhs => rw [hb]
  simp only [uncastVarBody, ProgramRewrite.append, cast_exec]

theorem body_safe (bounds : RegionBounds) (X : RegionName) (sxk N B : Nat) (s : BlockState) :
    Stmt.TraceSafeListR .triv bounds (uncastVarBody X sxk N B) s ↔
      Stmt.TraceSafeListR .triv bounds (rmsVarLoopBody X sxk N B) s := by
  have hb : rmsVarLoopBody X sxk N B =
      (rmsVarLoopBody X sxk N B).take 3 ++
      [.assign .real [B] "xf" (.castFloat .real .real (.ref .real [B] "x"))] ++
      (rmsVarLoopBody X sxk N B).drop 4 := rfl
  conv_rhs => rw [hb]
  apply context_congr bounds _ _ _ _ _ (cast_exec B)
  intro t
  simp [Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def]
  split <;> split <;> trivial

theorem uncast_safe (bounds : RegionBounds) (X W O : RegionName)
    (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.TraceSafeR .triv bounds s ↔
      (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.TraceSafeR .triv bounds s := by
  have hbody : (uncastMathKernel X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.body =
      rmsVarPreLoop sxb sxm B ++
      [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] ++
      (rmsStdPostLoop N B eps ++
      [.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody X W O sxk sw sob som sok N B)]) := by
    simp [uncastMathKernel, rmsVarPreLoop, uncastVarBody, rmsVarLoopBody,
      rmsStdPostLoop, rmsOutLoopBody, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?]
    repeat' constructor
  unfold Kernel.TraceSafeR
  rw [hbody, rmsnorm_implementation_toAlg_body]
  have hc := context_congr bounds (rmsVarPreLoop sxb sxm B)
    [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)]
    [.forRange "block_n_strart_ptr" 0 N B (rmsVarLoopBody X sxk N B)]
    (rmsStdPostLoop N B eps ++ [.forRange "block_n_strart_ptr" 0 N B (rmsOutLoopBody X W O sxk sw sob som sok N B)])
  have hs : ∀ t, Stmt.TraceSafeListR .triv bounds
      [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] t ↔
      Stmt.TraceSafeListR .triv bounds
      [.forRange "block_n_strart_ptr" 0 N B (rmsVarLoopBody X sxk N B)] t := by
    intro t
    simp only [Stmt.TraceSafeListR, Stmt.TraceSafeR]
    simp only [range_congr bounds "block_n_strart_ptr" N B _ _ (body_safe bounds X sxk N B) (body_exec X sxk N B)]
    split <;> split <;> simp
  have he : ∀ t, stepStmts [.forRange "block_n_strart_ptr" 0 N B (uncastVarBody X sxk N B)] t =
      stepStmts [.forRange "block_n_strart_ptr" 0 N B (rmsVarLoopBody X sxk N B)] t := by
    intro t
    simp only [stepStmts, stepStmt]
    rw [ProgramRewrite.range_congr _ _ _ _ _ _ (body_exec X sxk N B)]
  simpa only [List.append_assoc] using hc hs he s

theorem prefix_exec (sxb sxm : Nat) (s : BlockState) :
    stepStmts (indexPrefix Bool.true sxb sxm) s = stepStmts (indexPrefix Bool.false sxb sxm) s := by
  simp [indexPrefix, stepStmts, stepStmt, evalOp.eq_def, Tile.bop, NumericDType.mul, Nat.mul_comm]

theorem typed_safe_iff (bounds : RegionBounds) (input weight output : Storage) (v : Bool)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ) (s : BlockState) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s ↔
      (rmsnorm_implementation X W O sxb sxm sxk sw sob som sok N B eps).toAlgKernel.TraceSafeR .triv bounds s := by
  rw [typed_projection]
  apply Iff.trans _ (uncast_safe bounds X W O sxb sxm sxk sw sob som sok N B eps s)
  cases v
  · rfl
  · change (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).TraceSafeR .triv bounds s ↔
      (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).TraceSafeR .triv bounds s
    unfold Kernel.TraceSafeR
    rw [← List.take_append_drop 3 (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body,
      ← List.take_append_drop 3 (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body]
    have ht : (mathKernel Bool.true X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 =
        (mathKernel Bool.false X W O sxb sxm sxk sw sob som sok N B eps).body.drop 3 := rfl
    rw [prefix_source, prefix_source, ht, append_iff, append_iff, prefix_exec]
    have hp (v : Bool) (t : BlockState) : Stmt.TraceSafeListR .triv bounds (indexPrefix v sxb sxm) t := by
      cases v <;> simp [indexPrefix, Stmt.TraceSafeListR, Stmt.TraceSafeR, Op.SafeAtR.eq_def,
        stepStmtR_triv, stepStmt, evalOp.eq_def]
    simp only [hp, true_and]

end Safety

/-- Actual per-execution memory safety for both complete erased-real programs.
All active read/write bounds suffice; no output or intermediate value is assumed. -/
theorem typed_trace_safe (input weight output : Storage) (v : Bool) (bounds : RegionBounds)
    (X W O : RegionName) (sxb sxm sxk sw sob som sok N B : Nat) (eps : ℝ)
    (hB : 0 < B) (hXO : X ≠ O) (hWO : W ≠ O) (hsok : 0 < sok) (s : BlockState)
    (hX : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sxb + s.pids 1 * sxm + (t.val * B + j.val) * sxk < bounds X)
    (hW : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      (t.val * B + j.val) * sw < bounds W)
    (hO : ∀ (t : Fin (rmsNumSteps N B)) (j : Fin B), t.val * B + j.val < N →
      s.pids 0 * sob + s.pids 1 * som + (t.val * B + j.val) * sok < bounds O) :
    (typedKernel input weight output v X W O sxb sxm sxk sw sob som sok N B eps).eraseDType.toAlgKernel.TraceSafeR .triv bounds s :=
  (Safety.typed_safe_iff bounds input weight output v X W O sxb sxm sxk sw sob som sok N B eps s).mpr
    (legacy_trace_safe .triv bounds X W O sxb sxm sxk sw sob som sok N B eps hB hXO hWO hsok s hX hW hO)

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormImplementation
