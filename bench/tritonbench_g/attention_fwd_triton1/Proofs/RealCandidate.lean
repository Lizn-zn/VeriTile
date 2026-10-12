import bench.tritonbench_g.attention_fwd_triton1.Proofs.RealLoop

/-! Mathematical correctness of the scalar-left candidate. This proof uses
real multiplication in the erased interpreter and does not import or assume
the separately admitted FP multiplication relation. -/
namespace VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealCandidate
open VeriTile.Triton
set_option linter.unusedSimpArgs false
set_option maxRecDepth 10000

private theorem scale_commutes (s : BlockState) (BT BD : Nat) (scale : ℝ) :
    evalOp (.mul .real .scalarL (.const scale) (.ref .real [BT, BD] "b_q")) s =
      evalOp (.mul .real .scalarR (.ref .real [BT, BD] "b_q") (.const scale)) s := by
  simp only [evalOp_mul, evalOp_const, evalOp_ref]
  cases hq : s.regs .real [BT, BD] "b_q" with
  | none => rfl
  | some q =>
    simp only [Option.bind_eq_bind, Option.bind_some]
    congr 1
    ext i
    change WithBot.realMul (some scale) (q.data i) = WithBot.realMul (q.data i) (some scale)
    cases q.data i using Option.rec <;> simp [WithBot.realMul, Option.map₂, mul_comm]

private theorem contextual_scale (before after : List Stmt)
    (BT BD : Nat) (scale : ℝ) (s : BlockState) :
    stepStmts (before ++ .assign .real [BT, BD] "b_q"
      (.mul .real .scalarL (.const scale) (.ref .real [BT, BD] "b_q")) :: after) s =
    stepStmts (before ++ .assign .real [BT, BD] "b_q"
      (.mul .real .scalarR (.ref .real [BT, BD] "b_q") (.const scale)) :: after) s := by
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts, stepStmt, scale_commutes]
  | cons stmt tail ih =>
    simp only [List.cons_append, stepStmts]
    cases stepStmt stmt s with
    | none => rfl
    | some st => exact ih st

theorem body_same_execution (q k v h o : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState) :
    stepStmts (RealProjection.scalarLeftBody q k v h o s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND) s =
    stepStmts (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND) s := by
  have hsplit : aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND =
      (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).take 7 ++
        .assign .real [BT, BD] "b_q"
          (.mul .real .scalarR (.ref .real [BT, BD] "b_q") (.const scale)) ::
        (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).drop 8 := rfl
  conv_rhs => rw [hsplit]
  simpa only [RealProjection.scalarLeftBody, List.append_assoc, List.singleton_append] using
    contextual_scale
      ((aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).take 7)
      ((aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).drop 8)
      BT BD scale s

private theorem range_same_execution (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hbody : ∀ s, stepStmts lhs s = stepStmts rhs s) (s : BlockState) :
    stepForRangeAux index cur stop stride lhs s = stepForRangeAux index cur stop stride rhs s := by
  by_cases hz : stride = 0
  · subst stride
    simp only [stepForRangeAux.step_zero_step]
  · have hs : 0 < stride := Nat.pos_of_ne_zero hz
    by_cases hc : cur < stop
    · rw [stepForRangeAux.step_lt hz hc, stepForRangeAux.step_lt hz hc, hbody]
      cases stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some st =>
        exact range_same_execution index (cur + stride) stop stride lhs rhs hbody st
    · rw [stepForRangeAux.step_ge hz (by omega), stepForRangeAux.step_ge hz (by omega)]
termination_by stop - cur
decreasing_by omega

/-- Complete real executions agree, including both stores and failures. -/
theorem same_execution (q k v h o : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState) :
    exec (TypedSource.scalarLeft q k v h o s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s =
    exec (TypedSource.original q k v h o s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s := by
  rw [RealProjection.original_matches_transcription]
  simp only [exec, RealProjection.scalarLeft_matches_body, attention_fwd_triton1_body_splitG]
  have hloop (st : BlockState) :
      stepStmt (.forRangeDyn "i" (.constNat 0) (aft1StopOpG BT NT) (.constNat 1)
        (RealProjection.scalarLeftBody q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)) st =
      stepStmt (.forRangeDyn "i" (.constNat 0) (aft1StopOpG BT NT) (.constNat 1)
        (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)) st := by
    simp only [stepForRangeAux.forRangeDyn_unfold]
    congr 1
    funext a
    congr 1
    funext b
    congr 1
    funext c
    exact range_same_execution _ _ _ _ _ _
      (body_same_execution q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND) st
  generalize aft1PrologueG BD = before
  induction before generalizing s with
  | nil => simp only [List.nil_append, stepStmts, hloop]
  | cons stmt tail ih =>
    simp only [List.cons_append, stepStmts]
    cases stepStmt stmt s with
    | none => rfl
    | some st => exact ih st

private theorem contextual_scale_safe (bounds : RegionBounds) (before after : List Stmt)
    (BT BD : Nat) (scale : ℝ) (s : BlockState) :
    Stmt.TraceSafeList bounds (before ++ .assign .real [BT, BD] "b_q"
      (.mul .real .scalarL (.const scale) (.ref .real [BT, BD] "b_q")) :: after) s ↔
    Stmt.TraceSafeList bounds (before ++ .assign .real [BT, BD] "b_q"
      (.mul .real .scalarR (.ref .real [BT, BD] "b_q") (.const scale)) :: after) s := by
  induction before generalizing s with
  | nil =>
    simp only [List.nil_append, Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
      stepStmt, scale_commutes]
  | cons stmt tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeList]
    cases stepStmt stmt s with
    | none => rfl
    | some st => exact and_congr_right (fun _ => ih st)

theorem body_same_safety (bounds : RegionBounds) (q k v h o : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState) :
    Stmt.TraceSafeList bounds (RealProjection.scalarLeftBody q k v h o s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND) s ↔
    Stmt.TraceSafeList bounds (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale
      BT BD NT STORE IFCOND) s := by
  have hsplit : aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND =
      (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).take 7 ++
        .assign .real [BT, BD] "b_q"
          (.mul .real .scalarR (.ref .real [BT, BD] "b_q") (.const scale)) ::
        (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).drop 8 := rfl
  conv_rhs => rw [hsplit]
  simpa only [RealProjection.scalarLeftBody, List.append_assoc, List.singleton_append] using
    contextual_scale_safe bounds
      ((aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).take 7)
      ((aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND).drop 8)
      BT BD scale s

private theorem range_same_safety (bounds : RegionBounds) (index : RegName) (cur stop stride : Nat)
    (lhs rhs : List Stmt) (hbody : ∀ s, stepStmts lhs s = stepStmts rhs s)
    (hsafe : ∀ s, Stmt.TraceSafeList bounds lhs s ↔ Stmt.TraceSafeList bounds rhs s)
    (s : BlockState) :
    Stmt.forRangeTraceSafe bounds index cur stop stride lhs s ↔
      Stmt.forRangeTraceSafe bounds index cur stop stride rhs s := by
  rw [Stmt.forRangeTraceSafe, Stmt.forRangeTraceSafe]
  by_cases hz : stride = 0
  · simp only [hz, ↓reduceIte]
  · simp only [hz, ↓reduceIte]
    by_cases hc : cur < stop
    · simp only [hc, ↓reduceIte, hsafe, hbody]
      cases stepStmts rhs (s.setReg index .nat [] (Tile.scalar cur)) with
      | none => rfl
      | some st =>
        exact and_congr_right (fun _ =>
          range_same_safety bounds index (cur + stride) stop stride lhs rhs hbody hsafe st)
    · simp only [hc, ↓reduceIte]
termination_by stop - cur
decreasing_by omega

/-- The candidate introduces no new memory access or address condition. -/
theorem same_safety (bounds : RegionBounds) (q k v h o : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState) :
    Kernel.TraceSafe bounds (TypedSource.scalarLeft q k v h o s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s ↔
    Kernel.TraceSafe bounds (TypedSource.original q k v h o s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s := by
  rw [RealProjection.original_matches_transcription]
  simp only [Kernel.TraceSafe, RealProjection.scalarLeft_matches_body,
    attention_fwd_triton1_body_splitG]
  have hloop (st : BlockState) :
      Stmt.TraceSafe bounds (.forRangeDyn "i" (.constNat 0) (aft1StopOpG BT NT) (.constNat 1)
        (RealProjection.scalarLeftBody q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)) st ↔
      Stmt.TraceSafe bounds (.forRangeDyn "i" (.constNat 0) (aft1StopOpG BT NT) (.constNat 1)
        (aft1LoopBodyG q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)) st := by
    simp only [Stmt.TraceSafe]
    apply and_congr_right; intro _
    apply and_congr_right; intro _
    apply and_congr_right; intro _
    cases evalOp (.constNat 0) st <;> cases evalOp (aft1StopOpG BT NT) st <;>
      cases evalOp (.constNat 1) st
    all_goals try exact Iff.rfl
    exact range_same_safety bounds _ _ _ _ _ _
      (body_same_execution q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND)
      (body_same_safety bounds q k v h o s_qh s_hh s_ht scale BT BD NT STORE IFCOND) st
  generalize aft1PrologueG BD = before
  induction before generalizing s with
  | nil =>
    have singleton (stmt : Stmt) : Stmt.TraceSafeList bounds [stmt] s ↔ Stmt.TraceSafe bounds stmt s := by
      rw [Stmt.TraceSafeList]
      cases stepStmt stmt s <;> simp only [Stmt.TraceSafeList, and_true]
    simp only [List.nil_append, singleton]
    exact hloop s
  | cons stmt tail ih =>
    simp only [List.cons_append, Stmt.TraceSafeList]
    cases stepStmt stmt s with
    | none => rfl
    | some st => exact and_congr_right (fun _ => ih st)

theorem executes (Q K V H O : RegionName) (s_qh s_hh s_ht : Nat)
    (scale : ℝ) (BT BD NT : Nat) (STORE IFCOND : Bool) (s : BlockState)
    (hBT : 0 < BT) (hstride : BD ≤ s_ht)
    (hQO : Q ≠ O) (hKO : K ≠ O) (hVO : V ≠ O)
    (hQH : Q ≠ H) (hKH : K ≠ H) (hVH : V ≠ H) (hHO : H ≠ O) :
    ∃ st, exec (TypedSource.scalarLeft Q K V H O s_qh BD 1 s_hh s_ht (NT * BT)
      scale BT BD NT STORE IFCOND).eraseDType.toAlgKernel s = some st ∧
      RealLoop.Invariant Q K V H O s_qh s_hh s_ht scale BT BD STORE s NT st := by
  rw [same_execution]
  exact RealLoop.original_executes Q K V H O s_qh s_hh s_ht scale BT BD NT STORE IFCOND s
    hBT hstride hQO hKO hVO hQH hKH hVH hHO

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionFwdTriton1.RealCandidate
