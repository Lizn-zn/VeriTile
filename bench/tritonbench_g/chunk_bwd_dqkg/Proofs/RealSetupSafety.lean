import bench.tritonbench_g.chunk_bwd_dqkg.Proofs.RealLoopSafety

/-! Bounds for the prologue gate reads. The last gate load is unmasked, so
its address has an explicit bound even when the checked tile has no active
lanes. All coordinates come from the actual program launch. -/
namespace VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSetupSafety
open VeriTile.Triton RealMath RealStep
noncomputable section
set_option maxHeartbeats 500000
set_option maxRecDepth 14000
set_option linter.unusedSimpArgs false

structure Bounds (s : BlockState) (T BT : Nat) (g : RegionName) (bounds : RegionBounds) : Prop where
  gate : ∀ i : Fin BT, s.pids 1 * BT + i.val < T →
    s.pids 2 * T + s.pids 1 * BT + i.val < bounds g
  last : s.pids 2 * T + (min (s.pids 1 * BT + BT) T - 1) < bounds g

private theorem pointer_eval (s : BlockState) (g : RegionName) (T BT head time : Nat)
    (hh : s.regs .nat [] "i_bh" = some (Tile.scalar head))
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar time)) :
    evalOp (Op.makeBlockPtrDynOffsets g
      (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_bh") (Op.constNat T))
      [T] [BT] [1]
      [Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)]) s =
      some (⟨fun _ : TileIndex [BT] =>
        { region := g, baseOffset := head * T, parentShape := [T],
          blockShape := [BT], strides := [1], offsets := [time * BT] }⟩ : Tile .blockPtr [BT]) := by
  simp only [evalOp, hh, ht, List.mapM_cons, List.mapM_nil,
    Option.bind_some, Option.pure_def, Option.bind_eq_bind]
  rfl

private theorem last_eval (s : BlockState) (T BT time : Nat)
    (ht : s.regs .nat [] "i_t" = some (Tile.scalar time)) :
    evalOp (Op.sub .nat Broadcast.nil
      (Op.where
        (Op.lt ComparableDType.nat Broadcast.nil
          (Op.add .nat Broadcast.nil
            (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)) (Op.constNat BT))
          (Op.constNat T))
        (Op.add .nat Broadcast.nil
          (Op.mul .nat Broadcast.nil (Op.ref .nat [] "i_t") (Op.constNat BT)) (Op.constNat BT))
        (Op.constNat T)) (Op.constNat 1)) s =
      some (Tile.scalar (min (time * BT + BT) T - 1)) := by
  simp only [evalOp, ht, Option.bind_some, Tile.bop, Tile.cop, Tile.select,
    Tile.scalar, Broadcast.leftIndex, Broadcast.rightIndex, NumericDType.add,
    NumericDType.mul, NumericDType.sub, ComparableDType.lt]
  by_cases hl : time * BT + BT < T
  · simp [hl, Nat.le_of_lt hl]
  · simp [hl, Nat.not_lt.mp hl]

theorem safe (s : BlockState) (g : RegionName) (T BT BK : Nat)
    (bounds : RegionBounds) (hb : Bounds s T BT g bounds) :
    Stmt.TraceSafeList bounds (cbdPreLoop g T BT BK) s := by
  unfold cbdPreLoop
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 0 s))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 1 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_programId 2 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_numPrograms 2 _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (evalOp_arange BT _))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (pointer_eval _ g T BT (s.pids 2) (s.pids 1)
    (by simp) (by simp)))]
  refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
  apply TraceProofs.assign_cons
  · apply RealLoopSafety.block_load_safe _ _ _ _ _ (by simp; rfl)
    rintro ⟨i,⟨⟩⟩ hi
    simp only [TileShape.indexToList, BlockPtr.inBounds_1d, decide_eq_true_eq] at hi
    simpa only [TileShape.indexToList, BlockPtr.address_1d, Nat.mul_one, Nat.add_assoc]
      using hb.gate i hi
  · intro vs
    rw [TraceProofs.cons_of_run_iff (stepStmt_assign_eq_some (last_eval _ T BT (s.pids 1) (by simp)))]
    refine ⟨by simp [Stmt.TraceSafe, Op.SafeAt], ?_⟩
    apply TraceProofs.assign_cons
    · rw [Op.SafeAt.eq_def]
      refine ⟨by simp [Op.SafeAt], trivial, ?_⟩
      apply TraceProofs.region_safe
        (offsets := Tile.scalar (s.pids 2 * T + (min (s.pids 1 * BT + BT) T - 1)))
      · exact add_nat _ _ _ _ _ (mul_ref _ _ _ _ (by simp)) (by simp [evalOp_ref])
      · intro _
        exact hb.last
    · intro last
      apply TraceProofs.all_safe
      simp [Stmt.TraceSafe, Op.SafeAt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkBwdDqkg.RealSetupSafety
