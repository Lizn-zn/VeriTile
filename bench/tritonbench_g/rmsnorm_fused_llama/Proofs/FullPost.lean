import bench.tritonbench_g.rmsnorm_fused_llama.Proofs.FullVariance

namespace VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
open VeriTile.Triton Finset
set_option maxHeartbeats 3000000

noncomputable def meanSquare (s : BlockState) (X : RegionName) (stride N : Nat) : ℝ :=
  (∑ k : Fin N, s.readMem X (s.pids 0 * stride + k.val) ^ 2) / (N : ℝ)

noncomputable def inverseStd (s : BlockState) (X : RegionName)
    (stride N : Nat) (eps : ℝ) : ℝ := (Real.sqrt (meanSquare s X stride N + eps))⁻¹

/-- Positive epsilon makes the actual full-row square-root operand positive. -/
theorem std_positive (s : BlockState) (X : RegionName) (stride N : Nat)
    (eps : ℝ) (heps : 0 < eps) : 0 < Real.sqrt (meanSquare s X stride N + eps) := by
  apply Real.sqrt_pos.2
  have hm : 0 ≤ meanSquare s X stride N := by
    unfold meanSquare
    exact div_nonneg (Finset.sum_nonneg (fun _ _ => sq_nonneg _)) (Nat.cast_nonneg N)
  linarith

noncomputable def postState (s0 s : BlockState) (X : RegionName)
    (stride N : Nat) (eps : ℝ) : BlockState :=
  (s.setReg "var" .real [] (Tile.scalar (some (meanSquare s0 X stride N)))).setReg
    "rstd" .real [] (Tile.scalar (some (inverseStd s0 X stride N eps)))

theorem sum_eval (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (h : VarInv s0 X Y stride N B c s) :
    evalOp (.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (.ref .real [B] "_var")) s =
      some (Tile.scalar (some (∑ j : Fin B, accumulator s0 X stride N B c j))) := by
  rw [evalOp_reduceSum, evalOp_ref, h.acc]
  apply congrArg some
  ext idx
  rw [Tile.reduceSum_false, Tile.reduceSumDrop_data]
  change @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
    (fun j => (some (accumulator s0 X stride N B c j) : WithBot ℝ)) =
    some (∑ j : Fin B, accumulator s0 X stride N B c j)
  exact withBot_sum_some _

theorem mean_eval (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (hB : 0 < B) (hN : N ≤ c * B)
    (h : VarInv s0 X Y stride N B c s) :
    evalOp (.div .real .nil
      (.reduceSum (⟨0, by simp⟩ : Fin [B].length) Bool.false (.ref .real [B] "_var"))
      (.const (N : ℝ))) s = some (Tile.scalar (some (meanSquare s0 X stride N))) := by
  have hs := sum_eval s0 s X Y stride N B c h
  rw [accumulator_sum s0 X stride N B c hB hN] at hs
  simp only [evalOp_div, evalOp_const, Option.bind_eq_bind]
  erw [hs]
  simp only [Option.bind_some]
  apply congrArg some
  ext idx
  rfl

theorem post_exec (s0 s : BlockState) (X Y : RegionName)
    (stride N B c : Nat) (eps : ℝ) (hB : 0 < B) (hN : N ≤ c * B)
    (h : VarInv s0 X Y stride N B c s) :
    stepStmts (post N B eps) s = some (postState s0 s X stride N eps) := by
  unfold post
  rw [stepStmts.cons_some (stepStmt_assign_eq_some
    (mean_eval s0 s X Y stride N B c hB hN h))]
  simp [stepStmts, stepStmt, evalOp.eq_def, postState, inverseStd,
    Tile.bop, Tile.uop, NumericDType.add, NumericDType.div, WithBot.realSqrt,
    Tile.scalar, -Tile.scalar_eta, Function.comp_def, one_div]

structure WriteContext (s0 : BlockState) (X Y W : RegionName)
    (stride N : Nat) (eps : ℝ) (s : BlockState) : Prop where
  pids : s.pids = s0.pids
  xptr : s.regs .ptr [] "X" = some (Tile.scalar (X, s0.pids 0 * stride))
  yptr : s.regs .ptr [] "Y" = some (Tile.scalar (Y, s0.pids 0 * stride))
  rstd : s.regs .real [] "rstd" = some (Tile.scalar (some (inverseStd s0 X stride N eps)))
  readX : s.readMem X = s0.readMem X
  readW : s.readMem W = s0.readMem W

theorem post_context (s0 s : BlockState) (X Y W : RegionName)
    (stride N B c : Nat) (eps : ℝ) (h : VarInv s0 X Y stride N B c s) :
    WriteContext s0 X Y W stride N eps (postState s0 s X stride N eps) := by
  constructor
  · simpa [postState] using h.pids
  · simpa [postState] using h.xptr
  · simpa [postState] using h.yptr
  · simp [postState]
  · funext o; simp [postState, BlockState.readMem, h.mem]
  · funext o; simp [postState, BlockState.readMem, h.mem]

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.RmsnormFusedLlama.General
