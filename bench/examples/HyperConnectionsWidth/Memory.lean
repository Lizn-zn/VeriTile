import bench.examples.HyperConnectionsWidth.Kernels
import VeriTile.Triton.Math.SinkhornMemory

/-! Row-major bounds for both matrix products and all normalization iterations. -/
namespace VeriTile.Bench.Examples.HyperConnectionsWidth.Memory
open VeriTile Triton TraceProofs Kernels
set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

theorem traceSafe (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid * (S * D) + S * D ≤ bounds "res")
    (hreslog : S * S ≤ bounds "h_res")
    (hprelog : S * T ≤ bounds "h_pre")
    (hmix : s.pid * (S * D) + S * D ≤ bounds "res_mix")
    (hbranch : s.pid * (T * D) + T * D ≤ bounds "branch_in") :
    Kernel.TraceSafe bounds
      (if optimized then (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).toAlgKernel
       else (matrixOriginal S T D iters tau).toAlgKernel) s := by
  cases optimized <;> simp only [Bool.false_eq_true, ↓reduceIte]
  all_goals
    change Stmt.TraceSafeList bounds _ s
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => (s.pid * (S * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hres
    intro res
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, S] => 0 + i.1.val * S + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 S S i.1 i.2.1) (by simpa using hreslog)
    intro reslog
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe S S iters bounds _) ?_
    intro q hq
    have hb := LogSinkhorn.loop_register S S iters _ q hq .nat [] "b" (by decide) (by decide) (by decide)
    have hs := LogSinkhorn.loop_register S S iters _ q hq .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have ht := LogSinkhorn.loop_register S S iters _ q hq .nat [T] "offs_t" (by decide) (by decide) (by decide)
    have hd := LogSinkhorn.loop_register S S iters _ q hq .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb hs ht hd
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro mix
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, T] => 0 + i.1.val * T + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul, hb, hs, ht, hd]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 S T i.1 i.2.1) (by simpa using hprelog)
    intro prelog
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v2
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe S T iters bounds _) ?_
    intro r hr
    have hb2 := LogSinkhorn.loop_register S T iters _ r hr .nat [] "b" (by decide) (by decide) (by decide)
    have hs2 := LogSinkhorn.loop_register S T iters _ r hr .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have ht2 := LogSinkhorn.loop_register S T iters _ r hr .nat [T] "offs_t" (by decide) (by decide) (by decide)
    have hd2 := LogSinkhorn.loop_register S T iters _ r hr .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb2 hs2 ht2 hd2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights2
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro branch
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2] at hval
    subst val
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2] at hval
    subst val
    refine Stmt.TraceSafeList.cons_intro ?_ ?_
    · simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => (s.pid * (S * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hmix
    intro t hstore
    have hregs := store_regs _ _ _ _ _ hstore
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := ⟨fun i : TileIndex [T, D] => (s.pid * (T * D)) + i.1.val * D + i.2.1.val⟩)
    · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, ht, hd, hb2, hs2, ht2, hd2, hregs]
    · intro i
      exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (T * D)) T D i.1 i.2.1) hbranch

/-- Both sources remain in the fragment supported by address relocation. -/
theorem flattenOk (optimized : Bool) (S T D iters : Nat) (tau : ℝ) :
    (if optimized then (matrixOptimized "res" "h_res" "h_pre" "res_mix" "branch_in" S T D iters tau).toAlgKernel
     else (matrixOriginal S T D iters tau).toAlgKernel).FlattenOk := by
  cases optimized <;> simp [Kernel.FlattenOk, matrixOriginal, matrixKernel, matrixOptimized, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

end VeriTile.Bench.Examples.HyperConnectionsWidth.Memory
