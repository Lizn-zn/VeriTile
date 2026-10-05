import bench.examples.HyperConnectionsDepth.Kernels
import VeriTile.Triton.Math.SinkhornMemory

/-! Address safety for the general matrix source and its optimized version.
All accesses use row-major windows; the normalization loops preserve addresses. -/
namespace VeriTile.Bench.Examples.HyperConnectionsDepth.Memory
open VeriTile Triton TraceProofs Kernels
set_option maxHeartbeats 6400000
set_option linter.unusedSimpArgs false

theorem traceSafe (optimized : Bool) (S T D iters : Nat) (tau : ℝ)
    (bounds : RegionBounds) (s : BlockState)
    (hres : s.pid * (S * D) + S * D ≤ bounds "res_mix")
    (hbranch : s.pid * (T * D) + T * D ≤ bounds "branch_out")
    (hpost : T * S ≤ bounds "h_post")
    (hout : s.pid * (S * D) + S * D ≤ bounds "out") :
    Kernel.TraceSafe bounds
      (if optimized then (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau).toAlgKernel
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
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [T, D] => (s.pid * (T * D)) + i.1.val * D + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (T * D)) T D i.1 i.2.1) hbranch
    intro branch
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul] at hval
    subst val
    apply assign_cons
    · simp only [Op.SafeAt.eq_def, and_true, true_and]
      apply region_safe (offsets := ⟨fun i : TileIndex [T, S] => 0 + i.1.val * S + i.2.1.val⟩)
      · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
          Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
          NumericDType.add, NumericDType.mul]
      · intro i
        exact lt_of_lt_of_le (matrix_offset_lt 0 T S i.1 i.2.1) (by simpa using hpost)
    intro post
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro z
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro u
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro v
    refine Stmt.TraceSafeList.cons_intro (LogSinkhorn.loop_traceSafe T S iters bounds _) ?_
    intro q hq
    have hb := LogSinkhorn.loop_register T S iters _ q hq .nat [] "b" (by decide) (by decide) (by decide)
    have hs := LogSinkhorn.loop_register T S iters _ q hq .nat [S] "offs_s" (by decide) (by decide) (by decide)
    have hd := LogSinkhorn.loop_register T S iters _ q hq .nat [D] "offs_d" (by decide) (by decide) (by decide)
    simp only [BlockState.setReg] at hb hs hd
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro weights
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro mix
    apply assign_cons (by simp [Op.SafeAt.eq_def])
    intro out
    apply assign_eval_cons (by simp [Op.SafeAt.eq_def])
    intro val hval
    simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, hd] at hval
    subst val
    refine Stmt.TraceSafeList.cons_intro ?_ (fun _ _ => .nil_intro)
    simp only [Stmt.TraceSafe, MemAccess.SafeAt, MaskOpt.SafeAt, Op.SafeAt.eq_def, and_true, true_and]
    apply region_safe (offsets := ⟨fun i : TileIndex [S, D] => s.pid * (S * D) + i.1.val * D + i.2.1.val⟩)
    · simp [evalOp.eq_def, BlockState.setReg, Tile.bop, Tile.vec, Tile.scalar,
        Tile.expandDim, TileShape.dropInsertedIndex, TileShape.insertAxis,
        NumericDType.add, NumericDType.mul, hb, hs, hd]
    · intro i
      exact lt_of_lt_of_le (matrix_offset_lt (s.pid * (S * D)) S D i.1 i.2.1) hout

/-- Both sources remain in the fragment supported by address relocation. -/
theorem flattenOk (optimized : Bool) (S T D iters : Nat) (tau : ℝ) :
    (if optimized then (matrixOptimized "res_mix" "branch_out" "h_post" "out" S T D iters tau).toAlgKernel
     else (matrixOriginal S T D iters tau).toAlgKernel).FlattenOk := by
  cases optimized <;> simp [Kernel.FlattenOk, matrixOriginal, matrixKernel, matrixOptimized, ComputeKernel.toAlgKernel,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

end VeriTile.Bench.Examples.HyperConnectionsDepth.Memory
