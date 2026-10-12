import bench.tritonbench_g.fast_layernorm.Proofs.RealExecution
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.BackwardRun
open VeriTile.Triton TypedSource RealExecution
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false
theorem layernorm_backward_flattenOk
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols : Nat) (eps : ℝ) (BLOCK_SIZE : Nat) :
    ((layernorm_backward Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE).toAlgKernel).FlattenOk := by
  unfold Kernel.FlattenOk
  simp [layernorm_backward,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    StmtList.FlattenOk, Stmt.FlattenOk, Op.FlattenOk.eq_def]

theorem layernorm_backward_exec_isSome
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s : BlockState) :
    ∃ s1, exec ((layernorm_backward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE).toAlgKernel) s = some s1 := by
  simp [exec, layernorm_backward,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot]

theorem layernorm_backward_frame
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (s s1 : BlockState)
    (hExec : exec ((layernorm_backward Y X W bias r mu Y_row_stride X_row_stride
        n_cols eps BLOCK_SIZE).toAlgKernel) s = some s1)
    (r' : RegionName) (o' : Nat)
    (hmissY : ∀ i : Fin BLOCK_SIZE, i.val < n_cols →
      ¬(Y = r' ∧ s.pids 0 * Y_row_stride + i.val = o'))
 :
    s1.mem r' o' = s.mem r' o' := by
  simp [exec, layernorm_backward,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind, ComputeKernel.toAlgKernel, stepStmts, stepStmt,
        evalOp, evalOp.eq_def, ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
        Tile.bop, Tile.cop, Tile.ptrAdd, Tile.uop, Tile.reduceSum,
        Tile.reduceSumDrop, TileShape.axisDim, TileShape.eraseAxis,
        TileShape.insertAxisIndex, NumericDType.add, NumericDType.mul,
        NumericDType.sub, NumericDType.div, ComparableDType.lt,
        FloatDType.cast, FloatDType.ofWithBot, FloatDType.toWithBot] at hExec
  subst hExec
  refine foldl_store_preserve_cell _ _ _ r' o' _ _ ?_
  intro k _ hmk hc
  exact hmissY k.1 (by simpa using hmk) hc

theorem layernorm_backward_traceSafe
    (Y X W bias r mu : RegionName)
    (Y_row_stride X_row_stride n_cols BLOCK_SIZE : Nat)
    (eps : ℝ) (bounds : RegionBounds) (s : BlockState)
    (hin1 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pids 0 * X_row_stride + j.val < bounds X)
    (hin2 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → j.val < bounds W)
    (hin3 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols → j.val < bounds bias)
    (hin0 : ∀ j : Fin BLOCK_SIZE, j.val < n_cols →
      s.pids 0 * Y_row_stride + j.val < bounds Y)
    (hout2 : s.pids 0 < bounds r)
    (hout3 : s.pids 0 < bounds mu) :
    Kernel.TraceSafe bounds
      ((layernorm_backward Y X W bias r mu Y_row_stride X_row_stride n_cols eps
        BLOCK_SIZE).toAlgKernel) s := by
  unfold Kernel.TraceSafe
  simp [layernorm_backward,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind, ComputeKernel.toAlgKernel,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?,
    Stmt.TraceSafeList, Stmt.TraceSafe, Op.SafeAt.eq_def,
    MaskOpt.SafeAt, MemAccess.SafeAt, stepStmt, evalOp.eq_def,
    MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
    MaskOpt.Active, Op.PointerAddressesSafeOn, Op.MemorySafe,
    BlockState.setReg,
    Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd,
    NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
    ComparableDType.lt, FloatDType.cast, FloatDType.ofWithBot,
    FloatDType.toWithBot,
    Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex]
  exact ⟨fun a ha => hin0 a ha,fun a ha => hin1 a ha,fun a ha => hin2 a ha,
    fun a ha => hin3 a ha,hout2,hout3,fun a ha => hin0 a ha⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.BackwardRun
