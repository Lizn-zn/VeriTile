import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealExecution
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardRun
open VeriTile.Triton TypedSource
noncomputable section
set_option maxHeartbeats 3000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

def original (isGemma : Bool) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) : Kernel :=
  (backward .fp32 .fp32 Bool.false Y X W r dW YR XR WR RR DWR N B eps isGemma).eraseDType.toAlgKernel

theorem flatten_ok (isGemma : Bool) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) :
    (original isGemma Y X W r dW YR XR WR RR DWR N B eps).FlattenOk := by
  cases isGemma
  all_goals simp [Kernel.FlattenOk,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def,original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind]

theorem executes (isGemma : Bool) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) (u : BlockState) :
    ∃ v, exec (original isGemma Y X W r dW YR XR WR RR DWR N B eps) u=some v := by
  cases isGemma
  all_goals simp [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,exec,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot]

theorem frame (isGemma : Bool) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) (u v : BlockState)
    (he : exec (original isGemma Y X W r dW YR XR WR RR DWR N B eps) u=some v)
    (q : RegionName) (o : Nat)
    (hmiss : ∀ j : Fin B,j.val<N → ¬(Y=q ∧ u.pids 0*YR+j.val=o)) :
    v.mem q o=u.mem q o := by
  cases isGemma
  all_goals
    simp [original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,exec,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot] at he
    subst v
    refine foldl_store_preserve_cell _ _ _ q o _ _ ?_
    intro j _ hj hq
    exact hmiss j.1 (by simpa using hj) hq

theorem trace_safe (isGemma : Bool) (Y X W r dW : RegionName) (YR XR WR RR DWR N B : Nat) (eps : ℝ) (bounds : RegionBounds) (u : BlockState)
    (hy : ∀ j : Fin B,j.val<N → u.pids 0*YR+j.val<bounds Y)
    (hx : ∀ j : Fin B,j.val<N → u.pids 0*XR+j.val<bounds X)
    (hw : ∀ j : Fin B,j.val<N → j.val<bounds W)
    (hr : u.pids 0*RR<bounds r) :
    Kernel.TraceSafe bounds (original isGemma Y X W r dW YR XR WR RR DWR N B eps) u := by
  cases isGemma
  all_goals
    simp [Kernel.TraceSafe,original,backward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,
      MaskOpt.SafeAt,MemAccess.SafeAt,MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,
      MaskOpt.Active,Op.PointerAddressesSafeOn,Op.MemorySafe,BlockState.setReg,exec,stepStmts,stepStmt,evalOp,evalOp.eq_def,Option.bind,
    Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,
    NumericDType.add,NumericDType.mul,NumericDType.sub,NumericDType.div,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot]
    exact ⟨fun j hj => hy j hj,fun j hj => hx j hj,fun j hj => hw j hj,hr,fun j hj => hy j hj⟩

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardRun
