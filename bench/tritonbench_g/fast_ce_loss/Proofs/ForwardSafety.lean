import bench.tritonbench_g.fast_ce_loss.Proofs.ForwardReal
namespace VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardReal
open VeriTile.Triton TypedSource RealMath
noncomputable section
set_option maxHeartbeats 6000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem trace_safe (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) (bounds : RegionBounds) (u : BlockState) (hb : 0<B)
    (hbL : u.pids 0 < bounds (Region.cast L))
    (hbY : u.pids 0 < bounds Y) (hbZ : u.pids 0 < bounds Z)
    (hbG : u.readMemValue .int (Region.cast L) (u.pids 0) ≠ -100 →
      u.pids 0*R+(u.readMemValue .int (Region.cast L) (u.pids 0)).toNat < bounds X)
    (hbX : ∀ j : Fin B, j.val<V → u.pids 0*R+j.val < bounds X) :
    (original cap scale X Y Z L V R B SC SL).TraceSafe bounds u := by
  unfold Kernel.TraceSafe
  cases cap <;> cases scale
  all_goals by_cases hi : u.readMemValue .int (Region.cast L) (u.pids 0) = -100
  all_goals
    simp only [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind]
    simp [Stmt.TraceSafeList,Stmt.TraceSafe,Op.SafeAt.eq_def,MaskOpt.SafeAt,
      MemAccess.ActiveAddressSafe,memAccessActiveAddressSafe,MemAccess.SafeAt,MaskOpt.Active,
      Op.PointerAddressesSafeOn,Op.MemorySafe,hbL,hbY,hbZ,hbG,exec,stepStmts,stepStmt,evalOp.eq_def,Option.bind,Option.map,
      Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,Tile.scalar,-Tile.scalar_eta,
      NumericDType.add,NumericDType.sub,NumericDType.mul,NumericDType.div,
      ComparableDType.lt,ComparableDType.ne,ComparableDType.eq,FloatDType.cast,hi,hb,
      Tile.reduceMax,Tile.reduceMaxDrop,Tile.reduceSum,Tile.reduceSumDrop,
      TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,TileShape.allIndices]
    exact fun j hj => hbX j hj

theorem flatten_ok (cap scale : Bool) (X Y Z : RegionName) (L : Region .int)
    (V R B : Nat) (SC SL : ℝ) :
    (original cap scale X Y Z L V R B SC SL).FlattenOk := by
  unfold Kernel.FlattenOk
  cases cap <;> cases scale
  all_goals
    simp [original,forward,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,↓reduceIte,Except.bind,bind,StmtList.FlattenOk,Stmt.FlattenOk,Op.FlattenOk.eq_def]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastCeLoss.ForwardReal
