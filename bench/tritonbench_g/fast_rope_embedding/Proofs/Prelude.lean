import bench.tritonbench_g.fast_rope_embedding.Proofs.Code
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Prelude
open VeriTile.Triton TypedSource Code
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem executes (Q C S : RegionName) (QR CR SR seq H heads B : Nat) (backwards : Bool) (u : BlockState) :
    ∃ v,stepStmts (prelude Q C S QR CR SR seq H heads B backwards) u=some v ∧
      Context v u.pid H B (Reference.coefficient u C CR seq H B)
        (fun j => Reference.signedSine backwards (Reference.coefficient u S SR seq H B j)) ∧
      v.regs .nat [] "head_start"=some (Tile.scalar (Reference.start (u.pids 1))) ∧
      v.regs .nat [] "head_end"=some (Tile.scalar (Reference.stop (u.pids 1) heads)) ∧
      v.pids=u.pids ∧ v.mem=u.mem := by
  cases backwards
  all_goals
    simp [prelude,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
      stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
      NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
      FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,IntegralDType.mod,Reference.coefficient,Reference.coefficientAddress,Reference.signedSine]

  all_goals
    refine ⟨?_,?_,?_⟩
    · constructor <;> simp [BlockState.setReg,Reference.coefficient,Reference.coefficientAddress,Reference.signedSine]
      all_goals first | rfl | (funext j; split_ifs <;> norm_num at *)
    · simp [Reference.start,BlockState.setReg]
    · simp [Reference.stop,Reference.start,BlockState.setReg,Tile.ext_iff,min_def_lt]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Prelude
