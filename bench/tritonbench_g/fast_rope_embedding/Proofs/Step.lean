import bench.tritonbench_g.fast_rope_embedding.Proofs.Code
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Step
open VeriTile.Triton TypedSource Code
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem executes (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u : BlockState) (hc : Context u row H B cs ss) :
    ∃ v,stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v := by
  rcases hc with ⟨hrow,hcols,hhalf,hmask,hcos,hsin⟩
  simp [body,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,hrow,hcols,hhalf,hmask,hcos,hsin]

theorem frame (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u v : BlockState) (hc : Context u row H B cs ss)
    (he : stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v)
    (q : RegionName) (o : Nat)
    (hfirst : ∀ j : Fin B,j.val<H/2 → ¬(Q=q ∧ Reference.first row QR H k j.val=o))
    (hsecond : ∀ j : Fin B,j.val<H/2 → ¬(Q=q ∧ Reference.second row QR H k j.val=o)) :
    v.mem q o=u.mem q o := by
  rcases hc with ⟨hrow,hcols,hhalf,hmask,hcos,hsin⟩
  simp [body,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,hrow,hcols,hhalf,hmask,hcos,hsin] at he
  subst v
  refine Eq.trans (foldl_store_preserve_cell _ _ _ q o _ _ ?_) (foldl_store_preserve_cell _ _ _ q o _ _ ?_)
  · intro j _ hj heq
    exact hsecond j.1 (by simpa using hj) (by simpa [Reference.second,Reference.first] using heq)
  · intro j _ hj heq
    exact hfirst j.1 (by simpa using hj) (by simpa [Reference.first] using heq)

theorem preserves_context (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u v : BlockState) (hc : Context u row H B cs ss)
    (he : stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v) :
    Context v row H B cs ss ∧ v.pids=u.pids := by
  rcases hc with ⟨hrow,hcols,hhalf,hmask,hcos,hsin⟩
  simp [body,Code.original,rope,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
    Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
    Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
    IntegralDType.eraseDType,ComputeDType.eraseDType,Bool.false_eq_true,Except.bind,bind,
    stepStmts,stepStmt,evalOp.eq_def,Option.bind,Tile.bop,Tile.cop,Tile.ptrAdd,Tile.uop,
    NumericDType.add,NumericDType.mul,NumericDType.sub,ComparableDType.lt,
    FloatDType.cast,FloatDType.ofWithBot,FloatDType.toWithBot,hrow,hcols,hhalf,hmask,hcos,hsin] at he
  subst v
  constructor
  · constructor
    all_goals simp [BlockState.setReg,hrow,hcols,hhalf,hmask,hcos,hsin]
  · simp [BlockState.foldl_writeMem_prop_masked_pids]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Step
