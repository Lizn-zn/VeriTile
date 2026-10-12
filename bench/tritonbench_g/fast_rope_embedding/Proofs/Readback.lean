import bench.tritonbench_g.fast_rope_embedding.Proofs.Step
namespace VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Readback
open VeriTile.Triton TypedSource Code
noncomputable section
set_option maxHeartbeats 4000000
set_option maxRecDepth 25000
set_option linter.unusedSimpArgs false

theorem first (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u v : BlockState) (hc : Context u row H B cs ss)
    (he : stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v)
    (j : Fin B) (hj : j.val<H/2) :
    v.readMem Q (Reference.first row QR H k j.val)=
      u.readMem Q (Reference.first row QR H k j.val)*cs j-
        u.readMem Q (Reference.second row QR H k j.val)*ss j := by
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
  unfold Reference.second Reference.first
  conv_lhs => unfold BlockState.readMem
  rw [foldl_store_preserve_cell _ _ _ Q (row*QR+k*H+j.val) _ _ (by
    intro t _ ht heq
    have := Reference.halves_disjoint row QR H k j t.1 hj
    exact this (by simpa [Reference.first,Reference.second] using heq.2.symm))]
  change BlockState.readMem _ Q (row*QR+k*H+j.val)=_
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (shape := [B]) _ _ _ _ (j,PUnit.unit) hj (by
    intro t _ ht
    apply Prod.ext
    · apply Fin.ext
      simpa using ht
    · exact Subsingleton.elim _ _)]
  simp [hj,Reference.first,Reference.second]
  rfl

theorem second (Q C S : RegionName) (QR CR SR seq H heads B row k : Nat) (backwards : Bool)
    (cs ss : Fin B → ℝ) (u v : BlockState) (hc : Context u row H B cs ss)
    (he : stepStmts (body Q C S QR CR SR seq H heads B backwards) (u.setReg "k" .nat [] (Tile.scalar k))=some v)
    (j : Fin B) (hj : j.val<H/2) :
    v.readMem Q (Reference.second row QR H k j.val)=
      u.readMem Q (Reference.second row QR H k j.val)*cs j+
        u.readMem Q (Reference.first row QR H k j.val)*ss j := by
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
  unfold Reference.second Reference.first
  rw [BlockState.scatter_readback_prop_masked_nd_of_true (shape := [B]) _ _ _ _ (j,PUnit.unit) hj (by
    intro t _ ht
    apply Prod.ext
    · apply Fin.ext
      simpa using ht
    · exact Subsingleton.elim _ _)]
  simp [hj,Reference.first,Reference.second]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRopeEmbedding.Readback
