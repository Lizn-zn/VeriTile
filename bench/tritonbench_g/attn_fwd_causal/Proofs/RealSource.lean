import bench.tritonbench_g.attn_fwd_causal.Proofs.RealPrologue

/-! Complete erased source decomposition for both typed kernels. This checks
every leading assignment, both helper setups/ranges, normalization and masked
store in their actual order. The decomposition uses named fragments. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 3000000
set_option linter.unusedSimpArgs false

def normalization (BM BD : Nat) (candidate : Bool) : Stmt :=
  .assign .real [BM,BD] "acc"
    (if candidate then
      .mul .real (.consSame (.consR .nil)) (.ref .real [BM,BD] "acc")
        (.div .real .scalarL (.const 1.0) (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))
    else
      .div .real (.consSame (.consR .nil)) (.ref .real [BM,BD] "acc")
        (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))

def store (BM BD N HA : Nat) : Stmt :=
  .store .real [BM,BD] (.ptr (.ref .ptr [BM,BD] "O_block_ptr")) (.ref .real [BM,BD] "acc")
    (.mask (RealInit.queryMask BM BD N HA))

def epilogue (BM BD N HA : Nat) (candidate : Bool) : List Stmt :=
  [normalization BM BD candidate, store BM BD N HA]

section
variable (Q K V QS KS Out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat) (hs : stage = 2 ∨ stage = 3)

theorem body (candidate : Bool) :
    ((if candidate then TypedSource.reciprocal else TypedSource.original)
      Q K V QS KS Out sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage hs).eraseDType.toAlgKernel.body =
      RealPrologue.code Q K V QS KS Out sqz sqh sqm sqk skn H N HD BM BN BD HA ++
        RealMiddle.code BM BN BD N HA HD (decide (stage=2)) ++ epilogue BM BD N HA candidate := by
  cases candidate
  all_goals
    simp only [TypedSource.original, TypedSource.reciprocal, Bool.false_eq_true, Bool.true_eq,
      ↓reduceIte, ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
      ComputeKernel.surfaceBody, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind]
    simp [RealPrologue.code, RealPrologue.configuration, RealCoordinates.code, RealCoordinates.ceilExpr,
      RealPointers.code, RealPointers.rowExpr, RealPointers.rowPointerExpr, RealPointers.keyExpr,
      RealPointers.queryScaleExpr, RealPointers.keyScaleExpr, RealInit.code, RealInit.queryMask,
      RealMiddle.code, RealFirstSetup.code, RealFirstSetup.copies,
      RealTwoLoops.code, RealSetupSource.statement, RealLoop.sourceBody,
      RealSetup.code, RealSetup.lowerExpr, RealSetup.upperExpr, RealSetup.lowerName, RealSetup.upperName,
      RealSetup.keyBase, RealSetup.scaleBase, RealSetup.valueBase,
      RealMemory.keyName, RealMemory.scaleName, RealMemory.valueName,
      RealBody.first, RealBody.second, RealBody.loadPrefix, RealBody.keyMask, RealBody.valueMask,
      RealBody.score, RealBody.scoreExpr, RealBody.diagonal, RealBody.center, RealBody.centerExpr,
      RealBody.suffix, RealBody.probability, RealBody.rescale, RealBody.loadValue, RealBody.accumulate,
      RealBody.advance, RealExpressions.maskExpr, RealExpressions.penaltyExpr, epilogue, normalization, store]
end

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdCausal.RealSource
