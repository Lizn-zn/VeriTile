import bench.tritonbench_g.attention_kernel.Kernels
import VeriTile.Triton.Source
import VeriTile.Triton.Math.BroadcastRewriteR
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! The two complete real programs differ only at final normalization. Both
ordinary and flat-memory executions agree independently of any FP assumption. -/
namespace VeriTile.Bench.TritonBenchG.AttentionKernel.RealNormalization
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 600000
set_option linter.unusedSimpArgs false

private def outputTail (out : RegionName) (BM BD N som son : Nat) : List Stmt :=
  [.assign .blockPtr [BM, BD] "O_block_ptr"
    (.makeBlockPtrDynOffsets out (.ref .nat [] "q_offset") [N, BD] [BM, BD]
      [som, son] [.mul .nat .nil (.ref .nat [] "start_m") (.constNat BM), .constNat 0]),
   .store .real [BM, BD] (.blockPtr (.ref .blockPtr [BM, BD] "O_block_ptr") [])
      (.ref .real [BM, BD] "acc") .none]

section
variable (storage : TypedSource.Storage)
    (q k v bias out : RegionName) (sm_scale : ℝ)
    (sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm
      Z H N P BIAS B0_SIZE BD BM BN : Nat)

/-- Locate the final acc assignment by name while retaining the whole loop. -/
private def commonPrefix : List Stmt :=
  (ComputeKernel.mk [] [] (Source.prefixBefore
    (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).surfaceBody "acc" 1)).eraseDType.toAlgKernel.body

private theorem original_body :
    (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel.body =
      commonPrefix storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN ++
      .assign .real [BM, BD] "acc" (.div .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
          (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i"))) :: outputTail out BM BD N som son := by
  cases storage <;>
    simp only [commonPrefix, TypedSource.original, TypedSource.reciprocal,
      TypedSource.FP16.original, TypedSource.FP16.reciprocal,
      TypedSource.BF16.original, TypedSource.BF16.reciprocal,
      Source.prefixBefore, ComputeKernel.surfaceBody, ↓reduceIte, String.reduceEq,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
      MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    rfl

private theorem reciprocal_body :
    (TypedSource.reciprocal storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel.body =
      commonPrefix storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN ++
      .assign .real [BM, BD] "acc" (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
          (.div .real .scalarL (.const 1.0)
            (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))) :: outputTail out BM BD N som son := by
  cases storage <;>
    simp only [commonPrefix, TypedSource.original, TypedSource.reciprocal,
      TypedSource.FP16.original, TypedSource.FP16.reciprocal,
      TypedSource.BF16.original, TypedSource.BF16.reciprocal,
      Source.prefixBefore, ComputeKernel.surfaceBody, ↓reduceIte, String.reduceEq,
      ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType, MemAccess.eraseDType,
      MaskOpt.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, Except.bind, bind,
      List.map_cons, List.map_nil]
  all_goals
    simp only [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType, FloatDType.toTileDType, NumericDType.eraseDType,
      ComparableDType.eraseDType, IntegralDType.eraseDType, List.map_cons, List.map_nil]
    rfl

/-- Complete executions preserve all state effects and failures. -/
theorem executions_agree (s : BlockState) :
    exec (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel s = exec (TypedSource.reciprocal storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel s := by
  unfold exec
  rw [original_body, reciprocal_body]
  exact RealBroadcastRewrite.contextual_div_mul_rcp _ _ _ _ _ _ 1.0 (by norm_num) s

private theorem flatten_append (A : FlatAlloc) (xs ys : List Stmt) :
    A.flattenStmts (xs ++ ys) = A.flattenStmts xs ++ A.flattenStmts ys := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp only [List.cons_append, FlatAlloc.flattenStmts, ih]

/-- The same numerical rewrite commutes with translation to flat memory. -/
theorem flat_executions_agree (A : FlatAlloc) (s : BlockState) :
    exec (A.flattenKernel (TypedSource.original storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel) s = exec (A.flattenKernel (TypedSource.reciprocal storage q k v bias out sm_scale sqh sqm sqk skh skn skk svh svk svn soh som son sbh sbm Z H N P BIAS B0_SIZE BD BM BN).eraseDType.toAlgKernel) s := by
  simp only [exec, FlatAlloc.flattenKernel, original_body, reciprocal_body,
    flatten_append, FlatAlloc.flattenStmts, FlatAlloc.flattenStmt,
    FlatAlloc.flattenOp.eq_def]
  exact RealBroadcastRewrite.contextual_div_mul_rcp _ _ _ _ _ _ 1.0 (by norm_num) s

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionKernel.RealNormalization
