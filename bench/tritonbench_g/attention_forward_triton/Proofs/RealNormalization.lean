import bench.tritonbench_g.attention_forward_triton.Kernels
import VeriTile.Triton.Source
import VeriTile.Triton.Math.BroadcastRewriteR
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit

/-! Real normalization of the complete attention sources. The two programs
share the full loop and final masked store. The execution and safety equalities
below are independent of the admitted FP rules; they do not by themselves
supply the independent attention formula. -/
namespace VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealNormalization
open VeriTile.Triton
set_option maxRecDepth 16000
set_option maxHeartbeats 2400000

private def outputStore (BM BD N HA : Nat) : Stmt :=
  .store .real [BM, BD] (.ptr (.ref .ptr [BM, BD] "O_block_ptr"))
    (.ref .real [BM, BD] "acc")
    (.mask (.boolAnd (.consR (.consL .nil))
      (.lt .nat .scalarR (.expandDim ⟨1, by simp⟩ (.ref .nat [BM] "offs_m")) (.constNat N))
      (.expandDim ⟨0, by simp⟩ (.lt .nat .scalarR (.arange BD) (.constNat HA)))))

private theorem erased_projection_supported (k : ComputeKernel)
    (h : k.toAlgorithm? = .ok k.toAlgKernel) :
    k.eraseDType.toAlgorithm? = .ok k.eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, h]
  simp

section
variable
    (q k v qs ks out : RegionName)
    (sqz sqh sqm sqk skz skh skn skk svz svh svk svn soz soh som son
      Z H N HD BM BN BD HA stage : Nat)

/-- Projection succeeds; the real semantics does not use the failure fallback. -/
theorem original_projection_supported :
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgorithm? =
      .ok (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by rfl)

/-- Projection succeeds; the real semantics does not use the failure fallback. -/
theorem reciprocal_projection_supported :
    (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgorithm? =
      .ok (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel := by
  exact erased_projection_supported _ (by rfl)

/-- Locate the final acc assignment by its name; intervening source statements
remain in the common prefix instead of relying on a numeric list offset. -/
private def commonPrefix : List Stmt :=
  (ComputeKernel.mk [] [] (Source.prefixBefore
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).surfaceBody "acc" 1)).eraseDType.toAlgKernel.body

private theorem original_body :
    (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel.body =
      commonPrefix q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage ++
      [.assign .real [BM, BD] "acc"
        (.div .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
          (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i"))),
       outputStore BM BD N HA] := by
  simp only [commonPrefix, TypedSource.original,
    Source.prefixBefore, ComputeKernel.surfaceBody, ↓reduceIte, String.reduceEq]
  simp only [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind]
  rfl

private theorem reciprocal_body :
    (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel.body =
      commonPrefix q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage ++
      [.assign .real [BM, BD] "acc"
        (.mul .real (.consSame (.consR .nil)) (.ref .real [BM, BD] "acc")
          (.div .real .scalarL (.const 1.0)
            (.expandDim ⟨1, by simp⟩ (.ref .real [BM] "l_i")))),
       outputStore BM BD N HA] := by
  simp only [commonPrefix, TypedSource.original, TypedSource.reciprocal,
    Source.prefixBefore, ComputeKernel.surfaceBody, ↓reduceIte, String.reduceEq]
  simp only [ComputeKernel.eraseDType, ComputeKernel.toAlgorithm?,
    ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    IntegralDType.eraseDType, Except.bind, bind]
  rfl

/-- Complete executions agree, including all register/memory effects and failure. -/
theorem executions_agree (s : BlockState) :
    exec (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s =
      exec (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s := by
  unfold exec
  rw [original_body, reciprocal_body]
  exact RealBroadcastRewrite.contextual_div_mul_rcp _ _ _ _ _ _ 1.0 (by norm_num) s

/-- The complete programs have the same active memory safety obligations. -/
theorem trace_safety_agrees (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafe bounds
        (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafe bounds
        (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s := by
  unfold Kernel.TraceSafe
  rw [original_body, reciprocal_body]
  exact RealBroadcastRewrite.contextual_traceSafe _ _ _ _ _ _ 1.0 (by norm_num) bounds s

/-- The complete programs have the same active memory safety obligations. -/
theorem trace_safetyR_triv_agrees (bounds : RegionBounds) (s : BlockState) :
    Kernel.TraceSafeR .triv bounds
        (TypedSource.original q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s ↔
      Kernel.TraceSafeR .triv bounds
        (TypedSource.reciprocal q k v qs ks out sqz sqh sqm sqk skz skh skn skk
      svz svh svk svn soz soh som son Z H N HD BM BN BD HA stage).eraseDType.toAlgKernel s := by
  unfold Kernel.TraceSafeR
  rw [original_body, reciprocal_body]
  exact RealBroadcastRewrite.contextual_traceSafeR_triv _ _ _ _ _ _ 1.0 (by norm_num) bounds s

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttentionForwardTriton.RealNormalization
