import bench.tritonbench_g.attn_fwd_triton.Kernels
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealContract

/-! Full erased-source equality transfers real formulas and memory guarantees.
It makes no FP equivalence claim between these differently typed sources. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProjection
open VeriTile.Triton
abbrev Parameters := AttnFwdCausal.RealContract.Parameters
set_option maxRecDepth 16000
set_option maxHeartbeats 2000000
def source (BM BN BD : Nat) (p : Parameters) (candidate : Bool) : ComputeKernel :=
  (if candidate then TypedSource.reciprocal else TypedSource.original)
    p.Q p.K p.V p.QS p.KS p.Out p.queryBatch p.queryHead p.queryRow p.queryDim
    p.keyBatch p.keyHead p.keyRow p.keyDim p.valueBatch p.valueHead p.valueRow p.valueDim
    p.outputBatch p.outputHead p.outputRow p.outputDim
    p.batches p.heads p.context p.headDim BM BN BD p.activeHead p.stage p.validStage
/-- Precision erasure removes the bf16 and V-to-fp16 casts while preserving
both actual helper loops and every other operation. -/
theorem full_erased_source (BM BN BD : Nat) (p : Parameters) (candidate : Bool) :
    (source BM BN BD p candidate).eraseDType =
      AttnFwdCausal.RealContract.kernel BM BN BD p candidate := by
  cases candidate <;>
    simp only [source, AttnFwdCausal.RealContract.kernel, AttnFwdCausal.RealContract.source,
      Bool.false_eq_true, Bool.true_eq, ↓reduceIte,
      TypedSource.original, TypedSource.reciprocal,
      AttnFwdCausal.TypedSource.original, AttnFwdCausal.TypedSource.reciprocal,
      ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?,
      ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
      ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
      Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
      MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType,
      NumericDType.eraseDType, ComparableDType.eraseDType, IntegralDType.eraseDType,
      Except.bind, bind] <;> rfl
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealProjection
