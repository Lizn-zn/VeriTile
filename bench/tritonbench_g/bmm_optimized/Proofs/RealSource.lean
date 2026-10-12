import bench.tritonbench_g.bmm_optimized.Kernels
import VeriTile.Triton.Float.Erasure

/-! Precision erasure preserves every mask specialization and CTA branch.
This connects the real interpreter to the shared typed source; it is not yet
the independent product formula or the memory-safety proof. -/
namespace VeriTile.Bench.TritonBenchG.BmmOptimized.RealSource
open VeriTile.Triton
set_option maxHeartbeats 12000000
set_option maxRecDepth 18000
set_option linter.unusedSimpArgs false

section
variable (dtype : TypedSource.Storage) (divM divN divK swapped : Bool)
    (A B O : RegionName) (M N K BM BN BK GROUP_M : Nat)

def source := TypedSource.specialize dtype divM divN divK swapped A B O M N K BM BN BK GROUP_M

theorem source_projection :
    (source dtype divM divN divK swapped A B O M N K BM BN BK GROUP_M).toAlgorithm? =
      .ok (source dtype divM divN divK swapped A B O M N K BM BN BK GROUP_M).toAlgKernel := by
  cases dtype <;> cases divM <;> cases divN <;> cases divK <;> cases swapped <;> rfl

theorem storage_erasure :
    (source dtype divM divN divK swapped A B O M N K BM BN BK GROUP_M).eraseDType =
      (source .fp32 divM divN divK swapped A B O M N K BM BN BK GROUP_M).eraseDType := by
  cases dtype <;> cases divM <;> cases divN <;> cases divK <;> cases swapped
  all_goals
    simp only [source, TypedSource.specialize, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
      ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
      Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
      NumericDType.eraseDType, ComparableDType.eraseDType,
      IntegralDType.eraseDType, Except.bind, bind, eraseDType_float]

  all_goals rfl

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.BmmOptimized.RealSource
