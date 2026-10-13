import bench.tritonbench_g.softmax_flaggems.Kernels
import VeriTile.Triton.Float.Erasure

/-! Real projections of the complete source bodies. Storage casts disappear only
in this real abstraction; these equalities do not assert floating-point dtype
interchangeability or establish source fidelity for pending fp64 profiles. -/
namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealProjection
open VeriTile Triton FullSource
set_option maxHeartbeats 2500000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem softmax_kernel_non_inner_erased (input output : Storage) (Y X : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_kernel_non_inner input output Y X M N K BN BK oneTile).eraseDType.toAlgKernel =
      (softmax_kernel_non_inner .fp32 .fp32 Y X M N K BN BK oneTile).eraseDType.toAlgKernel := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_non_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem softmax_kernel_inner_erased (input output : Storage) (Y X : RegionName)
    (M N B : Nat) (oneTile : Bool) :
    (softmax_kernel_inner input output Y X M N B oneTile).eraseDType.toAlgKernel =
      (softmax_kernel_inner .fp32 .fp32 Y X M N B oneTile).eraseDType.toAlgKernel := by
  cases input <;> cases output
  all_goals simp [softmax_kernel_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem softmax_backward_kernel_non_inner_erased (storage : Storage) (P G Y : RegionName)
    (M N K BN BK : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_non_inner storage P G Y M N K BN BK oneTile).eraseDType.toAlgKernel =
      (softmax_backward_kernel_non_inner .fp32 P G Y M N K BN BK oneTile).eraseDType.toAlgKernel := by
  cases storage
  all_goals simp [softmax_backward_kernel_non_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

theorem softmax_backward_kernel_inner_erased (storage : Storage) (P G Y : RegionName)
    (M N BM BN : Nat) (oneTile : Bool) :
    (softmax_backward_kernel_inner storage P G Y M N BM BN oneTile).eraseDType.toAlgKernel =
      (softmax_backward_kernel_inner .fp32 P G Y M N BM BN oneTile).eraseDType.toAlgKernel := by
  cases storage
  all_goals simp [softmax_backward_kernel_inner, ComputeKernel.surfaceBody, ComputeKernel.eraseDType,
    ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, ComputeDType.eraseDType, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def,
    NumericDType.eraseDType, ComparableDType.eraseDType,
    VeriTile.Triton.eraseDType, Region.cast, Except.bind, bind]

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealProjection
