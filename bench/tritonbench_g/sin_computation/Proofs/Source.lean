import bench.tritonbench_g.sin_computation.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SinComputation
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def commutedMathKernel (X Y : RegionName) (N B : Nat) : ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = $(B) * pid
  offsets = block_start + tl.arange(0, $(B))
  mask = offsets < $(N)
  x = tl.load(X + offsets, mask=mask)
  output = tl.sin(x)
  tl.store(Y + offsets, output, mask=mask)
}

def mathKernel (v : Bool) (X Y : RegionName) (N B : Nat) : Kernel :=
  if v then (commutedMathKernel X Y N B).toAlgKernel
  else (sin_kernel X Y N B).toAlgKernel

theorem typed_projects (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input output v X Y N B).toAlgorithm? =
      .ok (typedKernel input output v X Y N B).toAlgKernel := by
  cases input <;> cases output <;> cases v <;> rfl

theorem typed_projection (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input output v X Y N B).eraseDType.toAlgKernel =
      mathKernel v X Y N B := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases output <;> cases v
  all_goals simp only [typedKernel, mathKernel, commutedMathKernel, sin_kernel,
    Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input output v X Y N B).eraseDType.toAlgorithm? =
      .ok (typedKernel input output v X Y N B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinComputation
