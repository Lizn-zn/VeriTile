import bench.tritonbench_g.sin_kernel.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SinKernel
open VeriTile.Triton
set_option maxHeartbeats 2000000
set_option maxRecDepth 16000

def commutedMathKernel (X Y : RegionName) (N B : Nat) : ComputeKernel := triton {
  pid = tl.program_id(0)
  block_start = $(B) * pid
  offsets = block_start + tl.arange(0, $(B))
  mask = offsets < $(N)
  x = tl.load(X + offsets, mask=mask)
  output = tl.math.sin(x)
  tl.store(Y + offsets, output, mask=mask)
}

def mathKernel (v : Bool) (X Y : RegionName) (N B : Nat) : Kernel :=
  if v then (commutedMathKernel X Y N B).toAlgKernel
  else (kernel_function X Y N B).toAlgKernel

theorem typed_projects (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).toAlgorithm? =
      .ok (typedKernel input v X Y N B).toAlgKernel := by
  cases input <;> cases v <;> rfl

theorem typed_projection (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).eraseDType.toAlgKernel =
      mathKernel v X Y N B := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgKernel_fromKernelBody]
  cases input <;> cases v
  all_goals simp only [typedKernel, mathKernel, commutedMathKernel, kernel_function,
    Bool.false_eq_true, ↓reduceIte,
    ComputeKernel.surfaceBody, ComputeKernel.toAlgorithm?, ComputeKernel.toAlgKernel,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Except.bind, bind, Kernel.eraseDType, Stmt.eraseDTypeList,
    Stmt.eraseDType, Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def,
    VeriTile.Triton.eraseDType, NumericDType.eraseDType, ComparableDType.eraseDType,
    FloatDType.toTileDType, ComputeDType.eraseDType]
  all_goals rfl

theorem typed_supported (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    (typedKernel input v X Y N B).eraseDType.toAlgorithm? =
      .ok (typedKernel input v X Y N B).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel
