import bench.tritonbench_g.rmsnorm_fused.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsnormFused
open VeriTile.Triton
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def mathKernel (v : Bool) (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) : Kernel :=
  let source := (rms_norm_fwd_fused X Y W stride N B eps).toAlgKernel
  { source with inputs := [X, W], outputs := [Y], body :=
    if v then source.body.take 1 ++
      [.assign .ptr [] "Y" (.ptrAdd .nil (.ptrBase Y)
        (.mul .nat .nil (.constNat stride) (.ref .nat [] "row")))] ++ source.body.drop 2
    else source.body }

theorem typed_projects (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X Y W stride N B eps).toAlgorithm? =
      .ok (typedKernel input weight output v X Y W stride N B eps).toAlgKernel := by
  cases input <;> cases weight <;> cases output <;> cases v <;> rfl

/-- All floating storage profiles erase to the complete loop-shaped source. -/
theorem typed_projection (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel =
      mathKernel v X Y W stride N B eps := by
  cases input <;> cases weight <;> cases output <;> cases v
  all_goals simp [mathKernel, typedKernel, originalTyped, rms_norm_fwd_fused,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs, Except.bind, bind,
    ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?, ComputeKernel.fromKernelBody,
    ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?, ComputeExpr.toAlgorithm?,
    ComputeOp.toAlgorithm?, Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType.eq_def, ComputeDType.eraseDType, NumericDType.eraseDType,
    ComparableDType.eraseDType, eraseDType, Region.cast, List.cons_append, List.nil_append]


theorem typed_supported (input weight output : Storage) (v : Bool)
    (X Y W : RegionName) (stride N B : Nat) (eps : ℝ) :
    (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgorithm? =
      Except.ok (typedKernel input weight output v X Y W stride N B eps).eraseDType.toAlgKernel := by
  simp only [ComputeKernel.eraseDType, typed_projects,
    ComputeKernel.toAlgorithm?_fromKernelBody, ComputeKernel.toAlgKernel_fromKernelBody]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsnormFused
