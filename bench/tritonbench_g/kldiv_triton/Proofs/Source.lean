import bench.tritonbench_g.kldiv_triton.Kernels
import bench.tritonbench_g.kldiv_triton.Proofs.Algorithm
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.KldivTriton.Source
open VeriTile.Triton TypedSource
set_option maxHeartbeats 16000000
set_option maxRecDepth 16000
set_option linter.unusedSimpArgs false

def commuteMath (source : Kernel) (position : Nat) (O : RegionName) (stride : Nat) : Kernel :=
  { source with body := source.body.take position ++
    [.assign .ptr [] "O" (.ptrAdd .nil (.ptrBase O)
      (.castIntToNat (.mul .int .nil (.castNatToInt (.constNat stride))
        (.ref .int [] "pid"))))] ++ source.body.drop (position + 1) }

def forwardMath (v : Bool) (Y T O : RegionName) (sy st so N B : Nat)
    (flag : Bool) (reduction : Nat) : Kernel :=
  let source := FullAlgorithm.forward Y T O sy st so N B flag reduction
  if v then commuteMath source 3 O so else source

theorem forward_projects (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    (forward pred target output v Y T O sy st so N B flag reduction).toAlgorithm? =
      .ok (forward pred target output v Y T O sy st so N B flag reduction).toAlgKernel := by
  cases pred <;> cases target <;> cases output <;> cases v <;> rfl

/-- Storage erasure retains both branches, the complete signed loop, all loads and
actual per-chunk destination advancement. -/
theorem forward_projection (pred : Storage) (target : MathStorage) (output : Storage) (v : Bool)
    (Y T O : RegionName) (sy st so N B : Nat) (flag : Bool) (reduction : Nat) :
    (forward pred target output v Y T O sy st so N B flag reduction).eraseDType.toAlgKernel =
      forwardMath v Y T O sy st so N B flag reduction := by
  cases pred <;> cases target <;> cases output <;> cases v
  all_goals simp [forwardMath, commuteMath, forward, forwardOriginal, commuteOutput,
    FullAlgorithm.forward, FullAlgorithm.forwardSurface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
    Except.bind, bind, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, ComputeDType.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, eraseDType, Region.cast]

def backwardMath (v : Bool) (T O : RegionName) (st so N B : Nat) (flag : Bool) : Kernel :=
  let source := FullAlgorithm.backward T O st so N B flag
  if v then commuteMath source 1 O so else source

theorem backward_default_projects (target : Storage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).toAlgorithm? =
      .ok (backwardDefault target output v T O st so N B).toAlgKernel := by
  cases target <;> cases output <;> cases v <;> rfl

theorem backward_default_projection (target : Storage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardDefault target output v T O st so N B).eraseDType.toAlgKernel =
      backwardMath v T O st so N B Bool.false := by
  cases target <;> cases output <;> cases v
  all_goals simp [backwardMath, commuteMath, backwardDefault, backwardDefaultOriginal,
    backwardLog, backwardLogOriginal, commuteOutput,
    FullAlgorithm.backward, FullAlgorithm.backwardSurface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
    Except.bind, bind, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, ComputeDType.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, eraseDType, Region.cast]

theorem backward_log_projects (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).toAlgorithm? =
      .ok (backwardLog target output v T O st so N B).toAlgKernel := by
  cases target <;> cases output <;> cases v <;> rfl

theorem backward_log_projection (target : MathStorage) (output : Storage) (v : Bool)
    (T O : RegionName) (st so N B : Nat) :
    (backwardLog target output v T O st so N B).eraseDType.toAlgKernel =
      backwardMath v T O st so N B Bool.true := by
  cases target <;> cases output <;> cases v
  all_goals simp [backwardMath, commuteMath, backwardDefault, backwardDefaultOriginal,
    backwardLog, backwardLogOriginal, commuteOutput,
    FullAlgorithm.backward, FullAlgorithm.backwardSurface,
    ComputeKernel.eraseDType, ComputeKernel.surfaceBody, ComputeKernel.inputs, ComputeKernel.outputs,
    Except.bind, bind, ComputeKernel.toAlgKernel, ComputeKernel.toAlgorithm?,
    ComputeKernel.fromKernelBody, ComputeStmt.listToAlgorithm?, ComputeStmt.toAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, Kernel.eraseDType,
    Stmt.eraseDTypeList, Stmt.eraseDType, Op.eraseDType.eq_def, ComputeDType.eraseDType,
    NumericDType.eraseDType, ComparableDType.eraseDType, eraseDType, Region.cast]


#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivTriton.Source
