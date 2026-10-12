import FifthSphDraft.Kernels
import VeriTile.Triton.Float.Erasure
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.NormalizedSource
open VeriTile.Triton
noncomputable section
set_option maxRecDepth 30000
set_option maxHeartbeats 8000000
set_option linter.unusedSimpArgs false

def original (X dX g : RegionName) (B N M off stride : Nat) : Kernel :=
  (TypedSource.backward .fp32 .fp32 .fp32 Bool.false X dX g B N M off stride).eraseDType.toAlgKernel

/-- Cache the normalized algorithm with a kernel-checked equality to the
complete typed source. No execution result is used as a mathematical formula. -/
def reduced (X dX g : RegionName) (B N M off stride : Nat) :
    {k : Kernel // original X dX g B N M off stride=k} := by
  refine ⟨?_, ?_⟩
  swap
  conv_lhs =>
    simp (maxSteps := 3000000) only [original,TypedSource.backward,TypedSource.backward_fp32_fp32_fp32,ComputeKernel.surfaceBody,ComputeKernel.eraseDType,
      ComputeKernel.toAlgorithm?,ComputeKernel.toAlgKernel,ComputeKernel.fromKernelBody,
      ComputeStmt.listToAlgorithm?,ComputeStmt.toAlgorithm?,ComputeExpr.toAlgorithm?,ComputeOp.toAlgorithm?,
      Kernel.eraseDType,Stmt.eraseDTypeList,Stmt.eraseDType,
      Op.eraseDType.eq_def,MemAccess.eraseDType.eq_def,MaskOpt.eraseDType.eq_def,
      VeriTile.Triton.eraseDType,NumericDType.eraseDType,ComparableDType.eraseDType,
      IntegralDType.eraseDType,ComputeDType.eraseDType,Region.cast,Bool.false_eq_true,Except.bind,bind,↓reduceIte,Nat.sub_zero]

def algorithm (X dX g : RegionName) (B N M off stride : Nat) : Kernel :=
  (reduced X dX g B N M off stride).val

theorem original_eq_algorithm (X dX g : RegionName) (B N M off stride : Nat) :
    original X dX g B N M off stride=algorithm X dX g B N M off stride :=
  (reduced X dX g B N M off stride).property

theorem statement_count (X dX g : RegionName) (B N M off stride : Nat) :
    (algorithm X dX g B N M off stride).body.length=92 := by rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FifthOrderSphHarmonics.NormalizedSource
