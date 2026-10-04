import bench.examples.FloatDTypeSoftmax.Kernels
/- Use the shared softmax libdevice.exp implementation. The measured
fp32 tl.exp exp-sub relation failed admission under the configured probe
(B = 0.1608954387 > 0.05).
The reciprocal rewrite itself treats exp opaquely, at its stated precision. -/
/- Real correctness of the original fp32-load, fp64-work softmax pair.
Both sources are defined in Kernels.lean; their numerical annotations are erased
only when stating mathematical correctness. -/
import bench.examples.SoftmaxReciprocal.Correct
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect
open VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl


theorem div_projection (xReg yReg : RegionName) (B : Nat) :
    (floatStableSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (OnlineSoftmax.Kernels.batchSoftmaxKernel xReg yReg B).toAlgKernel := by
  simp [floatStableSoftmaxKernel, OnlineSoftmax.Kernels.batchSoftmaxKernel,
    OnlineSoftmax.Kernels.stableSoftmaxKernel,
    ComputeKernel.eraseDType, ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType]
    try rfl

def divIO (B : Nat) : KernelIO₁ where
  kernel := (floatStableSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, div_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_div_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (divIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  simpa only [Spec.Real, KernelIO₁.Implements, divIO, SoftmaxReciprocalCorrect.divIO,
    ComputeKernel.toAlgKernel, div_projection, SoftmaxReciprocalCorrect.div_projection]
    using SoftmaxReciprocalCorrect.softmax_div_correct B hB

theorem recip_projection (xReg yReg : RegionName) (B : Nat) :
    (floatSoftmaxRecipKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (SoftmaxReciprocalCorrect.recipMathKernel xReg yReg B).toAlgKernel := by
  simp [floatSoftmaxRecipKernel, SoftmaxReciprocalCorrect.recipMathKernel,
    ComputeKernel.eraseDType, ComputeStmt.toAlgorithm?, ComputeStmt.listToAlgorithm?,
    ComputeExpr.toAlgorithm?, ComputeOp.toAlgorithm?, ComputeDType.eraseDType,
    Kernel.eraseDType, Stmt.eraseDTypeList, Stmt.eraseDType,
    Op.eraseDType, VeriTile.Triton.eraseDType, NumericDType.eraseDType,
    MemAccess.eraseDType.eq_def, MaskOpt.eraseDType.eq_def]
  repeat' apply And.intro
  all_goals
    rw [Op.eraseDType.eq_def]
    simp [Op.eraseDType.eq_def, MemAccess.eraseDType.eq_def,
      MaskOpt.eraseDType.eq_def, VeriTile.Triton.eraseDType]
    try rfl

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (floatSoftmaxRecipKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, recip_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_recip_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (recipIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  simpa only [Spec.Real, KernelIO₁.Implements, recipIO, SoftmaxReciprocalCorrect.recipIO,
    ComputeKernel.toAlgKernel, recip_projection, SoftmaxReciprocalCorrect.recip_projection]
    using SoftmaxReciprocalCorrect.softmax_reciprocal_correct B hB

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect
