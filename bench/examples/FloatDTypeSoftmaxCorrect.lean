/- Real correctness of the original fp32-load, fp64-work softmax pair.
Both original kernels are written here; their numerical annotations are erased
only when stating mathematical correctness. -/
import bench.examples.SoftmaxReciprocalCorrect
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect

open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₁

@[simp] private theorem except_ok_bind {α β ε : Type _} (a : α)
    (f : α → Except ε β) :
    (Except.ok a : Except ε α) >>= f = f a := rfl

def floatStableSoftmaxKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x    := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m    := tl.max(x, axis=0)
  e    := tl.exp(x - m)
  s    := tl.sum(e, axis=0)
  y    := e / s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}

/-- Optimized stable softmax: precompute `1 / s` and use multiplication,
saving per-lane divisions versus `floatStableSoftmaxKernel`. Both implementations are interpreted over the reals for correctness. -/
def floatSoftmaxRecipKernel (xReg yReg : RegionName) (blockSize : Nat) : ComputeKernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(blockSize) + tl.arange(0, $(blockSize))
  x      := (tl.load($(xReg) + offs, dtype=tl.float32)).to(tl.float64)
  m      := tl.max(x, axis=0)
  e      := tl.exp(x - m)
  s      := tl.sum(e, axis=0)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store($(yReg) + offs, (y).to(tl.float32))
}


theorem div_projection (xReg yReg : RegionName) (B : Nat) :
    (floatStableSoftmaxKernel xReg yReg B).eraseDType.toAlgorithm? =
      Except.ok (OnlineSoftmax.batchSoftmaxKernel xReg yReg B).toAlgKernel := by
  simp [floatStableSoftmaxKernel, OnlineSoftmax.batchSoftmaxKernel,
    OnlineSoftmax.stableSoftmaxKernel,
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
