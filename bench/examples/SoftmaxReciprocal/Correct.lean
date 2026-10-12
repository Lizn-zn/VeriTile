import bench.examples.SoftmaxReciprocal.Kernels
import bench.examples.SoftmaxReciprocal.Proofs.Real
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Triton.Math.Softmax
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.SoftmaxReciprocalCorrect
open VeriTile.Bench.Examples.SoftmaxReciprocal.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

def divIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, div_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (softmaxRecipKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, recip_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification softmax_div_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (divIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (divIO B).kernel.toAlgKernel = (divMathKernel "x" "y" B).toAlgKernel := by
    simp [divIO, ComputeKernel.toAlgKernel, div_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact div_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact div_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := div_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

specification softmax_reciprocal_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (recipIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (recipIO B).kernel.toAlgKernel = (recipMathKernel "x" "y" B).toAlgKernel := by
    simp [recipIO, ComputeKernel.toAlgKernel, recip_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact recip_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact recip_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := recip_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxReciprocalCorrect
