import bench.examples.FloatDTypeSoftmax.Kernels
import bench.examples.FloatDTypeSoftmax.Proofs.Real
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Triton.Math.Softmax
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect
open VeriTile.Bench.Examples.FloatDTypeSoftmax.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

def divIO (B : Nat) : KernelIO₁ where
  kernel := (floatStableSoftmaxKernel "x" "y" B).eraseDType
  projection := erased_projection _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_div_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (divIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact div_flattenOk "x" "y" B
  · intro bounds s hx hy _
    exact div_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := div_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def recipIO (B : Nat) : KernelIO₁ where
  kernel := (floatSoftmaxRecipKernel "x" "y" B).eraseDType
  projection := erased_projection _ (by rfl)
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification float_softmax_recip_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (recipIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact recip_flattenOk "x" "y" B
  · intro bounds s hx hy _
    exact recip_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := recip_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FloatDTypeSoftmaxCorrect
