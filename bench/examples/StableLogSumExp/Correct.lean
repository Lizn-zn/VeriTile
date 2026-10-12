import bench.examples.StableLogSumExp.Kernels
import bench.examples.StableLogSumExp.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of the direct reference and conditional optimization.
Both sources are in Kernels.lean and retain their bf16 store; their mathematical
interpretation erases dtype. Auxiliary lemmas are in Proofs/Real.lean.
The historical max-shifted version is checked separately in Proofs/LegacyCorrect.lean. -/

namespace VeriTile.Bench.Examples.StableLogSumExpCorrect
open VeriTile.Bench.Examples.StableLogSumExp.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₁

def directIO (B : Nat) : KernelIO₁ where
  kernel := (directLSEKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, direct_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

specification direct_logsumexp_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (directIO B ⊨ fun xs _ => Real.log (∑ i, Real.exp (xs i))) := by
  have halg : (directIO B).kernel.toAlgKernel = (directMathKernel "x" "y" B).toAlgKernel := by
    simp [directIO, ComputeKernel.toAlgKernel, direct_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact direct_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact direct_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := direct_region_run B hB s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

/-- Mathematical interpretation of the exact candidate source. -/
def optimizedIO (B : Nat) : KernelIO₁ where
  kernel := (optimizedLSEKernel "x" "y" B).eraseDType
  projection := candidate_projection "x" "y" B
  inp := "x"
  out := "y"
  Bin := B
  Bout := 1
  read := fun pid => pid * B
  write := fun pid => pid

/-- Both conditional rewrites preserve log-sum-exp over the reals. -/
specification optimized_logsumexp_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (optimizedIO B ⊨ fun xs _ => Real.log (∑ i, Real.exp (xs i))) := by
  refine KernelIO₁.Implements.intro _ (candidate_flattenOk "x" "y" B) ?_ ?_
  · intro bounds s hx hy _
    exact candidate_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := candidate_region_run B hB s xs hx
    exact ⟨t, ht, hv, fun r o ho _ => hf r o ho⟩

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.StableLogSumExpCorrect
