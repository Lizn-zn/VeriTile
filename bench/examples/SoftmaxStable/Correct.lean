import bench.examples.SoftmaxStable.Kernels
import bench.examples.SoftmaxStable.Proofs.Real
import bench.examples.OnlineSoftmax.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.SoftmaxStableCorrect
open VeriTile.Bench.Examples.SoftmaxStable.Kernels
open VeriTile Triton Triton.TiledSoftmax
open scoped VeriTile.Triton.KernelIO₁

def naiveIO (B : Nat) : KernelIO₁ where
  kernel := (naiveSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, naive_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

def stableIO (B : Nat) : KernelIO₁ where
  kernel := (stableSoftmaxKernel "x" "y" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, stable_projection]
  inp := "x"
  out := "y"
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

specification naive_softmax_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (naiveIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (naiveIO B).kernel.toAlgKernel = (naiveMathKernel "x" "y" B).toAlgKernel := by
    simp [naiveIO, ComputeKernel.toAlgKernel, naive_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact naive_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact naive_traceSafe "x" "y" B hB bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := naive_region_run B s xs hx
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

specification stable_softmax_correct (B : Nat) (hB : 0 < B) :
    Spec.Real (stableIO B ⊨ fun xs i => Real.exp (xs i) / ∑ j, Real.exp (xs j)) := by
  have halg : (stableIO B).kernel.toAlgKernel =
      (OnlineSoftmax.Kernels.batchSoftmaxKernel "x" "y" B).toAlgKernel := by
    simp [stableIO, ComputeKernel.toAlgKernel, stable_projection]
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact OnlineSoftmax.batchSoftmax_flattenOk "x" "y" B
  · intro bounds s hx hy _
    rw [halg]; exact OnlineSoftmax.batchSoftmax_traceSafe "x" "y" B bounds s hx hy
  · intro s xs hx
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := OnlineSoftmax.batchSoftmax_region_run B hB s xs hx
    refine ⟨s1, he, fun i => ?_, fun r o hmiss _ => hf r o hmiss⟩
    change s1.readMem "y" (s.pid * B + i.val) = _
    rw [hv i, (OnlineSoftmax.online_softmax_recurrence_eq_batch hB xs).1,
      (OnlineSoftmax.online_softmax_recurrence_eq_batch hB xs).2]
    change stableSoftmaxMath xs (tileMax hB xs) i = naiveSoftmaxMath xs i
    exact (congrFun (naive_eq_stable xs (tileMax hB xs)) i).symm

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.SoftmaxStableCorrect
