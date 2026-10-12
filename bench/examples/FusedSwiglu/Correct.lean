import bench.examples.FusedSwiglu.Kernels
import bench.examples.FusedSwiglu.Proofs.Real
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FusedSwigluCorrect
open VeriTile.Bench.Examples.FusedSwiglu.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.MaskedKernelIO₂

def fusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_fused "x" "y" "out" n B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, fused_projection]
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n

specification fused_swiglu_correct (n B : Nat) :
    Spec.Real (fusedIO n B ⊨ fun xs ys i => (xs i * Real.sigmoid (xs i)) * ys i) := by
  have halg : (fusedIO n B).kernel.toAlgKernel =
      (swiglu_fused_math "x" "y" "out" n B).toAlgKernel := by
    simp [fusedIO, ComputeKernel.toAlgKernel, fused_projection]
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "y" "out" n B
  · intro bounds s hx hy ho hsc
    rw [halg]
    exact fused_traceSafe "x" "y" "out" n B bounds s hx hy ho
  · intro s xs ys hx hy
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := fused_region_run n B s xs ys hx hy
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def unfusedIO (n B : Nat) : MaskedKernelIO₂ where
  kernel := (swiglu_unfused "x" "y" "s" "out" n B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, unfused_projection]
  in1 := "x"
  in2 := "y"
  out := "out"
  B := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  write := fun pid => pid * B
  mask := fun pid i => pid * B + i.val < n
  scratch := [("s", fun pid => pid * B)]

specification unfused_swiglu_correct (n B : Nat) :
    Spec.Real (unfusedIO n B ⊨ fun xs ys i => (xs i * Real.sigmoid (xs i)) * ys i) := by
  have halg : (unfusedIO n B).kernel.toAlgKernel =
      (swiglu_unfused_math "x" "y" "s" "out" n B).toAlgKernel := by
    simp [unfusedIO, ComputeKernel.toAlgKernel, unfused_projection]
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact unfused_flattenOk "x" "y" "s" "out" n B
  · intro bounds s hx hy ho hsc
    rw [halg]
    exact unfused_traceSafe "x" "y" "s" "out" n B bounds s hx hy (hsc ("s", fun pid => pid * B) (by simp [unfusedIO])) ho
  · intro s xs ys hx hy
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := unfused_region_run n B s xs ys hx hy
    refine ⟨s1, he, hv, ?_⟩
    intro r o hmiss hsc
    apply hf r o hmiss
    by_cases hr : r = "s"
    · exact Or.inr (hsc ("s", fun pid => pid * B) (by simp [unfusedIO]) hr)
    · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSwigluCorrect
