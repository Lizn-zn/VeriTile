import bench.examples.FusedSiLU.Kernels
import bench.examples.FusedSiLU.Proofs.Real
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import VeriTile.Triton
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.FusedSiLUCorrect
open VeriTile.Bench.Examples.FusedSiLU.Kernels
open VeriTile Triton
open scoped VeriTile.Triton.KernelIO₃

def fusedIO (B : Nat) : KernelIO₃ where
  kernel := (fusedSiLUKernel "x" "gate" "residual" "out" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, fused_projection]
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B

specification fused_silu_correct (B : Nat) :
    Spec.Real (fusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) := by
  have halg : (fusedIO B).kernel.toAlgKernel =
      (fusedMathKernel "x" "gate" "residual" "out" B).toAlgKernel := by
    simp [fusedIO, ComputeKernel.toAlgKernel, fused_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact fused_flattenOk "x" "gate" "residual" "out" B
  · intro bounds s hx hg hr ho hsc
    rw [halg]
    exact fused_traceSafe "x" "gate" "residual" "out" B bounds s hx hg hr ho
  · intro s xs gs rs hx hg hr
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := fused_region_run B s xs gs rs hx hg hr
    exact ⟨s1, he, hv, fun r o hmiss _ => hf r o hmiss⟩

def unfusedIO (B : Nat) : KernelIO₃ where
  kernel := (unfusedSiLUKernel "x" "gate" "residual" "z" "silu" "out" B).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, unfused_projection]
  in1 := "x"
  in2 := "gate"
  in3 := "residual"
  out := "out"
  B1 := B
  B2 := B
  B3 := B
  Bout := B
  read1 := fun pid => pid * B
  read2 := fun pid => pid * B
  read3 := fun pid => pid * B
  write := fun pid => pid * B
  scratch := [⟨"z", fun pid => pid * B, B⟩, ⟨"silu", fun pid => pid * B, B⟩]

specification unfused_silu_correct (B : Nat) :
    Spec.Real (unfusedIO B ⊨ fun xs gs rs i =>
      rs i + (xs i * gs i) * Real.sigmoid (xs i * gs i)) := by
  have halg : (unfusedIO B).kernel.toAlgKernel =
      (unfusedMathKernel "x" "gate" "residual" "z" "silu" "out" B).toAlgKernel := by
    simp [unfusedIO, ComputeKernel.toAlgKernel, unfused_projection]
  refine KernelIO₃.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact unfused_flattenOk "x" "gate" "residual" "z" "silu" "out" B
  · intro bounds s hx hg hr ho hsc
    rw [halg]
    exact unfused_traceSafe "x" "gate" "residual" "z" "silu" "out" B bounds s hx hg hr (hsc ⟨"z", fun pid => pid * B, B⟩ (by simp [unfusedIO]))
      (hsc ⟨"silu", fun pid => pid * B, B⟩ (by simp [unfusedIO])) ho
  · intro s xs gs rs hx hg hr
    rw [halg]
    obtain ⟨s1, he, hv, hf⟩ := unfused_region_run B s xs gs rs hx hg hr
    refine ⟨s1, he, hv, ?_⟩
    intro r o hmiss hsc
    apply hf r o hmiss
    · by_cases hr : r = "z"
      · exact Or.inr (hsc ⟨"z", fun pid => pid * B, B⟩ (by simp [unfusedIO]) hr)
      · exact Or.inl hr
    · by_cases hr : r = "silu"
      · exact Or.inr (hsc ⟨"silu", fun pid => pid * B, B⟩ (by simp [unfusedIO]) hr)
      · exact Or.inl hr

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.FusedSiLUCorrect
