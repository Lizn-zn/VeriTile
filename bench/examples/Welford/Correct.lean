import bench.examples.Welford.Kernels
import bench.examples.Welford.Proofs.Real
import VeriTile.Triton
import VeriTile.Examples.Common
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.WelfordCorrect
open VeriTile.Bench.Examples.Welford.Kernels
open VeriTile Triton
open VeriTile.Triton.TiledReduction.WelfordRec
open VeriTile.Examples (InputRowLoadedAt onlineWelfordLoopBody)
open scoped VeriTile.Triton.KernelIO₁ₓ₂

def twopassIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (twopassWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, twopass_projection]
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification twopass_welford_correct (N stride : Nat) :
    Spec.Real (twopassIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (twopassIO N stride).kernel.toAlgKernel =
      (twopassMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    simp [twopassIO, ComputeKernel.toAlgKernel, twopass_projection]
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact twopass_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact twopass_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := twopass_region_run N stride s xs hx
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

def onlineIO (N stride : Nat) : KernelIO₁ₓ₂ where
  kernel := (onlineWelfordKernel "x" "mean" "var" N stride).eraseDType
  projection := by simp [ComputeKernel.toAlgKernel, online_projection]
  inp := "x"
  out1 := "mean"
  out2 := "var"
  Bin := N
  Bout1 := 1
  Bout2 := 1
  read := fun pid => pid * stride
  write1 := fun _ => 0
  write2 := fun _ => 0

specification online_welford_correct (N stride : Nat) :
    Spec.Real (onlineIO N stride ⊨ fun xs =>
      ((fun _ => (∑ i, xs i) / N),
       (fun _ => (∑ i, (xs i - (∑ j, xs j) / N) ^ 2) / N))) := by
  have halg : (onlineIO N stride).kernel.toAlgKernel =
      (onlineMathKernel "x" "mean" "var" N stride).toAlgKernel := by
    simp [onlineIO, ComputeKernel.toAlgKernel, online_projection]
  refine KernelIO₁ₓ₂.Implements.intro _ ?_ ?_ ?_
  · rw [halg]; exact online_flattenOk "x" "mean" "var" N stride
  · intro bounds s hx hm hv _
    rw [halg]; exact online_traceSafe "x" "mean" "var" N stride s bounds hx hm hv
  · intro s xs hx
    rw [halg]
    obtain ⟨t, he, hm, hv, hf⟩ := online_region_run N stride s xs hx
    refine ⟨t, he, ?_, ?_, ?_⟩
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassMean] using hm
    · intro i
      have hi : i.val = 0 := Nat.lt_one_iff.mp i.isLt
      simpa [hi, twoPassS, twoPassMean] using hv
    · intro r o hm hv _
      apply hf r o
      · rcases hm with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)
      · rcases hv with h | h
        · exact Or.inl h
        · exact Or.inr (h ⟨0, Nat.zero_lt_one⟩)

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.Examples.WelfordCorrect
