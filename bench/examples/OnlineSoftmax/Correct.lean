import bench.examples.OnlineSoftmax.Kernels
import bench.examples.OnlineSoftmax.Proofs.Real
import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Float
import VeriTile.Triton.DSL
import VeriTile.Triton.KernelLemmas
import VeriTile.Triton.Memory.KernelSpec
import VeriTile.Triton.Math.Softmax
import VeriTile.Examples.Common
import VeriTile.Meta.Specification
import VeriTile.Meta.StatementAudit

/-! Real correctness of the sources in Kernels.lean.
IO contracts and mathematical formulas are stated below; execution and memory
lemmas are in Proofs/Real.lean. Numerical casts are interpreted over the reals. -/

namespace VeriTile.Bench.Examples.OnlineSoftmax
open VeriTile.Bench.Examples.OnlineSoftmax.Kernels
open VeriTile.Triton VeriTile.Triton.TiledSoftmax
open VeriTile.Examples

section OnlineSoftmax.kernelIO

open VeriTile.Triton.KernelIO₁ (Implements)
open scoped VeriTile.Triton.KernelIO₁

/-- Real correctness of the complete online source, including the output store
and cell-level frame. The flat-memory specification below transports this result
through the separately proved address bounds; both proofs are independent of FP admission. -/
specification online_softmax_output_correctness (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ) (hx : InputLoadedAt s "x" N xs) :
    Spec.Real (∃ t, exec (onlineSoftmaxKernel "x" "y" N) s = some t ∧
      (∀ i : Fin N, t.readMem "y" (s.pid * N + i.val) =
        Real.exp (xs i - tileMax hN xs) / ∑ j, Real.exp (xs j - tileMax hN xs)) ∧
      (∀ (r : RegionName) o, (r ≠ "y" ∨ ∀ i : Fin N, o ≠ s.pid * N + i.val) → t.mem r o = s.mem r o)) := by
  exact online_output_run N hN s xs hx

/-- Read and write the row at pid * B; input and output windows both have B elements. -/
def batchSoftmaxIO (B : Nat) : KernelIO₁ where
  kernel := batchSoftmaxKernel ⟨"x"⟩ ⟨"y"⟩ B
  inp := ⟨"x"⟩
  out := ⟨"y"⟩
  Bin := B
  Bout := B
  read := fun pid => pid * B
  write := fun pid => pid * B

/-- The batch output equals the online recurrence normalization. The recurrence starts at M = -infinity, L = 0 and updates M_next = max(M, x), L_next = exp(M - M_next) * L + exp(x - M_next). -/
specification online_softmax_correctness (B : Nat) (hB : 0 < B) :
    Spec.Real (batchSoftmaxIO B ⊨ fun xs i =>
      Real.exp (xs i - (onlineSoftmaxM xs B).unbotD 0)
        / (onlineSoftmaxL xs B).unbotD 0) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact batchSoftmax_flattenOk ⟨"x"⟩ ⟨"y"⟩ B
  · intro bounds s h1 h2 _
    exact batchSoftmax_traceSafe ⟨"x"⟩ ⟨"y"⟩ B bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := batchSoftmax_region_run B hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- The complete two-pass source has the same bounded row windows as batch softmax. -/
def onlineSoftmaxIO (N : Nat) : KernelIO₁ where
  kernel := onlineSoftmaxKernel "x" "y" N
  inp := "x"
  out := "y"
  Bin := N
  Bout := N
  read := fun pid => pid * N
  write := fun pid => pid * N

/-- Real softmax correctness for the complete pointer kernel, including successful
execution, both input passes in bounds, all output stores and the flat-memory frame. -/
specification online_softmax_flat_correctness (N : Nat) (hN : 0 < N) :
    Spec.Real (onlineSoftmaxIO N ⊨ fun xs i =>
      Real.exp (xs i - tileMax hN xs) / ∑ j, Real.exp (xs j - tileMax hN xs)) := by
  refine KernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact Memory.flattenOk "x" "y" N
  · intro bounds s hx hy _
    exact Memory.traceSafe "x" "y" N bounds s hx hy
  · intro s xs hx
    obtain ⟨t, ht, hv, hf⟩ := online_softmax_output_correctness N hN s xs hx
    exact ⟨t, ht, hv, fun r o h _ => hf r o h⟩

#axiomsClean online_softmax_flat_correctness

end OnlineSoftmax.kernelIO

#axiomsClean online_softmax_correctness
#axiomsClean online_softmax_correct
#axiomsClean online_softmax_output_correctness
#stmtSurfaceSubset online_softmax_correctness ⊆
  [Spec.Real, batchSoftmaxIO, VeriTile.Triton.KernelIO₁.Implements,
   VeriTile.Triton.KernelIO₁.Bin, VeriTile.Triton.KernelIO₁.Bout,
   onlineSoftmaxM, onlineSoftmaxL, Zero.toOfNat0]

end VeriTile.Bench.Examples.OnlineSoftmax
