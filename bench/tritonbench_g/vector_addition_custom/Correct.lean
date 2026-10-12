import bench.tritonbench_g.vector_addition_custom.Kernels
import bench.tritonbench_g.vector_addition_custom.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Both sources in Kernels.lean implement mathematical pointwise addition.
The masked IO contract includes termination, active-address bounds and a frame
for all other memory cells. It is per program; the host launch is outside it. -/

namespace VeriTile.Bench.TritonBenchG.VectorAdditionCustom

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Symbolic windows and the actual active-lane mask. -/
def addCustomIO (A B C : RegionName)
    (size BLOCK : Nat) : MaskedKernelIO₂ where
  kernel := _add_kernel A B C size BLOCK
  in1 := A
  in2 := B
  out := C
  B := BLOCK
  read1 := fun pid => pid * BLOCK
  read2 := fun pid => pid * BLOCK
  write := fun pid => pid * BLOCK
  mask := fun pid j => pid * BLOCK + j.val < size

specification add_kernel_correctness
    (A B C : RegionName)
    (size BLOCK : Nat) :
    Spec.Real (addCustomIO A B C size BLOCK
      ⊨ fun as bs i => as i + bs i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_kernel_flattenOk A B C size BLOCK
  · intro bounds s h1 h2 h3 _
    exact add_kernel_traceSafe A B C size BLOCK bounds s h1 h2 h3
  · intro s₀ as bs ha hb
    obtain ⟨s1, hexec, hval, hframe⟩ := add_kernel_region_run
      A B C size BLOCK s₀ as bs ha hb
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- The candidate keeps the same windows and mask. -/
def commutedIO (A B C : RegionName) (size BLOCK : Nat) : MaskedKernelIO₂ :=
  { addCustomIO A B C size BLOCK with
    kernel := _add_commuted_kernel A B C size BLOCK
    projection := by rfl }

specification add_commuted_correctness (A B C : RegionName) (size BLOCK : Nat) :
    Spec.Real (commutedIO A B C size BLOCK ⊨ fun as bs i => as i + bs i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk A B C size BLOCK
  · intro bounds s ha hb hc _
    exact commuted_traceSafe A B C size BLOCK bounds s ha hb hc
  · intro s as bs ha hb
    obtain ⟨s1, he, hv, hf⟩ := add_kernel_region_run A B C size BLOCK s as bs ha hb
    refine ⟨s1, (commuted_exec A B C size BLOCK s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.VectorAdditionCustom
