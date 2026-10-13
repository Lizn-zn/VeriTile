import bench.tritonbench_g.sin_kernel.Proofs.Execution
import bench.tritonbench_g.sin_kernel.Proofs.Real
import bench.tritonbench_g.sin_kernel.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.SinKernel

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- **The headline**: `kernel_function` implements lane-wise `Real.sin` on its
masked IO signature — for every disjoint flat placement of the two buffers,
every program id whose active lanes are in bounds, and every launch state
whose input window holds `xs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `Real.sin (xs i)`, and
every other memory cell is unchanged. Proof:
`MaskedKernelIO₁.Implements.intro` assembles the region-model masked triple
with the flat-memory bridge side conditions. -/
specification kernel_function_correctness
    (x_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    sinIO x_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => Real.sin (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact kernel_function_flattenOk x_ptr output_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact kernel_function_traceSafe x_ptr output_ptr n_elements BLOCK_SIZE
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := kernel_function_region_run
      x_ptr output_ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩


def typedIO (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : MaskedKernelIO₁ :=
  { sinIO X Y N B with
    kernel := (typedKernel input v X Y N B).eraseDType
    projection := typed_supported input v X Y N B }

/-- Independent Real.sin formula for every active lane of either complete typed
source program, including successful flat execution, active bounds and full frame. -/
specification typed_real_correct (input : InputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO input v X Y N B ⊨ fun xs i => Real.sin (xs i)) := by
  intro A hd hr hc pid hx ho hscr xs s hp hu hxs
  obtain ⟨t, he, hv, hf⟩ := kernel_function_correctness X Y N B
    A hd hr hc pid hx ho hscr xs s hp hu hxs
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A input v X Y N B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinKernel
