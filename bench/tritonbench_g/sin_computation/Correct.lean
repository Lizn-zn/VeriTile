import bench.tritonbench_g.sin_computation.Proofs.Execution
import bench.tritonbench_g.sin_computation.Proofs.Real
import bench.tritonbench_g.sin_computation.Proofs.Candidate

namespace VeriTile.Bench.TritonBenchG.SinComputation

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- **The headline**: `sin_kernel` implements lane-wise `Real.sin` on its
masked IO signature — for every disjoint flat placement of the two buffers,
every program id whose active lanes are in bounds, and every launch state
whose input window holds `xs` at the active lanes, the translated pointer
kernel terminates, every active output lane holds `Real.sin (xs i)`, and every
other memory cell is unchanged. Proof: `MaskedKernelIO₁.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification sin_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    sinIO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => Real.sin (xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact sin_kernel_flattenOk in_ptr0 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact sin_kernel_traceSafe in_ptr0 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := sin_kernel_region_run
      in_ptr0 out_ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩


def typedIO (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) : MaskedKernelIO₁ :=
  { sinIO X Y N B with
    kernel := (typedKernel input output v X Y N B).eraseDType
    projection := typed_supported input output v X Y N B }

/-- Independent Real.sin formula for every active lane of either complete typed
source program, including successful flat execution, active bounds and full frame. -/
specification typed_real_correct (input : InputStorage) (output : OutputStorage) (v : Bool)
    (X Y : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO input output v X Y N B ⊨ fun xs i => Real.sin (xs i)) := by
  intro A hd hr hc pid hx ho hscr xs s hp hu hxs
  obtain ⟨t, he, hv, hf⟩ := sin_kernel_correctness X Y N B
    A hd hr hc pid hx ho hscr xs s hp hu hxs
  refine ⟨t, ?_, hv, hf⟩
  exact (typed_flat_exec A input output v X Y N B (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SinComputation
