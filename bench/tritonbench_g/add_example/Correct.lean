import bench.tritonbench_g.add_example.Kernels
import bench.tritonbench_g.add_example.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Real per-program correctness of both sources in Kernels.lean. The masked
IO contract includes termination, active-address bounds and preservation of
all other memory cells. Grid composition remains outside this contract. -/

namespace VeriTile.Bench.TritonBenchG.AddExample

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- Symbolic windows and mask, verified against the actual source accesses. -/
def addIO (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := add_kernel in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  in1 := in_ptr0
  in2 := in_ptr1
  out := out_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * BLOCK_SIZE
  read2 := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements


/-- Original source implements mathematical pointwise addition. -/
specification add_kernel_correctness
    (in_ptr0 in_ptr1 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (addIO in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact add_kernel_flattenOk in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact add_kernel_traceSafe in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := add_kernel_region_run
      in_ptr0 in_ptr1 out_ptr n_elements BLOCK_SIZE s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Same masked IO for the commuted candidate. -/
def commutedIO (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) : MaskedKernelIO₂ :=
  { addIO in_ptr0 in_ptr1 out_ptr n B with
    kernel := add_commuted_kernel in_ptr0 in_ptr1 out_ptr n B
    projection := by rfl }

/-- The candidate implements the same independent formula. -/
specification add_commuted_correctness (in_ptr0 in_ptr1 out_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO in_ptr0 in_ptr1 out_ptr n B ⊨ fun xs ys i => xs i + ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk in_ptr0 in_ptr1 out_ptr n B
  · intro bounds s hx hy hout _
    exact commuted_traceSafe in_ptr0 in_ptr1 out_ptr n B bounds s hx hy hout
  · intro s xs ys hx hy
    obtain ⟨s1, he, hv, hf⟩ := add_kernel_region_run in_ptr0 in_ptr1 out_ptr n B s xs ys hx hy
    refine ⟨s1, (commuted_exec in_ptr0 in_ptr1 out_ptr n B s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AddExample
