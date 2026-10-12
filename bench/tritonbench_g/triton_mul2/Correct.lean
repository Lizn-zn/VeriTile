import bench.tritonbench_g.triton_mul2.Kernels
import bench.tritonbench_g.triton_mul2.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! All four sources in Kernels.lean implement the real formula 2 * xs i.
The masked IO contracts include termination, active-address bounds and the
memory frame. The in-place contracts read the original values before writing
back to the same region. The host grid composition is outside these contracts. -/

namespace VeriTile.Bench.TritonBenchG.TritonMul2

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Symbolic windows and tail mask for ordinary output. -/
def mul2IO (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_kernel in_ptr0 out_ptr n_elements BLOCK_SIZE
  inp := in_ptr0
  out := out_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

specification mul2_kernel_correctness
    (in_ptr0 out_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2IO in_ptr0 out_ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_kernel_flattenOk in_ptr0 out_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact mul2_kernel_traceSafe in_ptr0 out_ptr n_elements BLOCK_SIZE
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_kernel_region_run
      in_ptr0 out_ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- Both roles share the same region for in-place output. -/
def mul2InplaceIO (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := mul2_inplace_kernel ptr n_elements BLOCK_SIZE
  inp := ptr
  out := ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < n_elements

specification mul2_inplace_kernel_correctness
    (ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    Spec.Real (mul2InplaceIO ptr n_elements BLOCK_SIZE
      ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact mul2_inplace_kernel_flattenOk ptr n_elements BLOCK_SIZE
  · intro bounds s h1 _ _
    exact mul2_inplace_kernel_traceSafe ptr n_elements BLOCK_SIZE
      bounds s h1
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := mul2_inplace_kernel_region_run
      ptr n_elements BLOCK_SIZE s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- The ordinary candidate retains the original masked contract. -/
def commutedIO (in_ptr0 out_ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2IO in_ptr0 out_ptr n B with
    kernel := mul2_commuted_kernel in_ptr0 out_ptr n B
    projection := by rfl }

specification mul2_commuted_correctness (in_ptr0 out_ptr : RegionName) (n B : Nat) :
    Spec.Real (commutedIO in_ptr0 out_ptr n B ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk in_ptr0 out_ptr n B
  · intro bounds s hx hout _
    exact commuted_traceSafe in_ptr0 out_ptr n B bounds s hx hout
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := mul2_kernel_region_run in_ptr0 out_ptr n B s xs hx
    refine ⟨s1, (commuted_exec in_ptr0 out_ptr n B s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

/-- The in-place candidate keeps inp = out = ptr, with the same tail mask. -/
def inplaceCommutedIO (ptr : RegionName) (n B : Nat) : MaskedKernelIO₁ :=
  { mul2InplaceIO ptr n B with
    kernel := mul2_inplace_commuted_kernel ptr n B
    projection := by rfl }

specification mul2_inplace_commuted_correctness (ptr : RegionName) (n B : Nat) :
    Spec.Real (inplaceCommutedIO ptr n B ⊨ fun xs i => 2 * xs i) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact inplace_commuted_flattenOk ptr n B
  · intro bounds s hx _ _
    exact inplace_commuted_traceSafe ptr n B bounds s hx
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := mul2_inplace_kernel_region_run ptr n B s xs hx
    refine ⟨s1, (inplace_commuted_exec ptr n B s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.TritonMul2
