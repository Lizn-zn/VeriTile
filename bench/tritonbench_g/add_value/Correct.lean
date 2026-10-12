import bench.tritonbench_g.add_value.Kernels
import bench.tritonbench_g.add_value.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Both sources in Kernels.lean implement the independent real formula
xs i + value. The masked IO includes termination, active-address bounds and
preservation of every other memory cell. Grid composition is outside the scope. -/

namespace VeriTile.Bench.TritonBenchG.AddValue

open VeriTile.Triton VeriTile.Examples
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- Symbolic input/output windows and the actual tail mask. -/
def addValueIO (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) : MaskedKernelIO₁ where
  kernel := puzzle1_kernel x_ptr output_ptr N BLOCK_SIZE value
  inp := x_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * BLOCK_SIZE
  write := fun pid => pid * BLOCK_SIZE
  mask := fun pid j => pid * BLOCK_SIZE + j.val < N

/-- Mathematical correctness of the original source. -/
specification puzzle1_kernel_correctness
    (x_ptr output_ptr : RegionName)
    (N BLOCK_SIZE : Nat) (value : ℝ) :
    Spec.Real (addValueIO x_ptr output_ptr N BLOCK_SIZE value
      ⊨ fun xs i => xs i + value) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact puzzle1_kernel_flattenOk x_ptr output_ptr N BLOCK_SIZE value
  · intro bounds s h1 h2 _
    exact puzzle1_kernel_traceSafe x_ptr output_ptr N BLOCK_SIZE value
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := puzzle1_kernel_region_run
      x_ptr output_ptr N BLOCK_SIZE value s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hcond _ => hframe r o hcond⟩

/-- The candidate retains the same windows and mask. -/
def commutedIO (x_ptr output_ptr : RegionName) (N B : Nat) (value : ℝ) : MaskedKernelIO₁ :=
  { addValueIO x_ptr output_ptr N B value with
    kernel := puzzle1_commuted_kernel x_ptr output_ptr N B value
    projection := by rfl }

specification puzzle1_commuted_correctness (x_ptr output_ptr : RegionName)
    (N B : Nat) (value : ℝ) :
    Spec.Real (commutedIO x_ptr output_ptr N B value ⊨ fun xs i => xs i + value) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact commuted_flattenOk x_ptr output_ptr N B value
  · intro bounds s hx hout _
    exact commuted_traceSafe x_ptr output_ptr N B value bounds s hx hout
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := puzzle1_kernel_region_run x_ptr output_ptr N B value s xs hx
    refine ⟨s1, (commuted_exec x_ptr output_ptr N B value s).trans he, hv, ?_⟩
    exact fun r o hmiss _ => hf r o hmiss

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.AddValue
