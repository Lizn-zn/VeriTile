import bench.tritonbench_g.kldiv_compute.Proofs.Contracts
import bench.tritonbench_g.kldiv_compute.Proofs.Safety
import bench.tritonbench_g.kldiv_compute.Proofs.Execution
import bench.tritonbench_g.kldiv_compute.Proofs.Precision

namespace VeriTile.Bench.TritonBenchG.KldivCompute

open VeriTile.Triton VeriTile.Examples
open VeriTile.Triton.TiledLoss
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- **The headline**: `kldivergence_kernel` implements the pointwise KL term
`klDivSpec x y = x * log (x / y)` on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose input windows hold `xs`/`ys`
at the active lanes, the translated pointer kernel terminates, every active
output lane holds `klDivSpec (xs i) (ys i)`, and every other memory cell is
unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification kldivergence_kernel_correctness
    (x_ptr y_ptr output_ptr : RegionName)
    (n_elements BLOCK_SIZE : Nat) :
    kldivIO x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      ⊨ fun xs ys i => klDivSpec (xs i) (ys i) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact kldivergence_kernel_flattenOk x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
  · intro bounds s h1 h2 h3 _
    exact kldivergence_kernel_traceSafe x_ptr y_ptr output_ptr n_elements BLOCK_SIZE
      bounds s h1 h2 h3
  · intro s₀ xs ys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := kldivergence_kernel_region_run
      x_ptr y_ptr output_ptr n_elements BLOCK_SIZE s₀ xs ys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

open TypedSource

/-- Independent pointwise real formula for either complete typed source, including
successful flat execution, active-address bounds and all unwritten memory. -/
specification typed_real_correct (x y output : Storage) (v : Bool)
    (_hSource : sourceSupported x y = Bool.true)
    (X Y O : RegionName) (N B : Nat) :
    VeriTile.Spec.Real (typedIO x y output v X Y O N B ⊨
      fun xs ys i => klDivSpec (xs i) (ys i)) := by
  exact typed_real_correct_bridge x y output v X Y O N B

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.KldivCompute
