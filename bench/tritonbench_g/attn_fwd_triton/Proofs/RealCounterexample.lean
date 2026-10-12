import bench.tritonbench_g.attn_fwd_triton.Proofs.Real
import bench.tritonbench_g.attn_fwd_causal.Proofs.RealFlatCounterexample

/-! Actual source execution in a concrete flat allocation refutes exact causal
masking in the erased real semantics. The input has zero Q/K, unit scales and
V=[0,1]; the finite penalty leaves a strictly positive first output. This small
symbolic fixture is not a GPU launch or a bf16 underflow claim. -/
namespace VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealCounterexample
open VeriTile.Triton
noncomputable section
abbrev parameters := AttnFwdCausal.RealFlatCounterexample.parameters
abbrev allocation := AttnFwdCausal.RealFlatCounterexample.allocation
abbrev initial := AttnFwdCausal.RealCounterexample.initial

def kernel (candidate : Bool) : Kernel :=
  (RealProofs.kernel 2 2 1 parameters candidate).toAlgKernel

theorem complete_source_matches (candidate : Bool) :
    kernel candidate = AttnFwdCausal.RealCounterexample.kernel candidate := by
  rw [kernel, RealProofs.kernel, RealProjection.full_erased_source]
  rfl

theorem positive_output (candidate : Bool) :
    ∃ t, exec (allocation.flattenKernel (kernel candidate))
        (allocation.flattenState initial) = some t ∧ 0 < t.readMem "flat" 10 := by
  rw [complete_source_matches]
  exact AttnFwdCausal.RealFlatCounterexample.positive_output candidate

refutation source_is_exact_causal_mask (candidate : Bool) :
    ∀ t, exec (allocation.flattenKernel (kernel candidate))
      (allocation.flattenState initial) = some t → t.readMem "flat" 10 = 0 := by
  obtain ⟨t, he, hv⟩ := positive_output candidate
  disprove [t, he, ne_of_gt hv]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.AttnFwdTriton.RealCounterexample
