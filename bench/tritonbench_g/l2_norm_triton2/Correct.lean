import bench.tritonbench_g.l2_norm_triton2.Proofs.Real
import bench.tritonbench_g.l2_norm_triton2.Proofs.Candidate
import bench.tritonbench_g.l2_norm_triton2.Proofs.Math
import bench.tritonbench_g.l2_norm_triton2.Proofs.SourceChecks
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.L2NormTriton2
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁ VeriTile.Triton.MaskedKernelIO₂

/-- **The forward headline**: `_l2_norm_fwd_1pass_kernel` implements the exact
L2-normalization `l2FwdSpec` (the `Math.TiledL2Norm.l2Norm` oracle over the
zero-padded active prefix) on its masked IO signature — for every disjoint
flat placement of the two buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs`, the
translated pointer kernel terminates, every active output-row lane `j` holds
`l2FwdSpec N BLOCK_N eps xs j`, and every other memory cell is unchanged.
Proof: `Implements.intro` assembles the region-model masked triple with the
bridge side conditions. -/
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2FwdIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2FwdSpec N BLOCK_N eps xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_fwd_1pass_kernel_flattenOk X Y stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 _
    exact l2_norm_fwd_1pass_kernel_traceSafe X Y stride_x_row N eps BLOCK_N
      bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_fwd_1pass_kernel_region_run
      X Y stride_x_row N eps BLOCK_N s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- **The backward headline**: `_l2_norm_bwd_kernel` implements the exact
L2-norm input gradient `l2BwdSpec` (the `Math.TiledL2Norm.l2NormBwd` oracle
over the zero-padded active prefixes) on its masked IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active `X`/`DY` row lanes
hold `xs`/`dys`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other
memory cell is unchanged. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N ⊨
      fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_bwd_kernel_flattenOk X DY DX stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact l2_norm_bwd_kernel_traceSafe X DY DX stride_x_row N eps BLOCK_N
      bounds s h1 h2 h3
  · intro s₀ xs dys hx hdy
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_bwd_kernel_region_run
      X DY DX stride_x_row N eps BLOCK_N s₀ xs dys hx hdy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

/-- The public forward specification covers every logical row element. -/
theorem forward_formula_full_row {N B : Nat} (hNB : N ≤ B) (eps : ℝ)
    (xs : Fin B → ℝ) (i : Fin N) :
    l2FwdSpec N B eps xs (Fin.castLE hNB i) =
      rowForward eps (fun j ↦ xs (Fin.castLE hNB j)) i :=
  padded_forward_eq_row hNB eps xs i

/-- Both backward reductions cover the complete logical row, including tails. -/
theorem backward_formula_full_row {N B : Nat} (hNB : N ≤ B) (eps : ℝ)
    (xs dys : Fin B → ℝ) (i : Fin N) :
    l2BwdSpec N B eps xs dys (Fin.castLE hNB i) =
      rowBackward eps (fun j ↦ xs (Fin.castLE hNB j))
        (fun j ↦ dys (Fin.castLE hNB j)) i :=
  padded_backward_eq_row hNB eps xs dys i

/-- Complete typed forward source with the independently declared masked row IO. -/
def typedFwdIO (storage : Storage) (v : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2FwdIO X Y stride N eps B with
    kernel := (typedFwd storage v X Y stride N eps B).eraseDType
    projection := fwd_supported storage v X Y stride N eps B }

/-- Complete typed backward source with separate X/DY input windows and DX output. -/
def typedBwdIO (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₂ :=
  { l2BwdIO X DY DX stride N eps B with
    kernel := (typedBwd storage gradient v X DY DX stride N eps B).eraseDType
    projection := bwd_supported storage gradient v X DY DX stride N eps B }

/-- Both complete source versions produce x / sqrt(sum(x²) + eps) in real
semantics, with successful flat execution, input/output bounds and a full frame. -/
specification forward_typed_real_correct (storage : Storage) (v : Bool)
    (X Y : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) (_hN : N ≤ B) :
    VeriTile.Spec.Real (typedFwdIO storage v X Y stride N eps B ⊨
      fun xs i => l2FwdSpec N B eps xs i) := by
  intro A hd hr hc pid hin hout hs xs s hp hu hx
  obtain ⟨t, he, hv, hf⟩ := l2_norm_fwd_1pass_kernel_correctness X Y stride N eps B
    A hd hr hc pid hin hout hs xs s hp hu hx
  refine ⟨t, ?_, hv, hf⟩
  exact (fwd_flat_exec A storage v X Y stride N eps B (A.flattenState s)).trans he

/-- Both complete backward sources implement the independent L2 gradient formula
with both reductions, successful flat execution and every unwritten cell preserved. -/
specification backward_typed_real_correct (storage gradient : Storage) (v : Bool)
    (X DY DX : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) (_hN : N ≤ B) :
    VeriTile.Spec.Real (typedBwdIO storage gradient v X DY DX stride N eps B ⊨
      fun xs dys i => l2BwdSpec N B eps xs dys i) := by
  intro A hd hr hc pid hxbound hdybound hout hs xs dys s hp hu hx hdy
  obtain ⟨t, he, hv, hf⟩ := l2_norm_bwd_kernel_correctness X DY DX stride N eps B
    A hd hr hc pid hxbound hdybound hout hs xs dys s hp hu hx hdy
  refine ⟨t, ?_, hv, hf⟩
  exact (bwd_flat_exec A storage gradient v X DY DX stride N eps B
    (A.flattenState s)).trans he

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton2
