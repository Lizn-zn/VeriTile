import bench.tritonbench_g.l2_norm_bwd.Proofs.Real
import bench.tritonbench_g.l2_norm_bwd.Proofs.Candidate
import bench.tritonbench_g.l2_norm_bwd.Proofs.Math
namespace VeriTile.Bench.TritonBenchG.L2NormBwd
open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- `_l2_norm_bwd_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring: input `X`,
  upstream gradient `DY`, input gradient `DX`);
* `B = BLOCK_N` — the row window each program owns;
* `read1`/`read2`/`write` — program `pid` (the row) reads and writes its row
  at `pid * stride_x_row` in all three buffers (the host-side
  one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = next_power_of_2(N)`) carry no obligations on either side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def l2BwdIO (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) : MaskedKernelIO₂ where
  kernel := l2_norm_bwd_kernel X DY DX stride_x_row N eps BLOCK_N
  in1 := X
  in2 := DY
  out := DX
  B := BLOCK_N
  read1 := fun pid => pid * stride_x_row
  read2 := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The headline**: `_l2_norm_bwd_kernel` implements the oracle L2-norm
backward `l2BwdSpec` (i.e. `dx = dy·rstd − (Σ dy·x)·(1/(var+eps))·rstd·x` over
the zero-filled active prefix) on its masked IO signature — for every disjoint
flat placement of the three buffers, every program id whose active lanes are
in bounds, and every launch state whose active input-row lanes hold
`xs`/`dys`, the translated pointer kernel terminates, every active output-row
lane `j` holds `l2BwdSpec N BLOCK_N eps xs dys j`, and every other memory cell
is unchanged. Proof: `MaskedKernelIO₂.Implements.intro` assembles the
region-model masked triple with the flat-memory bridge side conditions. -/
specification l2_norm_bwd_kernel_correctness
    (X DY DX : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2BwdIO X DY DX stride_x_row N eps BLOCK_N
      ⊨ fun xs dys i => l2BwdSpec N BLOCK_N eps xs dys i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact l2_norm_bwd_kernel_flattenOk X DY DX stride_x_row N eps BLOCK_N
  · intro bounds s h1 h2 h3 _
    exact l2_norm_bwd_kernel_traceSafe X DY DX stride_x_row N eps BLOCK_N
      bounds s h1 h2 h3
  · intro s₀ xs dys hx hy
    obtain ⟨s1, hexec, hval, hframe⟩ := l2_norm_bwd_kernel_region_run
      X DY DX stride_x_row N eps BLOCK_N s₀ xs dys hx hy
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

def typedIO (storage gradient : Storage) (v : Bool) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₂ :=
  { l2BwdIO X DY DX stride N eps B with
    kernel := (typedBwd storage gradient v X DY DX stride N eps B).eraseDType
    projection := bwd_supported storage gradient v X DY DX stride N eps B }

/-- Both complete kernels compute dy*rstd-dot(dy,x)/(variance+eps)*rstd*x,
where variance is the active row's sum of squares. The contract includes
successful flat execution, exact active input/output bounds and every unwritten
memory cell. Positive B and N≤B describe a complete legal row tile. -/
specification typed_real_correct (storage gradient : Storage) (v : Bool)
    (X DY DX : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat) (_hB : 0 < B) (_hN : N ≤ B) :
    VeriTile.Spec.Real (typedIO storage gradient v X DY DX stride N eps B ⊨ rowFormula N B eps) := by
  intro A hd hr hc pid hxbound hdybound hout hs xs dys s hp hu hx hdy
  obtain ⟨t,he,hv,hf⟩ := l2_norm_bwd_kernel_correctness X DY DX stride N eps B
    A hd hr hc pid hxbound hdybound hout hs xs dys s hp hu hx hdy
  refine ⟨t,?_,?_,hf⟩
  · exact (bwd_flat_exec A storage gradient v X DY DX stride N eps B (A.flattenState s)).trans he
  · intro i hi
    exact (hv i hi).trans (padded_formula N B eps xs dys i hi)

specification original_real_correct (storage gradient : Storage) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    VeriTile.Spec.Real (typedIO storage gradient Bool.false X DY DX stride N eps B ⊨ rowFormula N B eps) :=
  typed_real_correct storage gradient Bool.false X DY DX stride N eps B hB hN

specification candidate_real_correct (storage gradient : Storage) (X DY DX : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    VeriTile.Spec.Real (typedIO storage gradient Bool.true X DY DX stride N eps B ⊨ rowFormula N B eps) :=
  typed_real_correct storage gradient Bool.true X DY DX stride N eps B hB hN

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormBwd
