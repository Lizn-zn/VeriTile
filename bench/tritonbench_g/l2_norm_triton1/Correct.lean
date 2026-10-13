import bench.tritonbench_g.l2_norm_triton1.Proofs.Real
import bench.tritonbench_g.l2_norm_triton1.Proofs.Candidate
import bench.tritonbench_g.l2_norm_triton1.Proofs.Math

namespace VeriTile.Bench.TritonBenchG.L2NormTriton1
open VeriTile.Triton
open VeriTile.Triton.TiledL2Norm
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `_l2_norm_fwd_1pass_kernel`'s masked **IO signature** — the whole
kernel-specific audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_N` — the row window each program owns;
* `read`/`write` — program `row` reads its row of `X` and writes its row of
  `Y`, both at `row * stride_x_row` (the host passes the same row stride for
  both buffers; the one-program-per-row launch convention);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_N = min(MAX_FUSED_SIZE, next_power_of_2(N))`) carry no obligations on
  either side. The store mask equals the load mask, so `writeMask` keeps its
  default.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def l2NormIO (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    MaskedKernelIO₁ where
  kernel := l2_norm_fwd_1pass_kernel X Y stride_x_row N eps BLOCK_N
  inp := X
  out := Y
  B := BLOCK_N
  read := fun pid => pid * stride_x_row
  write := fun pid => pid * stride_x_row
  mask := fun _ j => j.val < N

/-- **The headline**: `_l2_norm_fwd_1pass_kernel` implements the exact L2
normalization over the active row prefix on its masked IO signature — for
every disjoint flat placement of the two buffers, every program id whose
active lanes are in bounds, and every launch state whose active input-row
lanes hold `xs`, the translated pointer kernel terminates, every active
output-row lane `j` holds `l2NormSpec N BLOCK_N eps xs j` (the `Math.*`
oracle `l2Norm` over the masked row), and every other memory cell is
unchanged. No `0 < BLOCK_N` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Implements.intro` assembles the
region-model masked triple with the bridge side conditions. -/
specification l2_norm_fwd_1pass_kernel_correctness
    (X Y : RegionName)
    (stride_x_row N : Nat) (eps : ℝ) (BLOCK_N : Nat) :
    l2NormIO X Y stride_x_row N eps BLOCK_N ⊨
      fun xs i => l2NormSpec N BLOCK_N eps xs i := by
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

def typedIO (storage : Storage) (commuted : Bool) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) : MaskedKernelIO₁ :=
  { l2NormIO X Y stride N eps B with
    kernel := (typedKernel storage commuted X Y stride N eps B).eraseDType
    projection := typed_supported storage commuted X Y stride N eps B }

/-- Complete real execution of either typed program, with exact active-lane
bounds, success and the full flat-memory frame. No positivity restriction on
epsilon is needed for Lean's total real sqrt/division semantics. -/
specification typed_real_correct (storage : Storage) (commuted : Bool)
    (X Y : RegionName) (stride N : Nat) (eps : ℝ) (B : Nat)
    (_hB : 0 < B) (_hN : N ≤ B) :
    Spec.Real (typedIO storage commuted X Y stride N eps B ⊨ rowFormula N B eps) := by
  intro A hd hr hc pid hin hout hscratch xs s hp hu hx
  obtain ⟨t,he,hv,hf⟩ := l2_norm_fwd_1pass_kernel_correctness X Y stride N eps B
    A hd hr hc pid hin hout hscratch xs s hp hu hx
  refine ⟨t,?_,?_,hf⟩
  · exact (typed_flat_exec A storage commuted X Y stride N eps B (A.flattenState s)).trans he
  · intro i hi
    refine (hv i hi).trans ?_
    exact padded_formula N B eps xs i hi

specification original_real_correct (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    Spec.Real (typedIO storage Bool.false X Y stride N eps B ⊨ rowFormula N B eps) :=
  typed_real_correct storage Bool.false X Y stride N eps B hB hN

specification candidate_real_correct (storage : Storage) (X Y : RegionName)
    (stride N : Nat) (eps : ℝ) (B : Nat) (hB : 0 < B) (hN : N ≤ B) :
    Spec.Real (typedIO storage Bool.true X Y stride N eps B ⊨ rowFormula N B eps) :=
  typed_real_correct storage Bool.true X Y stride N eps B hB hN

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.L2NormTriton1
