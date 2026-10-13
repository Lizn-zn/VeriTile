import bench.tritonbench_g.rms_norm_triton.Proofs.Real
import bench.tritonbench_g.rms_norm_triton.Proofs.Candidate
import bench.tritonbench_g.rms_norm_triton.Proofs.Math
namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton
open scoped VeriTile.Triton.Masked2DKernelIO₂
/-- `rms_norm_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `in1`/`in2`/`out` — which buffer is which argument (the wiring): the input
  matrix `X`, the per-column weights `W`, the output matrix `Y`;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read1`/`write` — **per-lane strided windows**: program `pid` reads its `X`
  row at `pid * x_stride_r + j * x_stride_c` and writes its `Y` row at
  `pid * y_stride_r + j * y_stride_c` (the host-side one-program-per-row
  launch convention with general row/column strides);
* `read2` — the weight window is **absolute and pid-independent**: every
  program reads `W[j]` (the host passes the same weight vector to all rows);
* `mask` — the active lanes `j < N`, **the same for every program**: the row
  prefix that actually exists in the matrix. Inactive lanes (the padding of
  `BLOCK_SIZE = next_power_of_2(N)`) carry no obligations on either side. The
  store mask equals the load mask, so `writeMask` keeps its default.

The grid is 1-D, so the second program-id axis is an unused parameter: windows
and mask are constant in `pid₁` (the headline's `∀ pid₁` quantification is
vacuous but honest). The windows and mask are declared, not parsed from the
kernel; the headline **proves** the kernel's actual addressing and masking
match them. Buffer sizes are not signature content: the headline quantifies
over every allocation whose extents cover the active lanes. -/
def rmsNormIO (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N : Nat)
    (eps : ℝ) (BLOCK_SIZE : Nat) : Masked2DKernelIO₂ where
  kernel := rms_norm_kernel Y X W y_stride_r y_stride_c x_stride_r x_stride_c
    N eps BLOCK_SIZE
  in1 := X
  in2 := W
  out := Y
  B := BLOCK_SIZE
  read1 := fun pid _ j => pid * x_stride_r + j.val * x_stride_c
  read2 := fun _ _ j => j.val
  write := fun pid _ j => pid * y_stride_r + j.val * y_stride_c
  mask := fun _ _ j => j.val < N

/-- **The headline**: `rms_norm_kernel` implements the exact RMS normalization
over the active row prefix on its masked strided IO signature — for every
disjoint flat placement of the three buffers, every program id whose active
lanes are in bounds, and every launch state whose active input lanes hold `xs`
(the `X` row) and `ws` (the weights), the translated pointer kernel
terminates, every active output lane `j` holds
`rmsNormSpec N BLOCK_SIZE eps xs ws j = (xs j · rrms) · ws j`, and every other
memory cell is unchanged. `0 < y_stride_c` is required: distinct lanes must
scatter to distinct output cells (a zero column stride would self-clobber the
row). No `0 < BLOCK_SIZE` side condition: the kernel's only reduction is a
`sum`, total on empty tiles. Proof: `Masked2DKernelIO₂.Implements.intro`
assembles the region-model masked triple with the flat-memory bridge side
conditions. -/
specification rms_norm_kernel_correctness
    (Y X W : RegionName)
    (y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE : Nat)
    (eps : ℝ) (hyc : 0 < y_stride_c) :
    rmsNormIO Y X W y_stride_r y_stride_c x_stride_r x_stride_c N eps
        BLOCK_SIZE ⊨
      fun _ _ xs ws i => rmsNormSpec N BLOCK_SIZE eps xs ws i := by
  refine Masked2DKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact rms_norm_kernel_flattenOk Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps
  · intro bounds s h1 h2 h3 _
    exact rms_norm_kernel_traceSafe Y X W y_stride_r y_stride_c x_stride_r
      x_stride_c N BLOCK_SIZE eps bounds s h1 h2 h3
  · intro s₀ xs ws hx hw
    obtain ⟨s1, hexec, hval, hframe⟩ := rms_norm_kernel_region_run Y X W
      y_stride_r y_stride_c x_stride_r x_stride_c N BLOCK_SIZE eps hyc s₀
      xs ws hx hw
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

def typedIO (storage weight : Storage) (commuted : Bool) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat) : Masked2DKernelIO₂ :=
  { rmsNormIO Y X W ysr ysc xsr xsc N eps B with
    kernel := (typed storage weight commuted Y X W ysr ysc xsr xsc N eps B).eraseDType
    projection := typed_supported storage weight commuted Y X W ysr ysc xsr xsc N eps B }

/-- Complete strided-row real execution and flat memory frame for either typed
source. The positive row length and complete tile describe a nonempty wrapper
row; the historical empty-tile theorem above is preserved unchanged. -/
specification typed_real_correct (storage weight : Storage) (commuted : Bool)
    (Y X W : RegionName) (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (_hN : 0 < N) (_hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight commuted Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i) := by
  intro A hd hr hc pid₀ pid₁ hxBound hwBound hyBound hscratch xs ws s hp₀ hp₁ hu hx hw
  obtain ⟨t, he, hv, hf⟩ := rms_norm_kernel_correctness Y X W ysr ysc xsr xsc N B eps hyc
    A hd hr hc pid₀ pid₁ hxBound hwBound hyBound hscratch xs ws s hp₀ hp₁ hu hx hw
  refine ⟨t, ?_, ?_, hf⟩
  · exact (typed_flat_exec A storage weight commuted Y X W ysr ysc xsr xsc N eps B
      (A.flattenState s)).trans he
  · intro i hi
    exact (hv i hi).trans (spec_eq_rowFormula N B eps xs ws i)

specification original_real_correct (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (hN : 0 < N) (hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight Bool.false Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i) :=
  typed_real_correct storage weight Bool.false Y X W ysr ysc xsr xsc N eps B hN hNB hyc

specification candidate_real_correct (storage weight : Storage) (Y X W : RegionName)
    (ysr ysc xsr xsc N : Nat) (eps : ℝ) (B : Nat)
    (hN : 0 < N) (hNB : N ≤ B) (hyc : 0 < ysc) :
    Spec.Real (typedIO storage weight Bool.true Y X W ysr ysc xsr xsc N eps B ⊨
      fun _ _ xs ws i => rowFormula N B eps xs ws i) :=
  typed_real_correct storage weight Bool.true Y X W ysr ysc xsr xsc N eps B hN hNB hyc

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.RmsNormTriton
