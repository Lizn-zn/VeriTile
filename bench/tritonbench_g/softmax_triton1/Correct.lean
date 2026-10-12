import bench.tritonbench_g.softmax_triton1.Kernels
import bench.tritonbench_g.softmax_triton1.Proofs.Real
import VeriTile.Meta.StatementAudit

/-! Per-program real correctness of the sources in Kernels.lean.
The host launch is outside these contracts; every row id and positive tile size
is covered under the declared masked bounds. -/

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton1
open VeriTile.Triton
open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `softmax_kernel`'s masked **IO signature** — the whole kernel-specific
audit surface of the `⊨` headline:

* `inp`/`out` — which buffer is which argument (the wiring);
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads its row at `pid * input_row_stride` and
  writes it at `pid * output_row_stride` (the host-side one-program-per-row
  launch convention);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def softmaxIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) :
    MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr input_row_stride
    output_row_stride n_cols BLOCK_SIZE
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * input_row_stride
  write := fun pid => pid * output_row_stride
  mask := fun _ j => j.val < n_cols

/-- The kernel computes mathematical softmax over the active prefix, with
lane-wise bounds, termination and preservation of every other memory cell.
`0 < BLOCK_SIZE` makes the reduction tile nonempty; the active mask may be empty.
For `n_cols ≤ BLOCK_SIZE`, the formula covers the complete row. -/
specification softmax_kernel_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    Spec.Real (softmaxIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin BLOCK_SIZE, if j.val < n_cols then Real.exp (xs j) else 0) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s h1 h2 _
    exact softmax_kernel_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_region_run output_ptr
      input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    refine ⟨s1, hexec, fun i hi => ?_, fun r o hout _ => hframe r o hout⟩
    exact (hval i hi).trans (softmaxSpec_eq_formula n_cols BLOCK_SIZE hB xs i hi)

/-- The candidate retains the same masked windows and memory contract. -/
def reciprocalIO (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₁ :=
  { softmaxIO output_ptr input_ptr input_row_stride output_row_stride n_cols BLOCK_SIZE with
    kernel := softmax_reciprocal_kernel output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
    projection := by rfl }

specification softmax_reciprocal_correctness
    (output_ptr input_ptr : RegionName)
    (input_row_stride output_row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    Spec.Real (reciprocalIO output_ptr input_ptr input_row_stride output_row_stride
        n_cols BLOCK_SIZE ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin BLOCK_SIZE, if j.val < n_cols then Real.exp (xs j) else 0) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact reciprocal_flattenOk output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE
  · intro bounds s hin hout _
    exact reciprocal_traceSafe output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB bounds s hin hout
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := softmax_kernel_region_run output_ptr input_ptr
      input_row_stride output_row_stride n_cols BLOCK_SIZE hB s xs hx
    refine ⟨s1, ?_, fun i hi => (hv i hi).trans
      (softmaxSpec_eq_formula n_cols BLOCK_SIZE hB xs i hi), fun r o hout _ => hf r o hout⟩
    exact (reciprocal_exec output_ptr input_ptr input_row_stride
      output_row_stride n_cols BLOCK_SIZE hB s).trans he

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.SoftmaxTriton1
