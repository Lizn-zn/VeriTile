import bench.tritonbench_g.softmax_triton3.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.SoftmaxTriton3
open VeriTile.Triton

section Unmasked

open scoped VeriTile.Triton.MaskedKernelIO₁

/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = false`
branch** — the whole kernel-specific audit surface of the unmasked `⊨`
headline:

* `inp`/`out` — which buffer is which argument (the wiring); `mask_ptr` is
  wired into the kernel but never touched on this branch, so it is not an
  interface buffer;
* `B = BLOCK_SIZE` — the row window each program owns;
* `read`/`write` — program `pid` reads and writes its row at
  `pid * row_stride` (the host-side one-program-per-row launch convention;
  this kernel uses one `row_stride` for both buffers);
* `mask` — the active lanes `j < n_cols`, **the same for every program**: the
  row prefix that actually exists in the matrix. Inactive lanes (the padding
  of `BLOCK_SIZE = next_power_of_2(n_cols)`) carry no obligations on either
  side.

The windows and mask are declared, not parsed from the kernel; the headline
**proves** the kernel's actual addressing and masking match them. Buffer sizes
are not signature content: the headline quantifies over every allocation whose
extents cover the active lanes. -/
def softmaxIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₁ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.false
  inp := input_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols

/-- **Headline, `HAS_MASK = false`** (the Python `mask_ptr is None` compile-time
branch): `softmax_kernel` implements the exact stable softmax over the active
row prefix on its masked IO signature — for every disjoint flat placement of
the two buffers, every program id whose active lanes are in bounds, and every
launch state whose active input-row lanes hold `xs`, the translated pointer
kernel terminates, every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE none xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
specification softmax_kernel_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs i => softmaxSpec n_cols BLOCK_SIZE none xs i := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.false
  · intro bounds s h1 h2 _
    exact softmax_kernel_traceSafe output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE hB bounds s h1 h2
  · intro s₀ xs hx
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_region_run output_ptr
      input_ptr mask_ptr row_stride n_cols BLOCK_SIZE hB s₀ xs hx
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end Unmasked

section Masked

open scoped VeriTile.Triton.MaskedKernelIO₂

/-- `softmax_kernel`'s masked **IO signature for the `HAS_MASK = true`
branch** — as `softmaxIO`, but the additive-mask row is a genuine second input
buffer: `in1` is the input matrix, `in2` the mask matrix (both read at
`pid * row_stride`), `out` the output matrix. Active lanes are `j < n_cols`
for every program, as on the unmasked branch. -/
def softmaxMaskedIO (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat) : MaskedKernelIO₂ where
  kernel := softmax_kernel output_ptr input_ptr mask_ptr row_stride n_cols
    BLOCK_SIZE Bool.true
  in1 := input_ptr
  in2 := mask_ptr
  out := output_ptr
  B := BLOCK_SIZE
  read1 := fun pid => pid * row_stride
  read2 := fun pid => pid * row_stride
  write := fun pid => pid * row_stride
  mask := fun _ j => j.val < n_cols

/-- **Headline, `HAS_MASK = true`** (the Python `mask_ptr is not None`
compile-time branch): `softmax_kernel` implements the exact stable softmax
with the additive mask on its masked IO signature — for every disjoint flat
placement of the three buffers, every program id whose active lanes are in
bounds, and every launch state whose active input-row lanes hold `xs` and
active mask-row lanes hold `ms`, the translated pointer kernel terminates,
every active output-row lane `j` holds
`softmaxSpec n_cols BLOCK_SIZE (some ms) xs j`, and every other memory cell is
unchanged. `0 < BLOCK_SIZE` is required: the kernel's `max` reduce (like
`Finset.sup'`) is only defined on non-empty tiles. Proof: `Implements.intro`
assembles the region-model masked triple with the bridge side conditions. -/
specification softmax_kernel_masked_correctness
    (output_ptr input_ptr mask_ptr : RegionName)
    (row_stride n_cols BLOCK_SIZE : Nat)
    (hB : 0 < BLOCK_SIZE) :
    softmaxMaskedIO output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE ⊨
      fun xs ms i => softmaxSpec n_cols BLOCK_SIZE (some ms) xs i := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact softmax_kernel_flattenOk output_ptr input_ptr mask_ptr row_stride
      n_cols BLOCK_SIZE Bool.true
  · intro bounds s h1 h2 h3 _
    exact softmax_kernel_masked_traceSafe output_ptr input_ptr mask_ptr
      row_stride n_cols BLOCK_SIZE hB bounds s h1 h2 h3
  · intro s₀ xs ms hx hm
    obtain ⟨s1, hexec, hval, hframe⟩ := softmax_kernel_masked_region_run
      output_ptr input_ptr mask_ptr row_stride n_cols BLOCK_SIZE hB s₀ xs ms hx hm
    -- scratch is empty, so its frame side condition is vacuous
    exact ⟨s1, hexec, hval, fun r o hout _ => hframe r o hout⟩

end Masked

/-! Typed source contracts. `reciprocal = false` selects the original and `true`
selects the candidate. Each contract covers the entire row when `n ≤ B`, and
otherwise exactly the source's active prefix. All strides and row IDs are symbolic. -/
section TypedUnmasked
open scoped VeriTile.Triton.MaskedKernelIO₁

def typedIO (inputType maskType : Storage) (reciprocal : Bool) (out inp mask : RegionName)
    (stride n B : Nat) : MaskedKernelIO₁ where
  kernel := (typedKernel inputType maskType reciprocal out inp mask stride n B Bool.false).eraseDType
  projection := typed_supported ..
  inp := inp
  out := out
  B := B
  read := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n

/-- Both complete typed sources implement mathematical softmax, including
termination, active-address safety and preservation of every other memory cell. -/
specification softmax_typed_real_correct (inputType maskType : Storage) (reciprocal : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hB : 0 < B) :
    Spec.Real (typedIO inputType maskType reciprocal out inp mask stride n B ⊨
      fun xs i => Real.exp (xs i) /
        ∑ j : Fin B, if j.val < n then Real.exp (xs j) else 0) := by
  refine MaskedKernelIO₁.Implements.intro _ ?_ ?_ ?_
  · exact typed_flattenOk inputType maskType reciprocal out inp mask stride n B Bool.false
  · intro bounds s hin hout _
    exact typed_traceSafe inputType maskType reciprocal out inp mask stride n B hB bounds s hin hout
  · intro s xs hx
    obtain ⟨s1, he, hv, hf⟩ := softmax_kernel_region_run out inp mask stride n B hB s xs hx
    refine ⟨s1, (typed_exec_eq inputType maskType reciprocal out inp mask stride n B
      Bool.false hB s).trans he, fun i hi => ?_, fun r o hm _ => hf r o hm⟩
    exact (hv i hi).trans (softmaxSpec_none_eq_formula n B hB xs i hi)
end TypedUnmasked

section TypedMasked
open scoped VeriTile.Triton.MaskedKernelIO₂

/-- The optional additive mask is a genuine second input channel with independent
storage; all three windows use the source's shared row stride and tail mask. -/
def typedMaskedIO (inputType maskType : Storage) (reciprocal : Bool) (out inp mask : RegionName)
    (stride n B : Nat) : MaskedKernelIO₂ where
  kernel := (typedKernel inputType maskType reciprocal out inp mask stride n B Bool.true).eraseDType
  projection := typed_supported ..
  in1 := inp
  in2 := mask
  out := out
  B := B
  read1 := fun pid => pid * stride
  read2 := fun pid => pid * stride
  write := fun pid => pid * stride
  mask := fun _ j => j.val < n

/-- Both complete typed sources implement softmax of input plus additive mask.
The stable shift uses the input maximum, and cancels from this real formula. -/
specification softmax_typed_masked_real_correct (inputType maskType : Storage) (reciprocal : Bool)
    (out inp mask : RegionName) (stride n B : Nat) (hB : 0 < B) :
    Spec.Real (typedMaskedIO inputType maskType reciprocal out inp mask stride n B ⊨
      fun xs ms i => Real.exp (xs i + ms i) /
        ∑ j : Fin B, if j.val < n then Real.exp (xs j + ms j) else 0) := by
  refine MaskedKernelIO₂.Implements.intro _ ?_ ?_ ?_
  · exact typed_flattenOk inputType maskType reciprocal out inp mask stride n B Bool.true
  · intro bounds s hin hm hout _
    exact typed_masked_traceSafe inputType maskType reciprocal out inp mask stride n B
      hB bounds s hin hm hout
  · intro s xs ms hx hm
    obtain ⟨s1, he, hv, hf⟩ := softmax_kernel_masked_region_run
      out inp mask stride n B hB s xs ms hx hm
    refine ⟨s1, (typed_exec_eq inputType maskType reciprocal out inp mask stride n B
      Bool.true hB s).trans he, fun i hi => ?_, fun r o hm _ => hf r o hm⟩
    exact (hv i hi).trans (softmaxSpec_mask_eq_formula n B hB xs ms i hi)
end TypedMasked

#guard_msgs (drop info) in
#auditModuleAxioms

end VeriTile.Bench.TritonBenchG.SoftmaxTriton3
