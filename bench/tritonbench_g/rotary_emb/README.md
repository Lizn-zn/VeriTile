# rotary_emb

- Source file: `rotary_emb.py`
- Corpus: TritonBench-G v1
- Status: historical store-slice proofs retained; complete-source upgrade in progress.

This directory is the per-kernel workspace for the TritonBench-G full-formalization roadmap.

The historical `rotary_emb_kernel_correctness` theorem concerns four separate
store slices. The complete 34-statement source proof is being developed in
`Kernels.lean` and `Proofs/`; the full execution/IO/safety contracts are not yet
complete. The historical declarations remain unchanged during this work.
The historical slices use their own sine-stride parameters and even-index
trigonometric windows, including for the odd-store slice. Their preserved
contracts therefore do not establish source correspondence for the complete
kernel, whose sine strides and odd trigonometric reads differ as described below.

`Proofs/Math.lean` and `Proofs/Scatter.lean` have passed targeted Lean compiles and
the module axiom audit. They provide the independent pair formula, a nonempty,
collision-free two-sequence/two-head/two-pair strided geometry witness, and
masked scatter readbacks, frames, and register preservation. These results do
not yet establish execution of the full kernel; the source and execution
modules below remain under validation.

The full source has several details that the slice theorem does not establish:

- Sine loads use the **cosine** strides. The two sine-stride arguments are unused
  in the Python kernel and remain unused in the transcription.
- Even and odd results read their corresponding even and odd cosine/sine values.
- Both Q stores precede K loads and the second set of trigonometric loads.
- Only `2 * floor(BLOCK_DMODEL / 2)` logical dimension lanes are generated.
  No lane is generated for an odd last dimension. Its physical cell is preserved
  when it lies outside the actual address footprint; arbitrary overlapping strides
  can alias a different generated lane to that cell.
- Trigonometric loads have shape `[BLOCK_SEQ, 1, BLOCK_DMODEL / 2]` and a sequence
  mask only. They can execute even when every data head is masked.

The new model factors 216 three-storage templates into independent Q/K/Cos/Sin
storage choices. The unchanged templates are compiled in 36 six-profile helper
modules under `Proofs/SourceProfiles_*.lean`, sharing `Proofs/SourceTypes.lean`,
so each successful profile group produces a reusable artifact. This is structural interpreter coverage, not native compilation
certification for every combination; same-format fp8 arithmetic lowering remains
unverified. The candidate only commutes an exact natural-number head-offset
multiplication. Numerical arithmetic and all four stores are retained.
The promotion transcription follows the pinned
[Triton 3.5.1 semantic implementation](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L60-L98):
each multiplication promotes its own operands before the final addition or
subtraction promotes those two results. Storage conversion is explicit when the
computed result dtype differs from Q or K storage.

The intended real formula uses initial Q/K and trigonometric memory. Its domain
requires distinct Q/K/trigonometric output regions (Cos and Sin may alias),
collision-free active even/odd output addresses, and source-mask-specific access
bounds. Strides remain independent nonnegative naturals; negative strides,
machine integer overflow, native `arange` shape restrictions, and valid launches
are separate backend boundaries. The host head grid is based on `HEAD_Q`; a
per-program theorem alone does not certify coverage of extra K heads beyond that
grid.
