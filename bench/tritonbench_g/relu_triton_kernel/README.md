# relu_triton_kernel

`Kernels.lean` contains the original Python transcription and a candidate that
inlines ReLU into the guarded store. The original's `pid == 0` write gate is
preserved; this kernel does not fill outputs belonging to other programs.

`Correct.lean` proves both sources implement `max 0 (xs i)` with successful flat
execution, active-address bounds, the tail mask, and a full memory frame. Sizes
remain symbolic; the real headline retains the original positive block-size
condition, also required by the FP headline. This is the default-fp32 source
specialization; other input storage types are not certified. `Proofs/Real.lean` contains the execution and safety lemmas.

`FPEquiv.lean` proves complete fp32 kernel equivalence with successful executions
and equal output memory. `Proofs/FP.lean` also proves equality of the entire memory
and the exact gated/tail frame, including inactive output cells. The only backend
requirement is support for fp32 `≤`; its Boolean result is unrestricted. There
are no numerical assumptions or input-domain restrictions. Temporary registers
may differ because the candidate removes `result`.

Both entries pass Lean and module axiom audits; `#print_fp_assumptions` reports
`none`. This is a structural optimization, with no GPU performance claim.
