# relu_strided_buffer

`Kernels.lean` shares both complete branch bodies between the real and FP proofs.
The candidate constant-folds the unused discrete tile-count calculation from the
symbolic source parameters. It preserves the loop, pointer element casts, ReLU
comparison, block pointers, boundary masks, and unspecified padding. This is
partial evaluation, with no GPU performance claim. The source specialization uses
nonnegative input strides, positive output strides, and the default fp32 profile.
The Python wrapper also accepts negative strides and other storage types; those
configurations are outside these contracts.

`Correct.lean` states original and candidate flat-memory contracts for both
one-tile and grid-stride modes: active output equals `max 0` of the initial input,
execution succeeds, accessed addresses are bounded, and all other cells are
preserved. Strides and dimensions remain symbolic. The new public specifications require positive tile size because the source
computes a division by it. Legacy original theorem signatures remain valid in the
total-Nat model; their zero-divisor case does not describe a valid Triton execution.
The original conditions are retained: positive output stride, plus distinct input/output regions and a legal
positive launch grid for grid-stride mode. Host grid composition is outside scope.

The migrated real helpers repair a legacy zero-padding mismatch. Actual omitted
block padding now uses arbitrary unspecified values; inactive values never reach
an active store. No zero-oracle assumption is needed. The historical explicit-zero
load helper remains available separately.

`FPEquiv.lean` uses the fp32 profile and an empty numerical domain. Exact discrete
constant folding preserves the complete state and execution failure for both
entire kernels, including all grid-stride iterations and every memory cell.
`#print_fp_assumptions` reports `none` for each theorem. The real and FP entries
are independent and share only the source definitions.
