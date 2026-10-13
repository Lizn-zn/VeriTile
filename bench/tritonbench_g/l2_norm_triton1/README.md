# l2_norm_triton1

`Kernels.lean` retains the historical transcription and defines a shared typed source family for fp16, bf16, fp32 and fp64 storage. X and Y have the same dtype, as in the Python wrapper's `empty_like`. The source loads with a zero mask, converts to fp32, sums the squared masked values, computes `1 / sqrt(var + eps)`, multiplies the loaded row and converts back to the output storage. Runtime epsilon explicitly converts to fp32. Both pointers use `stride_x_row`.

The candidate commutes the natural-number multiplication in the two row-base addresses. It changes no floating operation or reduction order; no speedup is claimed. `FPEquiv.lean` uses the default fp32 profile and states equality of complete typed executions, including failures and every register/memory cell, with no domain sites or numerical FP atoms.

`Correct.lean` gives both programs the independent active-row formula `x[i] / sqrt(sum(active x[j]^2) + eps)`, successful flat execution, exact active-lane bounds and a frame for all other memory. The new full-row contracts require a positive tile and `N ≤ BLOCK_N`; the historical contracts retain their original signatures, including total empty-tile behavior. The real interpretation uses total real square root and division and erases floating casts; it asserts no IEEE approximation bound or positivity condition on epsilon.

The proof boundary is one kernel invocation. Host reshape/contiguity handling, allocation, launch-grid selection and the wrapper's maximum tile-size check are not verified. Allocation regions must be disjoint and cover the active addresses, with the standard zero undefined-memory launch state. Strides use nonnegative natural-number indexing. The wrapper's tested float32 cases are included in the four supported storage profiles.

`Proofs/Real.lean` contains the preserved execution, sum-of-squares, bounds and frame proofs. `Proofs/Source.lean` checks the typed projection, IO ports and epsilon conversion; `Proofs/Candidate.lean` transports the full flat execution through address commutation. `Proofs/FP.lean` proves the exact pointer-expression congruence used by the public FP entry.

`Proofs/NumericWitness.lean` checks successful complete typed execution of both programs with fp16 storage, a nonzero row, stride 3 and a two-lane tile with one active lane. Input 3 and epsilon 7 produce the actual output cell 3/4; the masked input value 97 contributes nothing, and the masked output cell retains 99. The backend uses exact real square root and arithmetic with a negative-infinity carrier and identity casts; this is a numerical execution witness, not an IEEE rounding claim.

The public `Correct` and `FPEquiv` entry dependency build passes (3445 jobs). Five public contracts are registered with validated source fingerprints and `test-gap` status. Independent comparator replay remains pending.
