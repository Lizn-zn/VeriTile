# cache_transform_triton

- Source: `cache_transform_triton.py`, `prefill_cache_kernel` and `decoding_cache_kernel`.
- Status: compiled and accepted in the correctness-and-equivalence review round.
- Entry points: [Kernels.lean](Kernels.lean), [Correct.lean](Correct.lean), [FPEquiv.lean](FPEquiv.lean).

Both complete kernels retain signed metadata, independent cosine/sine
fp32/fp16/bf16 storage, both outputs, all masks and omitted `other` values.
Decoding uses the original BLOCK_SIZE × HIDDEN_DIM tile. The candidate only
commutes the exact integer product in token indexing; FP assumptions print
`none`. Both programs successfully execute for arbitrary opaque payloads and
all nine storage pairs, with nonempty metadata required by prefill's maximum.

Real correctness erases floating precision and gathers from independent
initial arrays. Prefill uses the source's signed maximum of selected cumulative
boundaries; decoding uses signed row indices already adjusted by the host's
lengths-minus-one. The contracts cover symbolic shapes and strides, successful
flat execution, actual access bounds, both active output values and preservation
of every other typed cell. Outputs must be distinct, stores collision-free,
and active decode rows nonnegative. The current flat bridge requires zero
unspecified values; typed FP equivalence and execution permit arbitrary ones.
Addresses use mathematical Int/Nat, without a fixed-width overflow claim.

The unmasked prefill metadata load is a real host boundary: the host chooses
next_power_of_2(num_seqs) but does not pad cumsum_lengths. [HostSafety.lean](Proofs/HostSafety.lean)
refutes trace safety whenever the allocation is shorter than N_ELEMENTS,
including three allocated entries with a four-entry load. Neither source
variant silently inserts padding or a mask. The host also reuses the cosine
cache's strides for sine and freshly contiguous outputs; the access bounds
retain those actual strides. The theorems do not certify every host call or
launch-wide race freedom.

[Proofs](Proofs) contains internal execution, value, safety and IO proofs.
The previous natural-metadata/single-sequence port is retained only as
[Legacy.lean](Proofs/Legacy.lean), outside the public specification inventory.
`bench/tests/FPCacheTransform.lean` checks signed lowering, mixed storage,
complete typed execution and inactive tails. `bench/tests/CacheTransformComplete.lean`
checks full flat decoding IO and the unpadded metadata failure. Shared signed
and natural reduction regressions are in `bench/tests/DiscreteReductions.lean`.
These are Lean interpreter checks, not GPU measurements.

Independent comparator/trust replay accepted all 20 case modules, 15 affected
shared core modules and three regressions. The audit checks 508 named proof
targets, 6,188 kernel theorem targets and all 1,077 snapshot input hashes.
Receipt: `Logs/comparator-check-dvgpy8_e`. Source gates, coverage fingerprints
and the 377-declaration / 198-file specification inventory also pass.
