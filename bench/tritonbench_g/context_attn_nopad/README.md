# Context attention without padding

`Kernels.lean` contains the complete typed `_fwd_kernel` and a scale-first
candidate. The candidate only exchanges qk and sm_scale at their fp32 product.
Both retain signed metadata, stored K/V pointers, causal -infinity masking,
natural tl.exp, normalized online dot-plus-add updates, both casts and stores.
All fp16/bf16/fp32 storage branches are present.

`Correct.lean` states two independent initial-array real weighted-sum
specifications. Causally masked weights are zero in the real model. The
contracts prove successful finite flat execution, active output readback and
preservation of every other tagged memory cell. Dimensions, strides and launch
IDs are symbolic. Necessary conditions are BN>0, nonnegative source start,
in-bounds active accesses, injective active output addresses and a finite
disjoint allocation. Input/output aliasing is allowed; there is no precomputed
result or zero-undefined-value assumption. WithBot propagation of masked
arithmetic belongs to the real model; this is not an IEEE NaN or GPU guarantee.

`FPEquiv.lean` proves full-program equivalence using only mul_commute, with
finite checks on the actual reached fp32 operands before the mask. It preserves
opaque exp, dots and casts; no numerical assumption about those operations is
added. Proofs/ contains source correspondence, execution, arithmetic, bounds,
flat-memory and frame arguments, plus complete real and abstract FP execution
witnesses. The witnesses establish that the contracts have instances.

Validation passed: repository proof closures, both source/type regressions,
source-only corpus gates, 25 audit/layout tests and independent comparator/trust
replay of all 59 files across these two cases. Final target inventories and
source fingerprints match `Logs/comparator-check-vqdq8v6p`.
