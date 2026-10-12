# Elementwise cosine

`Kernels.lean` contains the complete typed `cos_func` from cosine_compute.py
and an index-commuted candidate. Both retain the masked load with unspecified
inactive values, explicit fp32 conversion, intrinsic tl.cos and destination
conversion. The candidate only swaps the exact integer factors in pid*B.
All fp16/bf16/fp32 storage branches and symbolic n/B are covered.

`Correct.lean` gives two independent lane-wise Real.cos specifications. The
masked flat IO contract proves successful finite execution, active outputs
and preservation of all other tagged cells under active-address bounds.
The common flat adapter requires a zero unspecified-value oracle at launch;
region execution/output/frame proofs permit arbitrary unspecified values.
Input/output aliasing is allowed. This is real correctness after precision
erasure, not IEEE equality or concurrent-launch safety.

`FPEquiv.lean` proves complete typed program equivalence. The only rewrite is
exact integer multiplication commutativity, so no numerical atom or numerical
domain check is required; tl.cos remains opaque. Proofs/ holds the projection,
real execution/flat bridge and FP composition proofs, together with successful
real and abstract FP execution witnesses for a nonempty partial block.

Validation passed: repository proof closures, both source/type regressions,
source-only corpus gates, 25 audit/layout tests and independent comparator/trust
replay of all 59 files across these two cases. Final target inventories and
source fingerprints match `Logs/comparator-check-vqdq8v6p`.
