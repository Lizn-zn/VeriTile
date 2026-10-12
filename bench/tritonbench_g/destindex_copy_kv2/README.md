# Head-masked destination copy

`Kernels.lean` contains the complete `_fwd_kernel_destindex_copy_kv` source
from `destindex_copy_kv2.py` and its index-commuted candidate. Input and output
independently support fp16/bf16/fp32, including the destination store cast.
Destination metadata is a signed integer. Its scalar load is unmasked;
payload loads/stores only check `head < head_num`, with `other=0.0` on loads.
Every column of an active head is accessed. The host passes `head_dim`
directly to `tl.arange`; this port adds neither a width mask nor rounding.

`Correct.lean` gives two independent real copy specifications and explicit
IO windows/mask. They require nonnegative destination rows, finite disjoint
flat allocation, actual-access bounds, zero unspecified values and an
injective output tile layout. The metadata bound is still needed for an
entirely inactive payload tile. Successful execution, active output values
and preservation of every other tagged cell are proved. Input/output aliases
are allowed because the complete tile load precedes the store.

`FPEquiv.lean` proves full-program equivalence. Only the exact natural-number
source-row product is commuted. Signed metadata, all active columns, output
casts, head mask, zero load default, memory effects and failure are preserved.
No numerical atom or domain premise is used; assumptions print `none`.
The statement does not certify physical GPU/IEEE rounding, host compilation
and allocation for arbitrary widths, grid race freedom or fixed-width overflow.

`Proofs/` holds execution/readback, source projections, flat IO and inhabited
execution witnesses. Finite real and abstract typed witnesses cover all nine
input/output storage pairs and both variants, empty registers, distinct
nonzero source/destination rows and a 4×4 tile with a head tail and all columns
active. An extra finite witness has empty payload allocations while still
reading metadata. Source precision/mask regressions are in
`bench/tests/FPDestindexCopyKv2Source.lean`.

Validation passed: repository closure, full source expression correspondence,
precision regressions, source-only corpus gates, 31 audit/layout tests and
independent comparator/trust replay of all 13 files. Final target inventories
and source fingerprints match `Logs/comparator-check-qwtmvj4y`.
