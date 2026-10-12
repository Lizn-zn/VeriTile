# Masked multi-head destination copy

`Kernels.lean` contains the complete `_fwd_kernel_destindex_copy_kv` source
from `destindex_copy_kv1.py` and its index-commuted candidate. Input and output
independently support fp16/bf16/fp32, including the destination store cast.
The destination row is loaded as a signed integer. The metadata load is
unmasked; data loads and stores use both `head < head_num` and
`column < head_dim`, and the data load retains `other=0.0`.

`Correct.lean` exposes two real specifications, independent initial copy
formulas and explicit IO windows/masks. They require nonnegative destination
rows, finite disjoint flat allocation, actual-access bounds, zero unspecified
values and an injective output tile layout. The scalar metadata bound remains
required for an entirely inactive tile. Successful execution, active outputs
and preservation of all other tagged cells are proved. Input and output may
alias: the complete tile is read before the store.

`FPEquiv.lean` compares the complete typed kernels. Only the exact integer
source-row product is commuted; numerical values, storage conversions, signed
destination arithmetic, masks, memory effects and failure are preserved.
It needs no numerical assumptions or domain premise and prints `none`.
The per-program model does not establish host grid composition, cross-program
race freedom, fixed-width index overflow or physical GPU/IEEE rounding.

`Proofs/` contains source projections, signed-address execution/readback,
flat IO and concrete execution witnesses. Finite real and abstract typed
witnesses cover all nine storage pairs and both variants, with empty registers,
nonzero source/destination rows and head/column tails in a 4×4 tile. An extra
finite witness has empty payload allocations and a nonempty metadata read
when every head is masked. Precision and mask regressions live in
`bench/tests/FPDestindexCopyKv1Source.lean`.

Validation passed: repository closure, full source expression correspondence,
precision regressions, source-only corpus gates, 31 audit/layout tests and
independent comparator/trust replay of all 13 files. Final target inventories
and source fingerprints match `Logs/comparator-check-nbhwhlk1`.
