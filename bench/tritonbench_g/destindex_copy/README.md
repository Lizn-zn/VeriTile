# Destination-indexed dual copy

`Kernels.lean` retains the complete `_fwd_kernel_destindex_copy_kv` body and
an index-commuted candidate. The nope and rope payloads independently use
fp16/bf16/fp32; each output uses its corresponding input's storage dtype.
Destination metadata is loaded as a signed integer. The candidate changes
only the order of the natural-number source-row product.

The source does not use its head counts or head strides. It copies only the
first head, with all `BLOCK_DMODEL_NOPE` and `BLOCK_DMODEL_ROPE` lanes unmasked.
The Lean source preserves this behavior. The host rounds widths up to powers
of two without adding masks, so bounds must cover those actual padding lanes;
this proof does not establish safety for every unpadded host allocation.

`Correct.lean` gives two independent real copy specifications, symbolic
strides/dimensions and explicit IO windows. Each destination row receives the
corresponding initial source tile. The contract requires a nonnegative loaded
row, finite disjoint flat placement, actual-access bounds and zero unspecified
values. Output regions must differ, and each output's lane addresses must be
injective. These are necessary for the two output readback statements. Reads
precede both stores, so input/output aliases are allowed. The conclusion
includes successful execution and preservation of all other tagged cells.

`FPEquiv.lean` proves full typed program equivalence with no numerical atoms
and no domain requirements; its assumptions output is `none`. Payloads,
precisions, signed destination arithmetic, memory effects and failure are
preserved. This does not establish physical GPU rounding, fixed-width index
overflow, host grid composition or race freedom between programs targeting
the same row.

`Proofs/` contains source projections, address congruence, execution/readback,
flat-memory framing and inhabited execution witnesses. Finite real and
abstract FP witnesses cover every supported payload pair and both variants,
starting from empty registers with unequal power-of-two tile widths, a
nonzero source row and a different signed destination row. Source precision
and unmasked-access regressions live in `bench/tests/FPDestindexCopySource.lean`.

Validation passed: repository closure, full source expression correspondence,
precision regressions, source-only corpus gates, 31 audit/layout tests and
independent comparator/trust replay of all 14 files. Final target inventories
and source fingerprints match `Logs/comparator-check-ixadhmi3`.
