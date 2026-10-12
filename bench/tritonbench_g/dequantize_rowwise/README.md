# Rowwise dequantization

`Kernels.lean` contains the complete `_dequantize_rowwise` source and candidate:
signed int8 weights, fp32 row scales and fp16 output. The candidate changes
only `pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`. The masked weight load still
omits `other`, and the row-scale load is still unmasked.

`Correct.lean` exposes two real specifications and their full IO windows.
Each active output is the initial row scale times the signed integer weight
converted to real, times `inv_127`. The integer payload is represented on the
mathematical Int channel; floating precision is erased for these real proofs.
The contracts require a finite disjoint allocation, actual-access bounds and
a zero unspecified-value oracle. They conclude successful full flat execution,
active output formulas and preservation of every other tagged memory cell.

Positive `BLOCK_SIZE` and `P2` supply an active lane through which the current
IO interface records the required bound for the unmasked row-scale cell.
Sizes, program IDs, `inv_127` and unused `n_elements` remain parameters.
The host's grid, power-of-two selection and `inv_127 = 1/127` choice are outside
these per-program specifications. All loads precede the sole store, so the
contract does not require distinct input/output names.

`FPEquiv.lean` proves full-program equivalence and prints `none`. Integer
loads and conversion, fp32 numerical operations, the fp16 store, masks and
failure are preserved; the structural rewrite needs no numerical assumptions.
Proof helpers, finite real witnesses and abstract FP executions are in `Proofs/`.
The witnesses start with empty registers and negative integer cells at a
nonzero row, with three active lanes and a masked tail. They establish model
execution, not physical GPU or IEEE rounding behavior. Concurrent launches
and fixed-width index overflow are outside scope.

Validation passed: repository closure, full source expression correspondence,
precision regressions, source-only corpus gates, 31 audit/layout tests and
independent comparator/trust replay of all 10 files. Final target inventories
and source fingerprints match `Logs/comparator-check-nwk3uoa7`.
