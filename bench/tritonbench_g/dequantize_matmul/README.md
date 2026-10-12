# Dequantize matrix weights

`Kernels.lean` contains the full `dequantize_kernel` from the Python source,
with signed int8 weights, fp32 column scales and fp16/bf16/fp32 output storage.
The candidate commutes only the integer block product in the column mask.
The Python host's later `torch.mm` call is outside this JIT proof.

`Correct.lean` exposes both real specifications and their IO windows. A weight
is loaded through the signed integer channel, converted to a real value, then
multiplied by its initial column scale. The result is proved after erasing
floating precision. The decoded byte payload is represented by mathematical
Int; no integer arithmetic or narrowing is substituted for that load.
Both specifications include successful finite flat execution, active output
readback and preservation of every other tagged memory cell.

The scale read uses a wider column-only mask. Bounds for those reads remain
necessary even when no weight/output row is active. The flattened IO requires
a positive K block so every column is represented, and injective output tile
addresses so different lanes cannot overwrite each other. Sizes and strides
are symbolic. The finite allocation must be disjoint and cover actual accesses,
and the shared flat IO adapter requires zero unspecified values. Input and
output names may alias because all input reads precede the sole store.

`FPEquiv.lean` exposes full-program equivalence and prints `none`. Signed
loads, integer-to-floating conversion, numerical operations, masks, storage
casts and failure are preserved by the index rewrite. There are no numerical
atoms or extra numerical-domain restrictions.

Proof helpers are in `Proofs/`. Finite real witnesses use negative integer
cells, nonzero program IDs and a partially active tile, and also cover empty
weight/output allocations with the still-required scale read. An abstract FP
backend executes the full fp16-output source and candidate from empty registers.
These are model executions, not physical GPU or IEEE rounding measurements.
Concurrent launches and fixed-width index overflow are outside scope.

Validation passed: repository closure, full source expression correspondence,
precision regressions, source-only corpus gates, 31 audit/layout tests and
independent comparator/trust replay of all 13 files. Final target inventories
and source fingerprints match `Logs/comparator-check-6pu188lk`.
