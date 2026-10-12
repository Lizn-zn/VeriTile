# Scaled cross entropy forward and backward

`Kernels.lean` contains both complete Triton JITs from `cross_entropy2.py`,
with fp16/bf16/fp32 logit and gradient storage. Loss, LSE, z-loss and dloss are fp32.
The candidates only commute the exact integer product in the column offset;
no performance improvement is claimed. Smoothing, split and ignored-label
branches, signed class offsets, casts, masks and writes are retained.

`Correct.lean` has two real specifications for each JIT, one per source variant.
Forward proves the independent initial-array block log-sum-exp and loss formula after scaling logits. Z-loss is written
and specified only when SPLIT is false.
Backward computes probabilities from the input logits and LSE, adds the LSE-square
correction, subtracts the target/smoothing terms and multiplies by the upstream
loss derivative times logit_scale. LSE and dloss are legitimate backward inputs; probabilities
are computed by the source, not supplied as a precomputed answer.

The real contracts include successful finite flat execution, formula readback
and preservation of every unwritten tagged memory cell. Sizes, strides and
launch IDs are parameters. Forward needs positive block size, pairwise distinct outputs
and distinct LSE/logit regions, active-access bounds and the shared adapter's zero
unspecified-value oracle. Backward needs actual-access bounds; ignored labels
skip only dloss, while LSE and active logits are still read. All backward inputs
may alias the output because all reads precede its sole store. The backward
contract imposes no zero-oracle condition. Bounds and masks are described at
the public IO definitions.

`FPEquiv.lean` proves whole-program equivalence for each original/candidate
pair, including execution failure. Exact integer index commutation needs no FP
numerical assumptions; both declarations print `none`. No numerical-domain
restriction is added. The original numerical operations remain opaque.

`Proofs/` contains source projections, independent formulas, execution, loop-free
arithmetic, memory bounds, flat IO and frame proofs. `RealWitness.lean` executes
finite partial blocks from empty registers on every storage and compile-time
branch; the ignored-label backward witness uses an empty dloss allocation.
`TypedExecution.lean` witnesses successful original/candidate execution in a
consistent abstract FP model at a nonzero row/block with a masked tail.
The DSL retains the storage dtype of nested loads and floating bindings
joined across branches. These repairs have separate regression tests.
Precision regressions are in `bench/tests/FPCrossEntropy2Source.lean`.

These are per-program model contracts. Real negative-infinity sentinel handling
is explicit; it is not an IEEE NaN guarantee. The proofs do not establish host
split aggregation, fixed-width index overflow safety, concurrent launch safety,
physical GPU execution or end-to-end statistical error bounds.

Validation passed: complete repository closure, both-JIT source expression
checks, precision regressions, source-only corpus gates, 28 audit/layout tests
and independent comparator/trust replay of all 22 case/shared/regression files. Final target inventories
and source fingerprints match `Logs/comparator-check-5gyzxque`.
