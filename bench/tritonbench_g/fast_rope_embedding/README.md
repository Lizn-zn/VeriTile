# fast_rope_embedding

`Kernels.lean` preserves the complete dynamic four-head group from
`_rope_embedding`, including the backward sine sign, both Q loads before the
two stores, masks and exact addresses. The candidate only commutes the
natural-number group-index multiplication. Input, cosine and sine storage
independently range over fp16, bf16 and fp32. The unique `sin1.dtype` is resolved
explicitly to sine storage, including its unchanged type after sign reversal;
Q loads convert to it and stores convert back to Q storage.

`Correct.lean` exposes two independent real specifications for original and
candidate code. Symbolic row/group, strides, sequence length, head size/count
and block size remain parameters. With positive sequence length, a block
covering half the head, actual access bounds and a finite disjoint allocation,
each contract proves successful complete flat execution, both rotary formulas
for every active head/lane and preservation of every other tagged cell. Zero
undefined values are explicit in the flat-memory bridge. Odd head sizes leave
the last channel unchanged. Empty groups still load the coefficient tables
before executing no head iterations.

Coefficients come from initial memory because the source loads them before the
loop. The contract allows both tables to alias Q. A finite regression overlaps
both tables with the first head's written cells and proves that later heads
still use the original coefficients. Other finite witnesses cover three
iterations of a partial group, nonzero row/group, odd head sizes, unallocated
masked coefficient tails, tagged untouched cells and empty groups. They
instantiate the public specifications for both candidates and directions.

`FPEquiv.lean` proves whole typed-state/failure equivalence, with empty input
domain and no numerical FP atoms. All 54 original storage/direction
configurations and their candidate counterparts execute from empty registers
in the abstract algebra, with a nonempty final partial head group. The algebra
witnesses consistency, not physical GPU rounding. Host scheduling, concurrent
races and fixed-width overflow are outside the per-program contract.

Status: the complete source, real-contract and typed-equivalence review passed. Repository builds, source/dtype/layout gates and independent comparator/trust replay passed. No GPU experiment was run in this review.


Validation passed: complete source statement-tree comparison, repository proof
closure, precision and public finite/typed execution witnesses, source/layout
gates and 46 configured audit/tool/layout tests. The three-case independent
comparator/trust replay accepts all 52 selected source/proof/test files, with
current fingerprints and named theorem inventories. Receipt: `Logs/comparator-check-pq1sfkt8`.
This case's source/contract/equivalence review is **complete**, within the
explicit per-program mathematical/typed model and source behaviors above.
