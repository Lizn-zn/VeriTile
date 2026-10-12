# fast_layernorm

The two complete JIT bodies from `fast_layernorm.py` are represented by
`Kernels.lean`. The original and candidate differ only in the order of the
natural-number row/stride multiplication. The input/output storage, weight
storage and bias storage independently range over fp16, bf16 and fp32. Every
load is explicitly converted to fp32 before arithmetic; the final Y/dY store
converts to input storage, and r/mu are fp32.

`Correct.lean` exposes four independent real specifications: original and
candidate forward/backward. Each includes termination, finite allocation and
access bounds, all active outputs, and preservation of every other tagged
memory cell. Forward requires a nonempty row covered by the block, distinct
output regions and positive variance plus epsilon. Backward consumes the
caller's mean and reciprocal standard deviation and includes bounds for the
unused bias load. The flat-memory bridge requires zero undefined values.

The forward source centers **all block lanes**, including zero padding, before
summing squared deviations. Its variance therefore includes the centered
padding contribution. This is not ordinary LayerNorm on padded rows. A checked
complete-source counterexample uses the host-valid N=3, block=4 configuration:
for input values all equal to 3 and epsilon 1, it writes reciprocal standard
deviation 1/2, whereas ordinary LayerNorm would give 1. The formula contract
preserves that source behavior. No host composition or calculus derivative is
claimed for backward.

`FPEquiv.lean` exposes two complete typed program equivalences, with empty
input domains and no numerical FP assumptions. They preserve complete final
states and failure. Typed executions from empty registers cover all 27 storage
combinations for both original/candidate kernels. Finite real executions also
cover a nonzero program id, a masked unallocated input tail, all forward
outputs, nonconstant backward gradients, and an integer-tagged output neighbor.
The execution algebra is a consistency witness, not a physical GPU rounding
model. Host scheduling, races and fixed-width index overflow are outside scope.

Status: the complete source, real-contract and typed-equivalence review passed. Repository builds, source/dtype/layout gates and independent comparator/trust replay passed. No GPU experiment was run in this review.


Validation passed: complete source statement-tree comparison, repository proof
closure, precision and public finite/typed execution witnesses, source/layout
gates and 46 configured audit/tool/layout tests. The three-case independent
comparator/trust replay accepts all 52 selected source/proof/test files, with
current fingerprints and named theorem inventories. Receipt: `Logs/comparator-check-pq1sfkt8`.
This case's source/contract/equivalence review is **complete**, within the
explicit per-program mathematical/typed model and source behaviors above.
