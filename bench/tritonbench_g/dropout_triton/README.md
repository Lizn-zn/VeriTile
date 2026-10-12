# dropout_triton

`Kernels.lean` contains the complete `_dropout` source and its integer-index
commuted candidate. Data/output storage matches at fp16, bf16 or fp32. Keep
storage can be int32 (as in the Python examples) or bool. Integer conditions
use `value != 0`, retaining negative/even nonzero values. The explicit store
cast represents Triton's implicit conversion to the destination precision.
Both load masks, omitted `other` values, eager division and store mask remain.

`Correct.lean` displays original/candidate real specifications with formula
`keep ? initial_x / (1-p) : 0`, actual access bounds, finite allocation,
successful full execution and preservation of every unwritten tagged cell.
Shapes and program index are symbolic; input/output names may alias because
all reads precede the store. Zero unspecified values are a visible premise of
the flat-memory adapter. Real division is Lean's total division, including
`p=1`; this mathematical statement does not assert IEEE division behavior.

`FPEquiv.lean` uses `equiv_decompose`/`fp_prove` through its proof helper. The
candidate only changes `pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`. Floating
arithmetic is retained, so no numerical atom or division-domain premise is
needed. `#print_fp_assumptions` reports none, and failure is preserved.

`bench/tests/DropoutTritonExecution.lean` checks the public real contract for
all storage/keep choices on three active lanes in a four-lane tile: input -3,
keep values 0/2/-1 and p=1/2 yield 0/-6/-6, preserving an integer-tagged tail.
An all-masked p=1 case executes with empty allocations. Precision/mask and
integer truthiness regressions are in `bench/tests/FPDropoutTritonSource.lean`
and `bench/tests/IntegerWhereCondition.lean`. `Proofs/TypedExecution.lean`
inhabits the abstract typed semantics from empty registers for every variant
and arbitrary p; it is not a physical GPU rounding certificate.

Scope is one program, excluding random-mask generation, host launches,
concurrent races and fixed-width index overflow.

Validation passed: repository closure, complete source correspondence,
precision/mask/execution and existing frontend regressions, source-only corpus
gates, 31 audit/layout tests, coverage/manifest checks and independent
comparator/trust replay of 19 files. Current fingerprints and theorem inventories
match `Logs/comparator-check-sce_37j5`.
