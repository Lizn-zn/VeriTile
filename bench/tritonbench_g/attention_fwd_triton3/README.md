# attention_fwd_triton3

Source: `attention_fwd_triton3.py`'s `_attn_fwd`, including the inlined
`_attn_fwd_inner`. Both public real specifications and the complete FP
specification are proved and independently checked within the scope below.

| Entry | Contract |
|---|---|
| [Kernels.lean](Kernels.lean) | Complete typed original and reciprocal candidate; only END-arm normalization changes |
| [Correct.lean](Correct.lean) | Six named input windows, independent mathematical formulas, and real correctness of both sources |
| [FPEquiv.lean](FPEquiv.lean) | Full-state conditional FP equivalence and its actual intermediate-value domain |

## Real correctness

The six input arrays are Q, K, V and the initial M/L/Out carry. The formula
uses a maximum and base-2 weighted sums over the globally selected KV keys,
including the rescaled carry. It retains the source's exact scale decimal,
signed window distance, both window flags, INIT/END and checked partial KV
blocks. A loop with no iterations preserves initial carry; an executed block
whose row has no kept keys follows the weighted-sum formula instead.

INIT skips old M/L/Out reads. Continuation reads those initial values before
any store. END writes M+log2(L) and Out/L and preserves L memory; otherwise all
three carry values are stored. The original division and reciprocal candidate
satisfy the same formula under total real arithmetic, including zero denominators.
The formula does not read final memory or execute either implementation.

The IO theorem includes successful flat-memory execution, active-address
safety, output readback and preservation outside the written windows. Bounds
cover all M/Out block lanes, including padded rows, because these stores are
unmasked. L needs an allocation bound only when INIT=0 or END=0. Q uses the
actual checked/unchecked load choice; K/V input windows cover the global context.

The static conditions are:

- Positive BLOCK_N; if IS_EVEN_N disables boundary checks, NKV_CTX is divisible
  by BLOCK_N. Checked tails do not require divisibility.
- Positive output inner stride and `BLOCK_DMODEL * innerStride ≤ rowStride`.
  Independent strides, non-unit inner stride and row padding are supported.
- Distinct M/L/Out regions. Input/output aliasing is allowed: all input reads
  finish before the first store.

This is a per-program real-semantics contract. Mathematical integer addresses,
total real log/division and the finite memory view of a stored bottom center
are explicit model choices. Host launch validity, integer overflow, grid races,
concrete IEEE behavior and performance are outside the theorem.

## FP equivalence

The only numerical atom is `div_mul_rcp`. The END branch checks the actual
fp32 accumulator for finiteness and each row denominator for finiteness and
nonzero after the log2 update. The continuation branch requires no normalization
guard. The public printer lists only `div_mul_rcp`.

Both implementations preserve fp16 Q/K/V/Out storage, fp32 M/L and accumulation,
the probability-to-fp16 cast, dot options, opaque exp2, reductions, NaN alpha
repair, signed windows, all pointer advances and every store. The conditional
proof preserves complete state and failure behavior; it is not a bitwise IEEE
identity or a new experiment. The admitted rule selection is unchanged.

`Proofs/FPModel.lean` supplies a consistent abstract backend.
`Proofs/TypedExecution.lean` executes both complete programs with grouped heads,
checked tails, a nonempty signed window and END stores, and proves that the
normalization domain is inhabited. These are Lean witnesses, not GPU emulation.
Their concrete test dimensions do not restrict the symbolic FP theorem.

## Source mapping and historical counterexample

The helper JIT is inlined; storage types and V/Out element casts are explicit.
The resume-arm Out load reads fp16 before converting to fp32. The DSL now
retains the declared pointer storage through load/cast assignment inference,
so the source uses the original load and conversion without a `dtype=` kwarg. `FPAttentionResumeStorage.lean` checks both actual
source ASTs and executes the selected assignment with a nonzero stored value
and observable conversion, so a wrong storage channel cannot pass on zero.
Runtime signed division uses truncation toward zero, while constexpr arithmetic
uses the host's floor convention. Both INIT/END branches, dynamic block-pointer
metadata and signed loop bounds are retained. The textual source gate records
this inlining exception; the complete source decomposition, expression checks
and `bench/tests/FPAttentionFwdTriton3Source.lean` provide the reviewed mapping.

`Proofs/LegacySource.lean` preserves the historical altered transcription and
its limited proofs. It used natural-subtraction window distances and omitted
bounds, tails, casts and NaN repair. `Proofs/SourceReview.lean` connects a checked
row-0/key-1 mask counterexample to that actual old AST. Those historical claims
are excluded from the current public headline inventory. This refutation is
about the old source mapping, not rejection of a two-gates relation.

## Proof organization and validation

`Proofs/RealContract.lean` composes the complete source execution, independent
six-input formula, strided-window safety and output-layout proofs.
`Proofs/RealIOBridge.lean` translates the named six-input/three-output contract
to the existing `UKernelIO` bridge. It does not add an interpreter or a second
flattening proof. Real and FP entries share only Kernels and supporting framework
modules; neither specification imports the other.

Build with `lake build TritonBenchGVerified`. The detailed verification history
and final replay evidence are recorded in [verification_round.md](../verification_round.md).

The final integration batch passes independent comparator replay and trust
checks for 14 files, including both public specifications, named IO assembly,
source connections, model/execution witnesses and the historical source-mask
refutation. All 200 named theorem targets occur in the actual inventories,
and all 685 snapshot input hashes matched at that acceptance. Earlier execution/formula/safety
batches are recorded in the verification history. Logs: `Logs/comparator-check-ffxftyod`.
The verified library builds (3568 jobs), 29 inventory/layout/shard tests and
three relevant FP regressions pass, and all 173 source gates pass.


The load/cast correction was revalidated by the integrated 3,985-job build,
ten focused regressions and independent comparator/trust replay. All eleven
selected attention source/proof files passed in
`Logs/comparator-check-qgooy8if`; the complete batch's passing targets and
snapshot hashes were verified before later changes. The dtype regression's
unrelated local-name parser collision was fixed and accepted in
`Logs/comparator-check-cwzs4h63`. See `verification_round.md` for exact batch
counts and snapshot boundaries.
