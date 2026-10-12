# chunk_retention_ops

All four JIT sources in `chunk_retention_ops.py` have complete typed original and candidate implementations in `Kernels.lean`. `Correct.lean` exposes eight real correctness specifications; `FPEquiv.lean` exposes four conditional FP equivalences. Execution and memory proofs live in `Proofs/`.

| Source | Mathematical contract | Candidate and admitted atom |
|---|---|---|
| Forward state | Seeded decayed recurrence, pre-update history, optional final state, shorter final partial-chunk decay | Commute decayed carry and weighted dot; `add_commute` |
| Forward output | Row-decayed Q·H plus causal distance-decayed Q·K·V over all K blocks, then scale | Put the final scale factor first; `mul_commute` |
| Backward state | Descending decayed recurrence, storing history before each update; all BT/BK/BV independent, including an empty loop | Commute decayed carry and scaled-query/weighted-gradient dot; `add_commute` |
| Backward gradients | Complete V loop, all dQ/dK/dV outputs and their query/key/causal decay weights | Commute final weighted dQ carry and score-gradient dot; `add_commute` |

The head rate is `1 - 2^(-5 - head % H)`. Signed indices, intrinsic exp2/log2, casts, TF32-disabled dots, masks and stores remain in the typed source. The explicit `arange(...).to(tl.int32)` states the signed index type. Reverse time is represented by an ascending counter visiting `NT-1-j`.

The backward-state source differs from `chunk_retention`: it stores every history tile, reads scaled Q and row-weighted dO, and applies full-BT decay to the carry at each update. It has no square-tile restriction or post-loop-index use. Forward state shortens the last chunk's decay when needed; backward state deliberately keeps full-BT decay, as the Python source does.

The other three kernel bodies are identical to their `chunk_retention` counterparts after parameter renaming. `Proofs/SourceComparison.lean` proves equality of the complete typed source functions, including signatures and both candidates. Their real execution proofs are reused through those checked bodies; the public entries still show this case's implementation, IO, conditions and mathematical formulas.

All real contracts include successful finite flat-memory execution, bounds on actual active accesses and preservation of other whole tagged cells. Zero unspecified values, output injectivity and necessary aliasing conditions remain explicit. Dimensions and strides stay symbolic. Compatible storage supports fp32/fp16/bf16; independent seed, value and output types are exposed where the source supports them.

FP guards inspect the actual reached finite operands. The four nonempty typed executions in `Proofs/TypedExecution.lean` use a named abstract model and do not claim GPU or IEEE validation. Experiment shapes select assumptions and do not restrict the symbolic dimensions of the proof.

`bench/tests/ChunkRetentionOpsReal.lean` checks both complete backward-state variants in finite memory, with three time chunks, 16-wide tiles, a partial final chunk, nonzero program IDs, nontrivial carry decay and adjacent tagged-memory preservation. It also checks the empty reverse loop. The other three real regression families are reused through the complete-source equalities.

Accepted in `../verification_round.md`. The 3,472-job integrated build, source gates, scope review and new finite-memory regression pass. Independent comparator/trust receipt `Logs/comparator-check-w_2hgm4d` accepts all 10 case/regression files, 90 named declarations and 485 kernel theorem targets, with 1,390 matching source/configuration fingerprints. Host composition, grid/allocation construction, backend tile legality, cross-program races and performance remain outside these per-JIT contracts.
