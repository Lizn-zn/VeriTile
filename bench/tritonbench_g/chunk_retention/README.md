# chunk_retention

The four JIT sources in `chunk_retention.py` share complete typed original and candidate implementations in `Kernels.lean`. `Correct.lean` states eight real specifications; `FPEquiv.lean` states four conditional FP equivalences. Execution, loop, memory and arithmetic proofs are in `Proofs/`.

| Source | Real contract | Candidate and FP atom |
|---|---|---|
| `chunk_retention_fwd_kernel_h` | Seeded decayed recurrence, pre-update history, optional final state; includes the last partial chunk's shorter decay length | Commute decayed carry and weighted dot; `add_commute` |
| `chunk_retention_fwd_kernel_o` | Row-decayed Q·H plus causal, distance-decayed Q·K·V; sums all K blocks, then scales | Move the final scale factor to the left; `mul_commute` |
| `chunk_retention_bwd_kernel_dh` | Actual untransposed dO·V contraction over reverse chunks, multiplied by full-chunk decay and written once | Commute carry and weighted dot; `add_commute` |
| `chunk_retention_bwd_kernel_dqkv` | All V blocks and all three dQ/dK/dV outputs, with their query/key/causal decay weights | Commute the final weighted dQ carry and score-gradient dot; `add_commute` |

The head rate is `1 - 2^(-5 - head % H)`. Integer row offsets remain signed, including negative exponents in unused tail lanes. The explicit `arange(...).to(tl.int32)` states the signed index type; implicit float promotion is handled by the DSL. Intrinsic `tl.math.exp2` and `tl.math.log2`, casts, TF32-disabled dots, branches, pointer order and stores are retained. The reverse loop uses an ascending counter with index `NT-1-j`.

The supplied backward-state source requires square `BT` tiles. It does not read `q` or `scale`, loads an unused history tile, does not transpose V, and applies the chunk factor only after the loop. Its sole store uses base `head*historyHead + keyBlock*K*V`, row offset `valueBlock*BT`, and column offset zero. These behaviors are preserved in the formula and access bounds. Its public real contract requires `NT>0` because the Python source uses the loop index after the loop.

Each real contract includes successful execution in finite flat allocations, bounds on actual active accesses, active-output injectivity and preservation of all other tagged cells. Source block padding is unspecified; the zero-extended formulas explicitly require a zero unspecified-value oracle. Required aliasing conditions are visible in the public entries. Dimensions and strides remain symbolic. Compatible dot storage supports fp32/fp16/bf16, with the independent storage and output types exposed by each source.

FP requirements check the actual reached operands of the admitted local rewrite. They retain dtype, casts and dot options, and do not use precision erasure. `Proofs/TypedExecution.lean` supplies complete original/candidate executions with nonempty domains under a named abstract arithmetic model; these witnesses are not GPU measurements. Experiment shapes select atomic assumptions and do not become symbolic proof dimensions.

`bench/tests/ChunkRetentionReal.lean` exercises all four full real contracts with 16-wide tiles, nonzero program IDs and partial time/key/value tails. It checks multiple K/V/time iterations, all forward seed/final flags, nontrivial head decay, all gradient outputs and adjacent tagged-cell preservation.

Accepted in `../verification_round.md`. The 3,470-job integrated build, four real regressions, source gates and coverage review pass. Independent comparator/trust receipt `Logs/comparator-check-v3__2clu` accepts all 20 case/regression files, 299 named declarations and 1,759 kernel theorem targets, with 1,381 matching source/configuration fingerprints. Host composition, allocation/grid construction, backend tile legality, inter-program races and performance remain outside these per-JIT contracts.
