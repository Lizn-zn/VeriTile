# fast_ce_loss

The three complete JITs from `fast_ce_loss.py` share `Kernels.lean`:
ordinary forward, chunked forward, and in-place backward. Each has an original
and a candidate that commutes only the integer row-index/row-stride product.
`Source/Stage.lean` resolves constexpr flags before dtype inference. Runtime
signed-label and chunk branches remain in the kernels. Storage is fp16, bf16
or fp32; actual cast positions, libdevice tanh, masks and stores are retained.

`Correct.lean` exposes six independent mathematical specifications after
precision erasure. They prove successful finite flat execution, every output
and preservation of every other tagged memory cell for symbolic dimensions,
strides and program indices. Actual access bounds, finite allocation/coverage,
typed valid signed labels and zero unspecified values are explicit. Forward
requires positive enabled softcap/scaling parameters for its negative-infinity
padding. Backward permits arbitrary scaling and any nonzero enabled softcap. Forward also requires `0 < V <= B`; chunked
requires `V > 0` and `B > 0`; both distinguish loss and LSE regions. Ignored
backward labels need no upstream allocation. Backward uses the caller-supplied
LSE and upstream; it does not prove a host composition or a calculus theorem.

Two source behaviors are part of the specification:

- Softcapping transforms masked negative infinity to `-SOFTCAP`, so padded lanes
  contribute `exp(-SOFTCAP)` to LSE. With one zero logit, block size two and
  softcap one, the source loss is `log(1 + exp(-1)) > 0`, not zero.
- Chunked forward places both stores inside `chunk_idx == 0`. Other chunks
  execute and preserve all memory. No theorem validates the host aggregation
  over their unwritten LSE cells.

`Proofs/Counterexamples.lean` verifies both counterexamples from the full source.
`FPEquiv.lean` exposes three whole typed-state/failure equivalences with an empty
domain and **no numerical FP atoms**. Structural equivalence preserves these
source behaviors. It does not assert physical IEEE equality, host launch/race
correctness or freedom from fixed-width index overflow.

Validation passed: the final repository closure, 18 case/frontend regressions,
all 55 previously accepted cases' 110 public entries, full source comparisons,
source/layout gates, 44 configured audit tests and coverage/manifest checks.
Finite-memory witnesses and 120 original typed executions with their candidate
counterparts pass. Fresh independent comparator/trust replay accepts all 163
selected source, proof, test, shared-semantic and affected-entry files, with
current fingerprints and complete theorem inventories. Receipt: `Logs/comparator-check-gqnownvb`.
The example's source/contract/equivalence review is **complete**; the source
behaviors and excluded host/physical-numeric claims above remain explicit.
