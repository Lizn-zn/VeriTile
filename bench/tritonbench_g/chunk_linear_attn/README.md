# chunk_linear_attn

Source: `chunk_linear_attn.py`, all four `@triton.jit` kernels.

The three public files share one implementation:

- `Kernels.lean` contains all original sources and their candidates, including every loop, branch, cast, mask and store. A candidate commutes one fp32 addition; forward output instead puts scale on the left of the final fp32 multiplication.
- `Correct.lean` gives eight real specifications: an original and candidate for each kernel. Each states an independent formula over initial arrays, successful finite flat-memory execution and preservation of every other tagged cell.
- `FPEquiv.lean` gives four complete typed-program equivalences. The actual assumptions are printed at the end of each entry: `add_commute` for both state kernels and backward gradients, `mul_commute` for forward output.

| Source kernel | Public specification prefix | Real output |
|---|---|---|
| `chunk_linear_attn_fwd_kernel_h` | `forwardState` | Optional seed plus all earlier Kᵀ·V chunks; optional final state includes all chunks |
| `chunk_linear_attn_fwd_kernel_o` | `forwardOutput` | Q·H plus causal Q·K·V, then scale, over every K block |
| `chunk_linear_attn_bwd_kernel_dh` | `backwardState` | Sum of scaled Qᵀ·dO over strictly later chunks |
| `chunk_linear_attn_bwd_kernel_dqkv` | `backwardGradients` | Full dQ/dK/dV contractions over every V block, with both causal masks |

Real names end in `_correct`, `DotLeft_correct` or `ScaleFirst_correct`; FP names end in `_equiv`. `Proofs/` groups source correspondence, execution and memory arguments by kernel, plus shared FP proofs and nonempty typed executions. The real and FP proof layers are separate.

Dimensions, strides, scale and program IDs stay symbolic. Forward state covers every seed/final flag combination. The reverse source uses `j = 0 .. NT-1` and `i_t = NT-1-j`, preserving the original descending sequence and its empty `NT=0` case. Backward gradients retain runtime `tl.num_programs(2)` in the dV slice and distinguish the combined `NT*K` history bound from the Q/K `K` bound.

Compatible dot inputs support fp32, fp16 and bf16. Forward state has independent seed/final types; forward output has independent V storage and output type; reverse state has independent Q and incoming-gradient storage plus output type. The backward-gradient source uses compatible common storage. Every `allow_tf32=False`, cast and unspecified boundary load remains in the shared implementation.

The real contracts explicitly choose a zero unspecified-value oracle; the source does not request zero padding. They require disjoint finite allocation intervals, bounds on actual active accesses, and injectivity of active output addresses. There is no `K=BK`, `BV≤V` or contiguous-stride restriction. State inputs reloaded after a history store must be separate from that history. Backward gradients require the four repeatedly loaded inputs to be separate from dV, and its three outputs to be distinct; Q/K may alias dV because they are loaded first. Forward output permits input/output aliasing because all loads precede its store. The entry files expose these conditions.

FP guards check finite operands at the actual reached rewrite sites: each state iteration's carry and dot result, the forward output's final total and scale, or the backward gradient's final dQ carry and dot result. Shape/distribution from the experiment selects the assumptions; it is not added as a proof-size restriction. These are conditional proofs in the admitted FP model, not IEEE bitwise guarantees or end-to-end GPU measurements.

The real regression in `bench/tests/ChunkLinearAttnReal.lean` checks full flat executions with partial tiles and nonzero program IDs. Forward state covers all optional branches; reverse state checks both its time tail and an empty reverse loop. The backward-gradient fixture uses two value iterations and 16-wide tiles, producing dQ=1900, dK=912 and dV=96 in both value blocks while preserving an adjacent integer-tagged cell. A separate abstract-model witness establishes a nonempty typed FP domain for each source; it is not a GPU experiment.

This is a per-JIT proof. Host allocation/grid construction, composition of the four kernels, backend tile legality, inter-program races and performance are outside the contract. Integration and independent replay status are tracked in `../verification_round.md`.

Accepted in `../verification_round.md`. The 3,469-job integrated build, all four real regressions, source gates and scope checks pass. Independent comparator/trust receipt `Logs/comparator-check-n641rovk` accepts all 19 case/regression files, 257 named declarations and 1,684 kernel theorem targets, with 1,362 matching source/configuration fingerprints.
