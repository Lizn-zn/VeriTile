# embedding_triton_kernel

`Kernels.lean` contains the complete signed-int32-ID `embedding_kernel` and
its integer-index commuted candidate. Weight/output precision matches at
fp16, bf16 or fp32. The whole strided loop, vocabulary checks, row/column
masks, token default `vob_end_id` and zero weight default are retained.

`Correct.lean` displays two real specifications: each in-range signed token
reads its row of the initial weight matrix; out-of-range tokens output zero.
They prove successful finite flat execution, every active output and the
complete tagged-cell frame. Chunk sizes are arbitrary, including multiple
rows per iteration and a last chunk extending past `BLOCK_N`. Only the actual
`n_ctx` and hidden-column masks suppress stores. Output stride need cover
`min(BLOCK_DMODEL, hidden_size)`, so padded columns may exceed the stride.

The two input regions must differ from output, ensuring later iterations
still read initial inputs. Actual address bounds, finite allocation/coverage
and zero unspecified values for the flat adapter are explicit. Zero chunk
stride follows formal range semantics; it is not a Triton compilation claim.

`FPEquiv.lean` proves complete typed equivalence by commuting only the natural
index product `program_id * BLOCK_N`. It preserves states and failure without
floating-point atoms or numerical guards; assumption output is `none`.

`bench/tests/EmbeddingTritonExecution.lean` instantiates the public contract
with two-row chunks, `BLOCK_N=3`, padded width four and physical stride two.
It covers negative invalid and valid IDs, the overshooting final row and an
integer-tagged untouched tail. An all-masked run uses empty allocations.
`bench/tests/FPEmbeddingTritonSource.lean` checks actual signed comparisons,
subtraction, storage precision, load defaults and masks. The inhabited typed
model in `Proofs/TypedExecution.lean` runs three complete iterations from
empty registers for all storage choices and both candidates, including a
masked row and column. This does not certify physical GPU rounding.

Scope is one program, excluding host launch correctness, concurrent races
and fixed-width index overflow.

Validation passed: complete repository closure, public execution witnesses,
typed model execution, signed-operation/precision/mask checks, full source
correspondence, corpus source gates, 32 audit/layout tests, coverage/manifest
checks and independent comparator/trust replay of 16 files. Current source
fingerprints and theorem inventories match `Logs/comparator-check-ziggvwz7`.
