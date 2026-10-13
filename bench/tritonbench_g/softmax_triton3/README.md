# softmax_triton3

`Kernels.lean` contains both complete typed sources. Input and output share a
storage dtype; the optional additive-mask storage is independent. All nine
fp32/fp16/bf16 storage pairs and both mask branches are covered. Explicit fp32
load conversions, `tl.exp`, reduction order, negative-infinity row padding,
zero additive-mask padding and the masked output store are retained. The
candidate replaces division by multiplication with the scalar reciprocal.

`Correct.lean` gives real formulas for both source versions: `exp(x_i)` divided
by the active exponential sum, or `exp(x_i + mask_i)` divided by its sum. Each
contract includes termination, active-address safety and a full memory frame
for the cast-erased real projection. Block size is positive; dimensions,
strides and row IDs are symbolic. The formula covers the complete row when
`n_cols ≤ BLOCK_SIZE`. Original public theorem signatures are preserved.

`FPEquiv.lean` compares complete typed executions, retaining all casts, memory
cells, registers and failure behavior. Its domain checks actual computed
numerators for finiteness and the actual computed row sum for finiteness and
nonzero after the optional mask prefix. It uses only admitted `div_mul_rcp`.
`Proofs/FP.lean` lifts that existing atom through the explicit fp32 assignment;
it adds no numerical axiom.

Translation-surface exception: the DSL currently loses the output storage dtype
through `output_row_start_ptr` / `output_ptrs` aliases. The typed transcription
therefore spells Triton's implicit output conversion explicitly for fp16 and
bf16. The fp32 store remains unchanged, with no redundant conversion. This is
an interface-lowering repair, not a numerical optimization or source algorithm
change. `Proofs/Source.lean` checks the exact final store carrier and expression
for all nine storage pairs, both versions and both mask branches, plus the
unchanged normalization prefix and fp32 intrinsic exponentiation. The legacy
untyped source is retained solely for its existing theorem contracts.

Host scheduling, the alternate large-row launcher call and fixed-width address
overflow are outside these per-program contracts. No performance claim is made.
Independent comparator replay remains pending on this host.
