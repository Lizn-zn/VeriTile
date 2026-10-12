# attention_fwd_triton2

- Source file: `attention_fwd_triton2.py`
- Corpus: TritonBench-G v1
- Status: verified per-program real correctness and admitted FP equivalence.

## Public entries

`Kernels.lean` contains both complete inlined sources. Q/K/V use fp16 storage,
Q/K scales use fp32 and Out uses bf16, as in the supplied Python test/wrapper.
The explicit V-to-fp16 cast remains inside the second dot. That dot returns
fp16 before promotion into the fp32 accumulator; the final result is converted
to bf16 and stored with the original row/head mask. The candidate changes only
final division into reciprocal multiplication.

`Correct.lean` displays five input streams, the output window, masks and the
independent base-2 attention formula. Its two real specifications prove
successful flat-memory execution, active-address safety, the output values
and the frame outside the write-active window. The scope is
`N_CTX=BLOCK_N*numKVBlocks`, positive key-block size/count, shared batch/head
strides, contiguous head rows and unit inner stride. Active head lanes fit the
tile and row. Dimensions remain symbolic; host grid composition and concrete
IEEE arithmetic are outside the contract.

`FPEquiv.lean` compares the complete typed programs for arbitrary symbolic
source dimensions and strides, preserving full state and execution failures.
The sole numerical atom is `div_mul_rcp`. Its domain checks actual finite
fp32 accumulator lanes and finite/nonzero row denominators after the complete
loop. Neither input positivity nor an output equation is assumed. All casts,
exp2 calls, reductions, typed dots, masks and the final bf16 store are retained.
This is conditional equivalence under admitted atoms, not bitwise IEEE equality
or an end-to-end statistical error bound.

## Proofs and source correspondence

`Proofs/RealProjection.lean` checks equality of both complete precision-erased
sources with their corresponding `attention_forward_triton` sources. Only
this real projection erases the explicit V cast and output precision.
`Proofs/Real.lean` uses those equalities to transfer the previously checked
flat IO proofs, including every window, mask, independent formula and frame.
The public entry spells out its own complete contract, checked against that
helper by Lean. Historical rounding statements remain in
`Proofs/LegacyRounding.lean`.

`Proofs/FP.lean` uses `equiv_decompose` and `fp_prove` on this directory's own
sources. `TypedExecution.lean` supplies a model of the admitted scalar rules,
a satisfied normalization domain and successful nonempty executions of both
sources. Its opaque dot/reduction operations establish model consistency and
interpreter coverage; they are not another admission experiment or GPU emulator.
`bench/tests/FPAttentionFwdTriton2Source.lean` additionally checks the actual
nested V cast, fp16 dot result, fp32 accumulation, bf16 output cast/store, mask
and candidate preservation of the loop and store.

The helper JIT is inlined, and the Python's 128/96 head constants become
symbolic `BLOCK_DMODEL`/`HEAD_ACTIVE`. The output element-type alias is written
explicitly as `tl.bfloat16`; fp16 inputs and fp32 scale storage are explicit.
This reviewed transcription exception is registered in the corpus audit.
The outer-call lexical gate does not verify helper inlining or dtype inference.
The Python inner functions differ from `attention_forward_triton` only in the
V cast; their outer kernel ASTs match, while this wrapper allocates bf16 output.

## Partial KV blocks

The source masks padded K/V loads but includes padded scores in the denominator.
`Proofs/TailCounterexample.lean` checks successful execution of both complete
real sources at `N_CTX=1`, `BLOCK_N=2`, preserving the 128/96 head layout:
zero Q/K and unit V produce 1/2, while length-one standard attention is 1.
Both unrestricted tail specifications are therefore refuted. This small Lean
instance checks the mathematical source behavior, not GPU launchability.

## Validation

Both public entries and all helpers compile with `TritonBenchGVerified`.
Independent comparator replay and trust checks pass for the nine Lean files
in this directory and the source-precision regression; 27 named targets were
checked against the final source snapshot. The precision regression and
29 inventory/layout/shard tests pass. Coverage fingerprints, proof-gap entries
and specification sheets are current. The corpus source audit passes with
the documented transcription exception; it does not certify Python translation
or GPU execution.
