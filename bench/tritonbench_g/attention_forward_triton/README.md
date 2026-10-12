# attention_forward_triton

The public entries share the complete source from `attention_forward_triton.py`:

- [Kernels.lean](Kernels.lean): original and reciprocal candidate, including the
  inlined `_attn_fwd_inner` loop, fp16 Q/K/V/Out, fp32 scales and accumulation,
  both typed dots, `tl.math.exp2`, casts, masks and final store.
- [Correct.lean](Correct.lean): both real projections implement the independent
  base-2 attention weighted sum. The five input windows and output window are
  visible, with termination, active-address safety and the memory frame.
- [FPEquiv.lean](FPEquiv.lean): the complete typed programs are equivalent under
  `div_mul_rcp`. Guards inspect the actual finite fp32 accumulators and finite,
  nonzero row denominators after the full loop. The printer lists only that atom.

Real scope is `N_CTX = BLOCK_N * numKVBlocks`, positive key-block size/count,
shared batch/head strides, contiguous head rows and unit inner stride. Active
head lanes fit the block and row widths; dimensions remain symbolic. The real
IO contract fixes undefined masked-load values to zero. FP
contextual equivalence retains the general source parameters. Both contracts
are per-program; neither proves host-grid composition, bitwise IEEE equality
or a GPU performance claim.

The real proof combines initialization, the online-softmax loop invariant,
normalization and the final masked store. `Proofs/RealExecution.lean` connects
this execution to the mathematical formula and memory frame;
`Proofs/RealSafety.lean` checks all active accesses; `Proofs/Real.lean` assembles
the flat-memory IO contract. `Proofs/RealNormalization.lean` transfers execution
and safety to the candidate without relying on an FP rule.

`Proofs/TypedExecution.lean` supplies an abstract model of the selected scalar
rules, successful executions of both typed kernels on a nonempty 16-row/16-key
instance, and an inhabited normalization domain. These are Lean execution
witnesses, not numerical admissions or GPU experiments.

The source does not compute standard attention for a partially filled KV
block. [Proofs/TailCounterexample.lean](Proofs/TailCounterexample.lean) proves
that with one key in a two-key block, zero Q/K and unit V, both complete real
kernels store `1/2` while standard length-one attention is `1`. Masked padded
K/V loads still contribute scores to the denominator. The witness retains
128 head lanes and 96 active lanes; its small symbolic block size is not a
claim about GPU launch legality.

The historical untyped transcription and rounding-model statements are kept
in `Proofs/LegacyRounding.lean`. Its identity-fp16-rounding premise is not a
current FP-equivalence result. Shared mathematical and pointer-layout helpers
are reused by the new real proof; the current kernels are in `Kernels.lean`.

The mechanical source-sequence scan cannot align the inlined helper with the
Python outer call. Its registered transcription exception records this exact
mapping: the helper call and return become the displayed complete loop;
Python's 128/96 head constants become symbolic parameters; the Python test's
storage dtypes are explicit; the wrapper's `Out.type.element_ty` becomes
`tl.float16`. `Proofs/RealProjection.lean` and `Proofs/RealLoop.lean` check source
connections, and `bench/tests/FPAttentionSource.lean` checks the final typed
cast, store and mask. These checks supplement source review; they do not
independently certify translation from Python or execution on hardware.

Validation and the corpus-wide progress count are recorded in
[verification_round.md](../verification_round.md).
