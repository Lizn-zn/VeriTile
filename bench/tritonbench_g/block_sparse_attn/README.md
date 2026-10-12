# block_sparse_attn

Source: `block_sparse_attn.py`, TritonBench-G v1,
`block_sparse_attention_kernel` at line 7.

The public real and FP contracts are connected. Final validation of the
migrated directory is in progress; completion is tracked in
[verification_round.md](../verification_round.md).

## Read the example

- [Kernels.lean](Kernels.lean) contains both complete typed Triton sources.
- [Correct.lean](Correct.lean) displays the input/output windows, valid CSR
  indices, independent attention formula and both real specifications.
- [FPEquiv.lean](FPEquiv.lean) displays the original/candidate programs,
  reachable-iteration domain and admitted FP equivalence.
- `Proofs/` contains execution, address, recurrence and composition lemmas.
  `Proofs/LegacySummary.lean` retains the old natural-metadata summaries and
  output-copy slices for reference; neither public proof imports it.

The candidate changes only `qk * softmax_scale` to `softmax_scale * qk`.
It preserves both score dots, the causal mask, `tl.exp`, online normalization,
probability/output casts, reductions, signed CSR visits and all stores.
Q/K/V/output storage is fp32, fp16 or bf16; scores and accumulators are fp32.
`NUM_D_BLOCKS >= 2` selects exactly one additional channel block, matching
Python. It does not iterate over an arbitrary number of channel blocks.

## Real specification

`block_sparse_attention_correctness` and
`block_sparse_scalar_left_correctness` prove complete per-program flat-memory
execution and output values after precision erasure. For each stored query
row and channel, the output is

`sum(exp(score) * initial_value) / sum(exp(score))`

over the CSR-selected key occurrences at or before that query row. The score
is the initial Q/K dot over one or two channel blocks, multiplied by the
scale. Repeated column entries retain their multiplicity. Row and column CSR
strides, and every Q/K/V/output batch/head/row stride, remain independent.
The formula uses initial arrays, not executed registers or an assumed output.
Empty/reversed CSR rows give zero under the framework's total real division.

The displayed IO and specifications require:

- `BN > 0`, and an output row stride large enough for the active channels.
- Initial Q/K/V values pinned over their mathematical context; both CSR row
  endpoints and visited column cells pinned to independent signed arrays.
- Nonnegative complete column slots and selected key-block indices. Signed
  visits may themselves be negative when their layout offset makes the full
  slot valid. A first visible key is not required.
- Allocations for active loads and row-masked stores. Unchecked Q/K/V branches
  need padding; checked out-of-context lanes need no allocation. Second-channel
  bounds apply only when the corresponding source branch is enabled.
- Disjoint flat allocations and zero undefined-load values, as required by the
  shared flat-memory bridge. The underlying logical-region formula proof
  itself allows arbitrary undefined values on masked lanes.

The postcondition includes successful execution, real-tagged output cells and
preservation of every other flat cell. The output mask remains active even
when `EVEN_M` is set. No whole-loop execution, precomputed accumulator, trace
safety or computed-output equation is a caller premise.

## FP specification

`block_sparse_scaling_equiv` proves equivalence of the complete original and
candidate. At every reached signed-loop iteration, its domain checks the
actual fp32 score tile and scale for finiteness after the score dots. Prefix
execution and earlier visits remain in this domain's context.

`#print_fp_assumptions block_sparse_scaling_equiv` lists only `mul_commute`.
No exponential identity, online-update atom or reduction reassociation is
assumed. `tl.exp` stays unchanged. Experimental shape and input distribution
selected the atom; they do not fix this derivation's dimensions.

These are real correctness and equivalence under admitted FP assumptions,
respectively. Neither is a concrete IEEE equality, end-to-end error bound,
GPU measurement or launch-wide scheduling claim. Integer addresses use
mathematical signed arithmetic rather than fixed-width overflow.

## Proof and regression evidence

The proof follows actual source segments: address/query/CSR prologue, signed
column read and K score dots, normalization, V loads and value dots, signed
loop induction, and final masked stores. `RealInputs` connects initial arrays;
`RealFormula` establishes the independent attention formula; `RealWindows`
derives complete access safety; `RealFlat` uses the shared flattening bridge;
`RealIO` and `RealContract` connect those results to the public contracts.

Regressions in `../../tests/BlockSparse*.lean` cover distinct CSR/data strides,
negative visits, repeated blocks, empty/reversed intervals, partial query/key
tiles, both channel branches, insufficient allocation and tagged memory frames.
The independent-array fixture obtains 32/5 and 64/5 from five visible key
occurrences. `BlockSparseFlat.lean` calls the public specifications for every
storage/source/mask/channel choice, supplies real input/allocation witnesses,
and checks output at flat addresses 80/81 and untouched natural-number padding
at 83. `FPBlockSparseSource.lean` checks casts, intrinsics and reachable guards;
`Proofs/TypedExecution.lean` supplies complete abstract FP executions.

A source-address regression also covers the signed static-offset correction:
`cols + layout * stride + visit` adds signed terms before conversion to a
natural address, so layout 1, stride 13 and visit -2 read slot 11. Dynamic
pointer arithmetic retains its existing semantics and is connected to arrays
under the stated nonnegative key-start condition.

Final acceptance (2026-10-09): the verified benchmark library builds, all 13
case regressions pass, and the source gate, manifest, coverage fingerprints
and layout checks pass. Independent comparator/trust replay accepts all 46
case files and 13 regression files in `Logs/comparator-check-roalqlu7`.
The exact inventory contains 480 named declarations and 2,709 kernel theorem
targets; all 953 snapshot input hashes match at acceptance. This receipt
supersedes the affected component receipts before the public-file migration.
The history and reproducible checks are recorded in
[verification_round.md](../verification_round.md).
