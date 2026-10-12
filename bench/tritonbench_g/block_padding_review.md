# Checked block-pointer padding review

## Source/model mismatch (2026-10-09)

The experiment environment pins Triton 3.5.1. Its [load definition and
parameter documentation](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/core.py#L1899-L1965)
specify an empty default `padding_option`; out-of-bounds values are unspecified.
Only explicit `padding_option="zero"` supplies zero. `boundary_check` prevents
an out-of-bounds memory read; it does not by itself choose a padding value.

At discovery, the DSL parsed explicit zero padding but dropped the choice,
and all three interpreters supplied zero for checked-off block lanes. The core
and DSL now preserve the distinction; the source-proof migration below is in
progress. Identical-looking Python and Lean source alone did not expose this
gap: independent Lean replay established the old model's theorems, not faithful
padding semantics.

The review reopened three previously accepted cases: `chunk_bwd_dqkg`,
`chunk_cumsum_kernel`, and `chunk_cumsum_vector`. Their real IO already contains
a zero-unspecified-value contract; the repair must connect actual omitted
block padding to that contract and recheck the FP semantics, source bridge and
regressions. This is not a refutation of those conditional specifications.
All three have now passed their repairs and independent replay below. Scalar
and vector cumsum outputs are independent of arbitrary real padding; dqkg's
zero-oracle premise remains necessary, as shown by a full-source counterexample.
Current acceptance is **28 complete / 156 pending**. The only user-deferred
case remains `attention_kernel_aligned`. The previously pending
`chunk_delta_fwd` is now accepted after its complete source repair and replay.

## Reproducible source inventory

Run `python3 bench/audit_block_padding.py` for nonzero/unspecified sites, or add
`--json` for every checked JIT load. This reads Python ASTs without importing
Torch or Triton. Literal empty boundary checks, stores and host calls are
excluded. Dynamic boundary checks remain visible. It does not determine
whether an out-of-bounds lane is reached or affects an observable output.

The current corpus has 138 checked JIT loads: 5 explicitly zero-padded and
133 with unspecified padding in 18 source files. All sites below omit the
padding option. Counts include every JIT helper, not just the selected port.

| Directory | Sites | Python source lines |
|---|---:|---|
| `chunk_bwd_dqkg` | 7 | 46, 62, 63, 65, 66, 75, 76 |
| `chunk_cumsum_kernel` | 1 | 28 |
| `chunk_cumsum_vector` | 1 | 41 |
| `chunk_delta_fwd` | 4 | 52, 63, 64, 65 |
| `chunk_gated_attention` | 8 | 39, 77, 85, 87, 92, 96, 102, 106 |
| `chunk_gla_fwd` | 14 | 49, 50, 52, 53, 101, 102, 155, 156, 192, 236, 238, 240, 247, 248 |
| `chunk_gla_simple` | 5 | 48, 50, 52, 57, 63 |
| `chunk_linear_attn` | 15 | 23, 31, 32, 61, 62, 63, 71, 95, 97, 117, 118, 132, 133, 134, 135 |
| `chunk_retention` | 16 | 53, 62, 64, 116, 118, 120, 126, 170, 171, 172, 224, 225, 238, 239, 241, 243 |
| `chunk_retention_ops` | 15 | 31, 37, 38, 76, 77, 78, 83, 115, 117, 147, 148, 159, 160, 161, 162 |
| `int_scaled_matmul` | 2 | 82, 83 |
| `parallel_attention` | 17 | 41, 50, 53, 76, 78, 131, 132, 144, 146, 169, 171, 214, 214, 225, 226, 248, 249 |
| `parallel_retention_attention` | 16 | 59, 67, 69, 94, 96, 124, 138, 140, 158, 160, 194, 194, 206, 207, 226, 227 |
| `pow_scalar_tensor` | 2 | 153, 172 |
| `relu_strided_buffer` | 2 | 145, 164 |
| `reversed_cumsum` | 1 | 42 |
| `reversed_cumsum_scalar` | 1 | 28 |
| `triton_attention` | 6 | 63, 82, 211, 212, 217, 226 |

## Required correction

1. Retain a per-load distinction between zero and unspecified padding in the
   AST, precision erasure, relocation and all real/FP interpreters. Stores must
   retain their existing checked-write behavior.
2. Omitted/empty padding in the DSL must select unspecified values. Explicit
   zero padding must remain independent of the unspecified-value oracle.
   Do not silently add zero padding to the upstream Python or label a repaired
   implementation as the original.
3. For the three reopened contracts, propagate their existing zero-oracle
   premise into block-load and loop proofs. For `chunk_delta_fwd`, make any
   such premise visible in the final IO contract. At discovery its unrestricted
   execution theorem described the zero-padding model; the repaired contract
   now states the premise explicitly.
4. Check nonzero unspecified values and tails, typed precision preservation,
   no boundary checks, explicit zero padding, and memory safety/frame. Replay
   affected real/FP proofs and current source snapshots before reinstating
   acceptance. A source inventory alone cannot reinstate a case.

The zero-oracle premise is a conditional model contract, not a guarantee from
the original Triton source. After correcting the translation, check whether
unspecified lanes can affect each observable output. If they cannot, prove
that independence; if they can, retain a source-level counterexample to the
unconditional mathematical formula. Do not turn a source defect into a claim
of unconditional correctness by silently choosing zero for unspecified data.


## Core foundation checkpoint

The core now carries a per-load padding option through `Op.load`,
`ComputeOp.load`, precision erasure and address relocation. Real, rounded-real
and structural FP evaluation distinguish zero padding from the unspecified
value oracle. The zero-oracle load equivalence is proved in
`VeriTile/Triton/Semantics/BlockPadding.lean`; flat-memory transport uses the
same zero-oracle premise it required before this change. Locality needs only
agreement of the existing oracle, not that it is zero.

`bench/tests/BlockPadding.lean` checks a nonzero oracle, explicit zero,
unchecked loads, typed FP execution, fp32 compute-load interpretation and both
projections. These low-level tests pass, as do the existing block-pointer
regressions. The raw AST constructor retains its zero default for compatibility.
At this checkpoint the DSL still emitted that default. Its compatibility
checks therefore did not restore source-level acceptance for the three cases.

The foundation passes the joint `TritonBenchGInProgress` and
`TritonBenchSpecExamples` build (3,650 targets), both block-padding regression
files and 17 Python source-audit tests. Independent comparator/trust replay
accepted 18 files, 544 named proof targets and 6,045 kernel theorem targets;
all 1,153 snapshot inputs matched (`Logs/comparator-check-_a0onijy`). Coverage
metadata retains the source-padding qualification for the three reopened
cases. This receipt checks the foundation and compatibility with the old DSL
translation, not the subsequent DSL migration.

## DSL and vector-cumsum migration

Omitted and empty `padding_option` now select `.unspecified` for block loads;
explicit `"zero"` selects `.zero`. The frontend preserves this choice in both
algorithm and fp32 compute loads. Ordinary pointer loads retain their existing
behavior. Unsupported `"nan"`, duplicate options, nonempty ordinary-pointer
padding, and padding on stores/atomics are rejected rather than discarded.
`bench/tests/BlockPaddingDSL.lean` checks source lowering and actual real
execution, including a nonzero unspecified tail and an unchecked memory read.

`chunk_cumsum_vector` now connects the full typed source to the new AST. The
real proof threads its existing zero-oracle premise through loads, the loop
invariant and safety. `Masked3DTileKernelIO₁.Implements.intro_undef` exposes that
same premise to both obligations without changing the public IO relation. The
FP proof retains arbitrary unspecified values and still uses only
`add_commute` under its actual-operand guards. The typed execution witness and
both public entry points build. Removing the real zero-oracle premise remains
unproved, so this batch does not yet reinstate acceptance.

The targeted build, including `TritonBenchSpecExamples`, passes 3,593 targets.
Block-padding DSL, vector source, FP block-pointer and Triton smoke regressions
pass. Independent comparator/trust replay accepts all 18 files, 85 named
proof targets and 930 kernel theorem targets. All 1,155 current snapshot
inputs match (`Logs/comparator-check-t6_m7rbs`).

The combined in-progress build at that snapshot failed at the old source/AST bridges
in `chunk_bwd_dqkg/Proofs/LegacyReal.lean`,
`chunk_cumsum_kernel/Proofs/LegacyReal.lean`, and
`chunk_delta_fwd/ChunkDeltaFwd.lean`. Their handwritten loads still select zero.
The scalar and dqkg bridges have since been repaired as described below;
chunk delta remains open. No whole-corpus build success is claimed. The raw AST compatibility
default must not be used to conceal these failures.

## Scalar-cumsum independence from padding

`chunk_cumsum_kernel` now models the actual loaded input: valid lanes read
memory and checked-off lanes use the initial state's unspecified-value oracle.
Its loop invariant preserves that oracle and sums all loaded values into the
carry. The proof no longer claims that the final carry excludes padding.
Instead, every valid output prefix is proved to contain only valid input
positions, so it is independent of arbitrary real padding.

`RealPadding.source_outputs` connects this result to both complete typed
sources after precision erasure, for symbolic T and positive BT. The existing
flat IO specifications keep the shared adapter's zero-oracle condition,
active-address safety and full tagged-cell frame. The complete FP equivalence
retains unspecified values and still uses only guarded `add_commute` on the
actual carry and block total; no scan identity or reassociation is introduced.

The correct/equivalence/typed-execution build passes 3,452 targets. The source
AST regression and a complete three-chunk, partial-tail execution regression
pass; the latter covers arbitrary padding, including seven, in both versions.
Independent comparator/trust replay accepts 13 files, 93 named proof targets and 668 kernel theorem targets;
all 1,157 current snapshot inputs match (`Logs/comparator-check-xj59vm4q`). This reinstates scalar
cumsum acceptance within its stated scope. Vector cumsum and dqkg remain under
review, and chunk delta was already pending.

## Vector-cumsum independence from both padding axes

The complete vector proof now carries the actual loaded values in every
column, including unspecified time and feature lanes. Its invariant preserves
the initial oracle and derives every valid prefix from independent initial
input cells. The lower-triangular real dot only uses earlier rows in the same
column, so neither padding axis can affect a valid output. The final carry is
allowed to include tail values.

`RealPadding.source_outputs` proves successful execution, all valid prefixes
and the complete tagged-cell frame for both typed sources after erasure,
without a zero-oracle premise. Symbolic shapes, strides, all time chunks and
both partial tails remain covered. The existing flat IO specifications retain
the shared adapter's zero-oracle condition. Their proof now uses the ordinary
assembly lemma because region execution and safety no longer need that
condition. FP equivalence is unchanged and uses only `add_commute`.

The targeted build passes 3,458 targets. Three regression files pass, including
both full real executions at T=5/S=3/BT=2/BS=2 with arbitrary padding (also seven),
nonzero feature/head IDs, all active values and neighboring untouched cells.
Independent comparator/trust replay accepts 20 files, 96 named proof targets and 927 kernel theorem targets;
all 1,159 snapshot inputs match the checked worktree (`Logs/comparator-check-gcqf5awz`). Coverage review
and the proof-gap manifest pass. Vector cumsum is reinstated; dqkg and chunk
delta remain open, so no whole-corpus build success is claimed.

## Dqkg: conditional correctness and observable padding

The seven handwritten real block loads now retain `.unspecified`, reconnecting
the complete typed source to the real proof. The existing IO zero-oracle
condition is explicitly threaded through all value iterations and the postlude;
setup needs it only for the gate region, and the postlude only for q/k. Memory
bounds, tagged-cell frame, all storage combinations and the two independent
whole-V formulas are preserved. FP equivalence still uses only `add_commute`.

Unlike the two cumulative sums, a padded V lane is contracted by the dq dot.
The complete-source witness uses T=K=BT=BK=BV=16, V=1, scale=1, zero program
IDs and all-zero actual memory. Its oracle is one for V/H/DO/DH and zero for
other regions. `UndefinedTail.source_stores_fifteen` proves actual execution
through the final stores with dq[0,0]=15. The independent initial-array formula
is zero; `refutation dq_without_zero_padding` rejects the unconditional claim.
The proof's synthetic array represents these loaded tiles only; the executed
source still has V=1. No Python padding was silently repaired, and no GPU or
compiler-specific behavior is inferred from this real-model witness.

The final targeted build passes 3,469 jobs, including real correctness, FP
equivalence, typed execution, the host-bounds refutation and the padding
counterexample. Both source/precision and active-address regressions pass.
Coverage review (380 declarations / 201 files) and the proof-gap manifest pass.
Independent comparator/trust replay accepts **24 files**, **150 named
proof targets** and **1,124 kernel theorem targets**; all **1,161 snapshot
input hashes** match the workspace. Receipt: `Logs/comparator-check-hrl4w00r`.

## Chunk delta: complete IO and observable time padding

The four omitted-padding loads remain unspecified. Both public real contracts
now cover every inner/outer iteration, optional seed/final storage, all three
outputs, active memory safety and complete flat-memory frame, under an explicit
zero-oracle premise. They preserve the actual ceil(BT/BC)*BC window width,
including overlap. The full typed FP proof still uses only add_commute.

`UndefinedTail.source_stores_fifteen` executes the original with T=1,
K=V=BT=BC=BK=BV=16, NT=1, no seed, final storage enabled and all-zero input
memory. K/V time-tail values are one; valid final H[0,0] becomes 15 instead of
the independent formula's zero. The matching declaration-level refutation
rejects dropping the padding premise. The synthetic wide array only describes
loaded tiles; source dimensions and store masks remain unchanged.

Independent comparator/trust replay accepts **31 files**, **179 named proof
targets** and **1,271 kernel theorem targets**. All **1,171 snapshot input
hashes** match the current workspace. Receipt: `Logs/comparator-check-4rbqaiz0`.
The targeted build (3,474 jobs), three Lean regressions, 18 source-checker
tests, source gates, coverage review and manifest pass.
