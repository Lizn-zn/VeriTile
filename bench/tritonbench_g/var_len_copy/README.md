# var_len_copy

Source: `var_len_copy.py` (TritonBench-G v1).

The full-loop real and FP contracts are in `Correct.lean` and `FPEquiv.lean`.
The historical contracts remain unchanged, with helpers in `Proofs/Legacy.lean`:
they certify a fixed chunk and the positive single-chunk full-kernel case.
The new contracts cover every signed segment length within the domains below.

The new source preserves the Python mask `offset < length` in every iteration.
Consequently, with positive block size B, a nonpositive length copies nothing;
a positive length at most B copies that length; a length greater than B copies
`ceil(length / B) * B` elements. In particular, length 3 and block size 2 copy
four elements. Bounds and frames refer to this actual span.

`Kernels.lean` contains 36 independent floating input/output storage pairs:
fp16, bf16, fp32, fp64, float8e4nv and float8e5. Source loads retain their
storage type. When input and output differ, an explicit `.to(output)` represents
the source store's implicit destination conversion; it is not a pre-load cast.
The candidate commutes only the signed `old_start + i` address sum.
The floating real contracts concern storage-erased mathematics; typed FP
equivalence retains destination conversion and does not assert that narrowing
preserves the original real value. Nine additional homogeneous profiles cover
int8/int16/int32/int64, uint8/uint16/uint32/uint64, and bool/int1. Their real
contracts use exact typed Int, Nat, or Bool readback and full raw-cell frames.
Discrete widths share these carriers; homogeneous copies perform no payload
narrowing, and this is not a fixed-width arithmetic model.

Metadata is loaded once into signed registers before the loop. The int32 and
int64 labels both use the existing mathematical Int channel: machine-width
wraparound is not modeled. Signed additions precede the pointer's Int-to-Nat
conversion. Real snapshot contracts currently require nonnegative initial
source/destination starts and separate source/destination regions. Metadata
may alias the destination because its initial loaded values remain in registers.
Mixed floating/discrete pairs and conversions between distinct discrete
widths or signedness remain uncertified. Concrete shared-DSL blockers are
recorded in [conversion_gaps.md](conversion_gaps.md). This is not a claim of
exhaustive low-level dtype coverage.

Region proofs preserve arbitrary undefined masked-load values. The flat-memory
bridge has an explicit zero-undef domain. Host scheduling, cross-program races,
and fixed-width overflow remain outside these per-program contracts.

`Proofs/Execution.lean` checks successful complete execution in a nonempty
exact-real typed backend: fp64 input and bf16 output, length 3 and block size 2,
destination values 11, 12, 13, 14, followed by an unchanged sentinel 7. Both
versions also preserve all memory for every nonpositive signed length. This
witness checks actual typed storage and loop execution, not IEEE rounding.
`Proofs/DiscreteExecution.lean` additionally checks complete typed execution
for all nine homogeneous discrete profiles and both versions: signed negative
values, unsigned values, and alternating booleans retain the fourth overcopied
cell and preserve the following sentinel.

All local source, real, FP, execution, and public-entry compiler checks passed,
with standard-axiom module audits and no additional FP assumptions. The
historical 24 declaration headers and 16 docstrings are preserved verbatim.
The dependency-aware public build passes all 3468 jobs. All ten public
contracts have only standard axioms and are registered with validated
source/support hashes as `test-gap`; independent comparator replay remains pending.

`Proofs/SourceChecks.lean` pins the exact destination store and conversion AST
for all 36 floating pairs and both versions, including the fp64 precision
boundary, destination storage tag, signed address, and unchanged lane-only mask.

The six new real specifications use `Spec.Real` over the named
`floatingFlatCopyContract` and `discreteFlatCopyContract` conclusions. Both
include successful full execution, the actual-span frame, and initial-source
readback. Their bounds and metadata hypotheses remain explicit in each theorem.
