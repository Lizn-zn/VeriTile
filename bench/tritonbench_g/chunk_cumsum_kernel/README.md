# chunk_cumsum_kernel

**Padding repair:** the source and proof AST now preserve unspecified tail
loads. Both complete real source programs compute valid prefixes for arbitrary
unspecified values; the FP proof still uses only `add_commute`. Compilation,
source/padding regressions and independent replay pass, reinstating this case
for the stated scope. See [the shared padding review](../block_padding_review.md).

Source: `chunk_cumsum_kernel.py`, `chunk_global_cumsum_scalar_kernel`.
Both real specifications and the complete FP equivalence were checked for
the scope below; compilation, regressions and independent replay pass.

- `Kernels.lean`: complete typed original and sum-left candidate. The only
  candidate change is `b_z = b_zz + b_z`; fp32 load conversion, scan, sum,
  boundary masks, all iterations and explicit output storage cast are retained.
- `Correct.lean`: both versions implement an independent global prefix sum
  over the initial input cells after precision erasure. T and positive BT are
  symbolic, including multiple chunks and a partial tail. The IO contract
  requires distinct input/output regions, disjoint allocations, sufficient
  active-address bounds and zero unspecified load values in the shared flat IO
  adapter. It proves successful flat execution, every active output and the
  complete tagged-cell frame. The additional `RealPadding.source_outputs`
  theorem proves valid region outputs for arbitrary unspecified real values.
- `FPEquiv.lean`: both full typed programs are equivalent under `add_commute`
  at each reached fp32 carry update. The domain checks the actual carry and
  computed block total after that iteration's load, scan and reduction.
- `Proofs/`: real projection, loop/context rewrite, independent formula/IO,
  FP composition and a complete abstract execution witness. `LegacyReal.lean`
  contains the reused real scan and full-loop proofs; its identity-rounding
  specialization is used internally, not as a numerical FP correctness claim.

The actual carry includes every loaded lane. In the partial last chunk its
final value can therefore include unspecified padding. A valid output prefix
only uses earlier valid input positions, so those final carry terms cannot
change any valid output. This independence is proved for arbitrary T and
positive BT, without fixing the padding to zero.

The FP scan is an opaque backend operation retaining kind, axis, direction
and computation precision. No scan identity, decomposition, reduction
association or fp32 widening law is assumed. The explicit fp32 source load
conversion precedes the scan. An absent scan backend fails explicitly.

`Proofs/TypedExecution.lean` checks a consistent abstract backend: fp16 input,
bf16 output, program ID 1, T=5, BT=2, all three chunks, active values and both
neighboring untouched output cells. This inhabits the full FP execution/domain;
it is not a GPU experiment. `bench/tests/FPScan.lean` distinguishes scan
parameters and precision, and `FPChunkCumsumSource.lean` checks the actual source
casts, scan, sum and unspecified padding AST. `ChunkCumsumPadding.lean`
checks both complete real source executions at program ID 1, T=5 and BT=2
with arbitrary padding, including seven: all five prefixes are retained.
Experimental shape is not a theorem restriction.

Host launch composition, fixed-width address overflow, concrete IEEE identities
and GPU performance are separate from these per-program contracts.

Independent receipt after the padding repair: `Logs/comparator-check-xj59vm4q` covers
all eleven case files and both source/padding regressions: 13 files, 93 named proof targets and 668 kernel theorem targets;
all 1,157 current snapshot inputs match. The targeted real/FP/typed-execution
build passes 3,452 targets. This is not a whole-corpus build result.
