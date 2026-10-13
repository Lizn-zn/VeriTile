# rmsnorm_fused

The typed source, public real/FP entries, precision checks and numerical witnesses
have passed local Lean checks. Read-only source/contract review found no blocker.
Dependency integration passes all 3457 jobs; six final axiom checks use only
standard Lean axioms, and the four public contracts have validated `test-gap`
registrations. Independent comparator replay remains pending. Historical
specification headers and docstrings remain intact. This integration result
precedes the shared negative-constexpr DSL refresh.

The source retains both complete loops, row-based X/Y pointer updates, the
extra `tl.where` variance mask, both explicit fp32 input conversions and the
weight load without `other`. Logical column `k` uses `row * stride + k`
for both X and Y and `k` for W: contiguous columns and one shared row stride
are part of the actual kernel addressing. Host reshape/output-layout and launch
composition are outside the per-program contract. The reciprocal standard deviation is spelled
`1 / tl.sqrt(var + eps)`; it is not replaced by an rsqrt intrinsic. The prepared
216 floating profiles independently select input, weight and output storage
from fp32, fp64, fp16, bf16 and two fp8 carriers. Only fp64 weight promotes the
last product to fp64. The final cast represents implicit destination conversion.

The candidate commutes the natural row-by-stride multiplication in the output
pointer update. It preserves all other source statements. Whole-state FP
equivalence holds for arbitrary N/B and includes failure, aliases and undefined
masked values without numerical rewrite laws; its assumption audit is empty.

The new real contract covers arbitrary N with positive B, successful execution
of both full loops, an independent sum over the complete logical row, and every
untouched flat memory cell. It requires Y to differ from X and W so later blocks
continue reading the initial row snapshot. X and W may alias. Active addresses
must satisfy the explicit allocation bounds. `General.typed_full_trace_safe`
proves full typed-erased source trace safety from these bounds by following
actual intermediate states and both loop invariants.

The Python wrapper explicitly raises when N exceeds the chosen block size.
The retained wrapper helper covers `0 < N ≤ B` without output/input separation:
all input data are loaded before its single write phase. Eight nonempty numerical
checks cover both variants, fp64 input conversion and fp64 weight promotion, and
single-block and two-block calls, including untouched sentinels. Their exact-real
backend uses actual square root and reduction with identity casts; this is not an
IEEE numerical execution claim. Positive epsilon is proved sufficient for a
positive real square-root operand. The unrestricted mathematical contract uses
Lean’s totalized real square root and division, including N=0 or nonpositive
epsilon. The ordinary nonzero-denominator RMS interpretation uses N>0 and
positive epsilon; the wrapper helper already requires positive N.

The source wrapper does not reject discrete storage. Thus 216 floating
combinations are not an exhaustive source-supported dtype claim. Declared
signed input followed by its required `.to(tl.float32)` currently fails the
shared DSL; mixed/discrete output casts inherit the conversion gaps documented
in `../var_len_copy/conversion_gaps.md`. Negative signed row strides,
fixed-width address overflow and cross-program scheduling remain outside the
natural-address per-program model. Flat IO states use explicit zero undef;
FP comparison and raw-source safety do not impose that convention.
