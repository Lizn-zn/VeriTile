# l2_norm_triton2

This port retains both complete Python JIT bodies: forward row normalization
and backward input-gradient calculation. The complete public-entry dependency
build passes (3445 jobs), including source checks and module axiom audits. It is
not yet an independently accepted case; comparator replay remains pending.

`Kernels.lean` preserves both legacy APIs and adds explicit floating-storage
families: four forward X/Y types (fp32, fp16, bf16, fp64) and sixteen backward
X/DY type pairs, with DX matching X. All loads convert to fp32 before arithmetic.
Runtime epsilon is explicitly fp32; the final destination conversion is retained.
The candidate commutes only natural-number row/stride address products. Loads,
reductions, sqrt/division, arithmetic order, masks and stores remain intact.

`Correct.lean` exposes the legacy contracts and new complete real IO contracts
for both versions. Forward is `x / sqrt(sum(x²) + eps)`; backward is
`dy*rstd - sum(dy*x)/(sum(x²)+eps)*rstd*x`. These are independent mathematical
formulas. The new real contracts require positive block size and `N ≤ BLOCK_N`.
`Proofs/Math.lean` proves that zero-padded finite sums equal sums over the entire
logical row, including a partial final tile. The IO contracts include successful
flat execution, active read/write bounds and preservation of every unwritten
cell; their launch skin requires a zero undefined-value function.

`FPEquiv.lean` states whole-program equality, including all registers, memory and
failure, with no floating-point rule assumptions. Both print an empty assumption
list after the shared indexed-max semantics rebuild. `Proofs/Execution.lean`
executes both versions in an exact-real numeric backend with typed fp16 input,
bf16 gradient and fp16 output. At a nonzero row address, with N=1 and BLOCK_N=2,
the forward result is 3/4 and the backward result is 7/32; masked output cells
retain their sentinel. This supplies successful executions as well as equality.

All three backward pointers use the source's same `stride_x_row`, including DY.
The model preserves that addressing even when a host tensor might have a different
row stride. Host reshape/contiguity, launch geometry, allocation, fixed-width
address overflow and non-floating storage types are outside these per-program
contracts. Real sqrt/division are Lean's total real operations; the real formulas
do not claim IEEE rounding or prove the host's intended gradient for exceptional
numeric inputs. Independent comparator replay remains pending.
