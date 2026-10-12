# diag_ssm_triton

`Kernels.lean` contains all four complete typed JITs from
`diag_ssm_triton.py`: real and complex forward/backward. Matching
fp16/bf16/fp32 storage is supported for real inputs; complex64 is represented
by interleaved fp32 components. The candidate changes only the integer
column index from `pid * BLOCK_SIZE` to `BLOCK_SIZE * pid`.

`Correct.lean` exposes eight real specifications, one for each original and
candidate. Forward state satisfies `h[0]=s; h[t+1]=h[t]*lambda+x[t]`.
Backward visits time in reverse, computes `gx[t]=gy[t]+carry`, updates
`carry=gx[t]*conj(lambda)`, and accumulates
`gl += gx[t]*conj(previous_state)`; the final carry is `gs`.
Conjugation is identity for real values. These formulas use only initial
memory and the independent recurrences in `Proofs/Reference.lean`.

The displayed contracts include allocation/bounds, successful finite flat
execution, all masked output values and the full tagged-cell frame. Sequence
length may be zero, and BLOCK_SIZE need not fit inside the active width.
Complex forward supports in-place input/output and odd allocation bases.
The real positive-length adapter requires distinct x/y. Both backward
contracts display sufficient alias conditions for nonempty sequences; real
backward additionally separates gx from lambda. Empty backward scans permit
aliased terminal outputs because both are zero. Static loads and terminal
stores still require bounds at zero length. Zero unspecified values remain
an explicit condition of the flat-memory adapter.

`FPEquiv.lean` proves four complete typed equivalences with
`equiv_decompose`/`fp_prove`. `#print_fp_assumptions` reports none: the change
is exact integer arithmetic and leaves floating-point operations unchanged.

`bench/tests/DiagSsm*Execution.lean` checks the public specifications on
two-step real/complex scans, both reverse-time
branches, negative values, odd complex bases, partial tiles and unchanged
integer-tagged tail cells. `Proofs/ZeroWitness.lean` uses empty temporal
allocations and aliased terminal gradients. `Proofs/TypedExecution.lean`
inhabits the typed abstract semantics from empty registers for every storage
choice; it does not certify physical GPU rounding.

Scope is one kernel program, excluding Python autograd and batch reduction,
concurrent grid race freedom and fixed-width index overflow.

Validation passed: repository closure, full source correspondence for all four
JITs, precision/mask regressions, source-only corpus gates, 31 audit/layout
tests and independent comparator/trust replay of all 50 files. Final target
inventories and source fingerprints match `Logs/comparator-check-kmtknxce`.

Public-contract execution regressions now live in `bench/tests/`; proof helpers
do not import the public entries. The four relocated sources are byte-identical
to the accepted checks and independently pass at their new paths in `Logs/comparator-check-sce_37j5`.
