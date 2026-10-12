# Fifth-order spherical harmonic polynomials

Status: development in a separate staging directory; repository acceptance is
pending. No GPU experiment has been run for this review.

`Kernels.lean` contains the complete `fifth_order_fwd` and `fifth_order_bwd`
sources from `fifth_order_sph_harmonics.py`, including every coefficient,
coordinate load, upstream load, old-gradient load, mask and store. Both
candidates only commute the natural-number coordinate-block product. The
backward transcription includes the source's `coord_stride = 3` assignment.

Input/output storage choices are independent fp16, bf16 or fp32; backward
also permits independent coordinate-gradient and upstream-gradient storage.
Unannotated decimal coefficient assignments create fp32 scalar tensors. Input
powers retain input precision until they combine with those coefficients.
Each implicit destination conversion is explicit in the source. Negative
literal constants are staged as single constants; negating a named tensor
coefficient remains an arithmetic operation. Source/type regression checks
cover this distinction. Masked loads without `other` retain unspecified
values rather than introducing zero padding.

`Correct.lean` exposes the independent expanded eleven-channel polynomial
and 3-by-11 backward coefficient tables. It has four `Spec.Real` statements,
for both complete sources and their candidates. The contracts quantify over
symbolic block sizes, offsets, strides and finite disjoint allocations; require
bounds for all active loads/stores; prove successful flat execution; describe
all written values; and preserve every other tagged memory cell. The shared
flat adapter's zero-undefined-value condition is explicit.

Forward requires output stride greater than ten, so channels and rows do not
collide, and sufficient xyz coordinates for each active output row. Backward
requires a complete xyz row and eleven upstream channels for each active
coordinate. It adds the update to the initial coordinate-gradient values.
All coordinate/upstream/old-gradient loads precede stores, so the formulas
refer to the initial memory even when these arguments name the same region.
The contracts are per program, without a host launch/composition or race claim.

The backward table must not be described as the exact derivative of the
forward decimal table. The coefficients are independently rounded in the
Python source. `Proofs/RealMath.lean` proves that at `(1,0,0)` the x derivative
of forward channel 2 is `8.67152307844475`, while the corresponding backward
coefficient is `8.67152307844476`. Their exact-decimal real difference is
`10^-14`. This is neither a measurement nor a bound on physical floating-point
error. The staged finite execution regression connects this coefficient fact
to the complete backward kernel once its checks pass.

`FPEquiv.lean` exposes two whole typed-state/failure equivalences. Input domains
and numerical FP atom lists are empty. The proofs preserve source arithmetic
order and precision rather than applying real polynomial identities to FP
operations. The numerical real simplification lives in the real proof layer.
Typed execution witnesses use an abstract algebra to establish nonempty
executions; physical IEEE rounding, fixed-width overflow, scheduling and
performance are outside scope.

Local checks have passed for all-storage source elaboration and precision,
real erasure and the address candidate, forward real execution and finite IO,
and both whole typed FP equivalences with their public entry. The complete-source
backward successful execution, trace-safety, flattening and tagged-cell frame
proofs have also passed. Polynomial readback and complete finite witnesses,
all-storage typed execution witnesses, repository integration/gates and fresh
independent comparator/trust replay remain pending. Source comparison checks
both complete statement trees, 84 decimal assignments and all 14 stores against
the current Python source. No local check is treated as repository acceptance.

Typed nonempty-execution witnesses use an explicitly singleton-valued abstract
algebra: integer addresses, masks and storage tags remain visible; numerical
operations are collapsed to one value. With no FP atoms this establishes
inhabited typed execution only. Numerical polynomial outputs are checked by the
separate real finite-memory witnesses; no physical rounding claim is made.


The development sources were recovered from this chat after an environment
restart. Current acceptance remains pending; all modules and execution
witnesses are being recompiled from the recovered sources. Historical
temporary build receipts were not reconstructed as successful checks.
