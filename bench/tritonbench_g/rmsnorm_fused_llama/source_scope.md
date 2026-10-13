# Source and proof boundary

All source/precision, real IO/frame/safety, FP and numerical leaf gates passed.
Dependency-aware integration and independent replay remain pending.

The certified source family is the complete `_rms_norm_fwd_fused` body, including
both `range(0,N,B)` loops and all masked accesses. There is no extra `where` in
the variance loop. X is converted to fp32 in both passes, W is converted to fp32
in the second pass, and all normalization/weight arithmetic is fp32. The final
explicit fp16 cast precedes destination conversion even for fp32/fp64 output.
No cast or store is reordered by the candidate; only `row*stride` in the output
pointer prefix becomes `stride*row` over natural addresses.

Six independent floating storage formats generate 216 X/W/Y profiles. This
includes and extends the wrapper's X=Y dtype relationship. Runtime epsilon is
modeled by its fp32 arithmetic occurrence. The backend controls rounding modes,
conversion dispatch, reduction, exceptional results and failure; equivalence
has no numerical side conditions and includes every state cell and register.

Real correctness is explicitly about dtype erasure. Its independent formula is
`X[row*stride+k] / sqrt(sum_j X[row*stride+j]^2 / N + eps) * W[k]`.
The actual two-pass execution is proved for any N and positive B. Snapshot IO
requires X != Y and W != Y. X and W may coincide. Full raw-cell framing preserves
every address outside the active row, including other regions. Trace safety uses
active input, weight and output bounds. Flat transport additionally uses the
existing allocation separation/coverage and zero-undef bridge premises.

N=0 follows totalized real arithmetic rather than a finite hardware RMS claim.
For the ordinary positive-variance interpretation use N>0 and positive epsilon.
Missing `other` on W retains unspecified inactive values, which are never stored.

The wrapper's override is exactly 16384, after a guard based on a different block
choice; it does not justify N<=B for all storage sizes. The proof therefore does
not use a single-block launch premise. Host reshaping, layout validation, launch
coverage, concurrent races, native compiler block-shape constraints, and device execution are outside the per-program
contract. Addresses and strides are unbounded Nat: negative strides and native
fixed-width arithmetic overflow remain outside this family. Source-accepted
integer/bool and mixed integer/floating conversion families remain unverified;
the six-format floating family is not a claim about every torch dtype.

Numeric executions instantiate an exact-real, nonempty WithBot-real backend with
actual sum and square root, identity casts and distinct floating storage tags.
They demonstrate successful multiblock execution and unchanged tail sentinels,
not a bit-accurate IEEE implementation.
