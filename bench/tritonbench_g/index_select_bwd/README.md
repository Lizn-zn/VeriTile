# index_select_bwd

The Python kernel is a masked scatter **overwrite**. It does not add gradients
and does not use atomics. All gradient and signed-index loads precede its one
scatter store. The wrapper checks matching strides and launches BI=1, BC=512;
the proofs retain arbitrary BI/BC, masks, both program axes, and both strides.

`Kernels.lean` preserves the legacy mathematical/natural-index transcription
and adds complete typed sources for six incoming and six destination floating
formats (fp32, fp16, bf16, fp8e4nv, fp8e5, fp64), independently, with int32 or
int64 index storage. Both variants cover all 72 profiles. Integer/boolean
gradient storage and other index formats remain explicit unsupported profiles;
the wrapper itself does not reject them. The candidate only
commutes `pid1 * BC` into `BC * pid1` in the column-address calculation.

The incoming gradient is explicitly converted to fp32 as in Python. Output
store conversions are made explicit for the other five destination formats;
this is a translation-surface exception exposing Triton's implicit store cast.
Fp64 input conversion retains the inner fp64 load scope before the fp32 cast.
Fp64 output conversion retains the fp32 operand scope before the fp64 cast.
The fp32 input's explicit `.to(tl.float32)` remains an explicit cast in the AST.
`Proofs/Precision.lean` checks exact load/cast/store shapes for every profile,
including actual `.int` index reads and the missing `other` on masked loads.

The new real contract uses an independent row-major last-active-write formula
for every allocated memory cell, plus a full flat-memory frame. Duplicate
indices are covered; no injectivity or assumed output values are required.
Input/output names may alias in the typed-cell memory model: values come from
the original snapshot because all reads precede the store. Arbitrary byte-level
reinterpretation between differently sized aliased tensor views is not modeled.
The legacy natural-index contract and all its declaration headers/docstrings
remain available separately.

Signed index arithmetic is retained through multiplication and addition.
The natural-address memory model requires nonnegative active signed rows and
bounds on actual reads and writes. Python does not add these checks; the
contract states them as domain requirements. Negative pointer displacements
are outside the source-fidelity claim, rather than being interpreted as Python
negative-index wrapping or silently substituted with natural index storage.
The two integer declarations share an exact `Int` carrier; fixed-width overflow
validity remains a frontend obligation.

The per-program interpreter orders scatter lanes in row-major order. This is
the stated mathematical scatter semantics, not a GPU race-resolution guarantee.
The Python duplicate-index tests collide across programs: host scheduling,
launch composition, and any unique global output for racing writes remain
outside the theorem. Target-specific fp8 availability is also not asserted.

The full-kernel FP equivalence admits no floating arithmetic atoms. Typed
execution witnesses cover every profile, duplicate destination rows, masked
padding, and same-storage input/output aliasing. External comparator replay
and GPU execution remain pending.

The integrated Correct/FPEquiv/precision dependency build passes. Three public
contracts have validated source/support fingerprints and `test-gap` status.
