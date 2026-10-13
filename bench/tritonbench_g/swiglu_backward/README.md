# swiglu_backward

The complete kernel loads X, Y and DOUT before writing DX, DY and then optional
OUT. Both `RECOMPUTE_OUTPUT` outcomes are retained, including the guarded OUT
row-pointer update and guarded forward recomputation after the gradient stores.
Both program axes, all six row strides, masked zero padding and every tensor
operation remain symbolic and in source order.

The typed source follows dtype relationships guaranteed by the wrapper:
X and Y share the dtype of `xy`; DX and DY share the dtype of `dxy`. Input,
upstream DOUT, gradient-output and optional recomputed-output storage each vary
independently over fp32, fp16, bf16, fp64, fp8e4nv and fp8e5. Thus 1296 storage
profiles are covered for each candidate and each flag. Low-level calls breaking
those X/Y or DX/DY correlations are outside this source family. Integer/boolean
storage and other formats remain unsupported profiles.

To avoid duplicating thousands of ASTs, `Kernels.lean` shares one complete fp32
source template per variant and substitutes only typed load/store nodes. It
does not substitute input values or skip computations. `Proofs/Precision.lean`
proves exact equality with independent complete source quotations for every
homogeneous storage and two mixed profiles. Generic profile checks pin all
three typed loads, both gradient stores, optional output store, statement count
and IO roles. Every input is explicitly converted to fp32. Output conversions
expose the Python source's implicit store conversions; this is a translation
surface exception. Fp64 boundaries retain their inner precision scopes. The
source has no literal-only multiplication requiring Python constexpr folding.

`Correct.lean` preserves the legacy contract and adds independent scalar
formulas for both gradients and the optional forward value. Both variants
include successful full flat execution, active read/write bounds, and every
untouched memory cell. The readback contract assumes DX, DY and OUT are pairwise
distinct, as the legacy contract does; this prevents later stores overwriting
an earlier output channel. This remains a stated sufficient condition when OUT
is inactive. Input/output names may alias because all input loads precede
stores. The FP equivalence itself imposes no alias restriction and preserves
ordered writes, full state and failure for both flags without numerical atoms.

The mathematical sigmoid denominator is proved positive. Real correctness
forgets rounding; it is not an analytic differentiation theorem or a hardware
rounding guarantee. The candidate only commutes the six natural row-address
products, including the conditional OUT product, leaving floating arithmetic
unchanged. Typed numerical witnesses use nonzero values, both nonzero program
axes, distinct strides, both flags and both variants. They check both gradients,
the enabled/disabled OUT behavior, and masked padding in an exact-real backend.
Additional OUT=DX witnesses check the final-store overwrite when recomputation
is enabled and preservation of the DX gradient when it is disabled.

The null OUT passed by the wrapper when recomputation is disabled is represented
by an unused symbolic output role. No active bound or readback is required for
that role. Host reshape/chunk view construction, autotuning, grid composition,
cross-program races, negative strides, fixed-width address overflow, arbitrary
partially overlapping or byte-reinterpreted views, and target-specific fp8
availability remain outside this per-program proof. Independent comparator
replay and GPU execution remain pending.

All helper/public compiler gates, legacy declaration audit and independent
source review pass locally. The combined four-entry SwiGLU Lake gate passed
(3459 jobs). This local result does not replace independent comparator replay.
