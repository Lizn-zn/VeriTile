# max_reduction

Source: `max_reduction.py`, all three Triton kernel bodies. The Python tests use
fp32; the wrappers do not restrict the input dtype. The current typed source
specializations cover fp32, fp16, bf16, float8e4nv, float8e5, and fp64.
Other storage profiles remain outside this typed-source claim.

`Kernels.lean` retains the legacy real transcriptions and adds complete typed
sources. The first-stage candidate commutes a natural index product; the
second-stage candidate complements the reversed natural bound; the dimension
candidate commutes the row-index product. These transformations require no
floating arithmetic atom. They retain all loads, masks, reductions, casts,
pointer calculations, and stores.

Precision follows [Triton 3.5.1 standard.py](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/standard.py#L134-L183):
ordinary max promotes half, bf16, and fp8 to fp32. Indexed max retains half/fp8 and
promotes bf16 to fp32. Its selected value and index use the same paired opaque
backend, distinct from ordinary maximum, so NaN/tie behavior is not silently
identified. Only default/explicit `True` leftmost ties are supported; `False`
is rejected. The index storage allocated by `max_dim` is signed int64.
The indexed combine retains equal fp8 operand types through `where`, per
[Triton 3.5.1 semantic.py](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L96-L98)
and its [selection implementation](https://github.com/triton-lang/triton/blob/v3.5.1/python/triton/language/semantic.py#L1484-L1498).
These frontend profiles do not assert every GPU target accepts every fp8 type.

The translation currently makes implicit output-store conversions explicit:
the two scalar value stages narrow their fp32 maximum once for half/bf16/fp8
output storage; the dimension stage narrows bf16 values once and converts the
natural index carrier to signed integer storage. This is a translation-surface
exception, not an arithmetic optimization or an assertion that bool/natural
storage substitutes for int64.

`Proofs/Math.lean` defines an independent finite maximum of the active input
window. Padding is mathematical bottom and readback of an all-padding real
result is zero; this does not identify hardware negative infinity with zero.
The real contracts require a nonempty reduction tile. The host launch,
autotuning, and composition of the two passes remain the trusted boundary.
`max_dim` uses a contiguous copy; the global `max` wrapper does not, so its
kernel window is the actual linear memory window supplied by the caller.

The public real contracts `stage1_real_correct`, `stage2_real_correct`, and
`dimension_real_correct` quantify both original and candidate independently.
They cover successful execution, original input windows, independent finite
maximum/leftmost-index formulas, and the full flat-memory frame. Dimension
reduction requires distinct value/index output regions and injective active
row addresses, preventing conflicting stores. The index uses the framework's
exact natural/signed-integer channels; fixed-width integer validity is a
frontend obligation, not a claim of an IEEE or machine-integer implementation.

The three public FP equivalences cover whole kernels and admit no floating
atoms. `Proofs/Execution.lean` also proves a successful nonempty dimension
execution for all six storage profiles and both candidates: input values
seven and nine produce value nine and signed index one. This witnesses the
actual load/reduction/cast/store path rather than two unsupported executions.
`Proofs/Source.lean` checks precision, typed padding, paired indexed reduction,
output narrowing and signed index storage at exact AST level. The fp64 profiles
use raw real-carrier loads under explicit fp64 computation metadata; their
ordinary and indexed reductions retain fp64, without invoking the fp32 load
callback or adding an identity conversion.

All 31 legacy declaration headers and docstrings were retained verbatim.
Targeted Lean checks and standard-axiom audits passed. External comparator
replay remains pending; no GPU execution or universal wrapper-dtype acceptance
is claimed. Integer, boolean, and other unimplemented wrapper storage profiles
remain explicit gaps rather than substitutions by the legacy real transcription.

The integrated public `Correct`/`FPEquiv` dependency build passes (3451 jobs). Nine contracts have validated source fingerprints and `test-gap` status; independent comparator replay remains pending.
