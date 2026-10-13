# rmsnorm_triton

Migration is in progress. New typed source and full-source real/FP contracts are
not yet accepted: the integer-power source primitive needs a shared semantics
extension. Historical contracts remain available during the split.

The source has two complete tiled loops, two program IDs and seven strides.
Both input passes explicitly convert X to fp32. Variance calls
`tl.extra.cuda.libdevice.pow(x.to(tl.float32), 2)`, then fp32 sum/division and
`tl.math.rsqrt`; pass two multiplies x32 by rstd32 and the uncast weight.
Only fp64 weight widens the final floating product. Weight loads have no
`other`; input loads explicitly use zero. The candidate commutes only the
first natural row-address product.

The prepared floating source family has 216 independent input/weight/output
storage combinations; the Python wrapper uses output storage equal to input.
This is not exhaustive dtype coverage: the wrapper does not reject discrete
inputs or outputs. Declared signed input followed by `x.to(tl.float32)` fails
in the current DSL (`tl.cast: signed integer casts are not modeled yet`).
Mixed/discrete output conversions inherit the recorded shared conversion
obligations from `var_len_copy/conversion_gaps.md`. Natural address geometry
also leaves negative signed strides and fixed-width overflow outside scope.

Pinned Triton 3.5.1 dispatches `(fp32,int32)` power to `__nv_powif`, distinct
from `__nv_powf` and multiplication:
[official libdevice source](https://raw.githubusercontent.com/triton-lang/triton/v3.5.1/third_party/nvidia/language/cuda/libdevice.py).
The existing DSL's literal-two multiplication shortcut cannot serve as the
full-source FP transcription. The new proof must retain the integer exponent
and an independent opaque integer-power backend. The identity `x^2=x*x` is
used only in exact-real mathematical proofs, never as an assumed FP law.

All 72 historical declaration headers and adjacent docstrings are preserved
in the initial split. Public compilation, exact precision checks, nonempty
numeric runs and independent final review remain pending.
