# mean_reduction

`Kernels.lean` retains the complete row-chunk accumulation loop, zero padding,
fp32 accumulator, final column reduction/division and masked row store. Its
typed family covers all sixteen fp32/fp16/bf16/fp64 input/output storage pairs.
Every input converts to fp32 before addition, including the wrapper's fp64
input test. The candidate changes only the natural-number program-base product
from `pid * BLOCK_M` to `BLOCK_M * pid`.

`Correct.lean` retains the legacy full-row invariant and independent sum/N
formula. The new real contract connects both complete typed sources through
checked dtype erasure to the streamed input windows, successful flat execution
and full frame outside active output rows. It requires positive reduction size
and tile dimensions. The older mathematical zero-size statements are retained
without presenting them as valid zero-divisor GPU launches. The legacy abstract
rounding contract describes the erased program; the new real contract uses
identity rounding only to express that mathematical semantics through the
existing streaming IO interface.

`FPEquiv.lean` preserves complete typed execution, registers, memory and failure
without numerical atoms. It does not reassociate additions, replace division,
or claim that floating accumulation equals the mathematical mean.
`Proofs/Source.lean` checks every dtype/version combination for the actual load
and conversion, zero padding, fp32 loop addition, fp32 sum/division, final store
carrier/conversion and explicit input/output metadata.

The DSL's fp64 explicit-load conversion must invoke the conversion backend,
not the fp32-load backend. Dedicated regressions distinguish these operations
and preserve old mathematical-region inference. Pointer rebinding loses implicit
output dtype information, so the source explicitly spells the destination
conversion for half/bfloat/double output, once; fp32 output is uncast.

This is a per-program contract for the contiguous matrix produced by the host's
`dim_compress`. Host permutation/allocation/launch composition, legal Triton
tile sizes, fixed-width address overflow and nonfloating storage profiles are
outside the current proof scope. Independent comparator replay remains pending.
