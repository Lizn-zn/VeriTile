# fused_activation

The new public real and FP specifications require `0 < num_weights`, as the
Python source takes a remainder by this parameter. The legacy
`fused_add_mul_activation_kernel_correctness` signature is retained in the
total-Nat model; its zero-divisor case is not a valid Triton execution claim.

`Kernels.lean` retains the full fp32 masked kernel from `fused_activation.py`,
including the modulo-indexed bias, all three loads, both sigmoid/ReLU branches,
and the in-place store. The candidate commutes only the final bias addition;
there is no performance claim.

`Correct.lean` states independent real correctness for both sources, using
`sigmoid(multiplier * input + x + bias)` or ReLU of the same expression.
The grouped IO covers symbolic program IDs, element counts, bias counts and
block sizes, termination, active-address bounds and the complete memory frame.
The original public theorem remains available in `Correct.lean`, and its helpers
are in `Proofs/Real.lean`. Host launch composition is outside the per-program scope.

`FPEquiv.lean` uses the fp32 profile and the admitted `add_commute` atom.
Its domain requires finiteness of the actual computed `multiplier * input + x`
and actual loaded bias at the addition site, including evaluated inactive lanes.
The conclusion preserves the whole execution result, registers, memory and
failure behavior, while retaining the opaque sigmoid and comparison operations.
The real proof does not use the FP assumptions.
