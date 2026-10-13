# matrix_transpose

`Kernels.lean` retains the entire upstream two-dimensional pointer construction,
unmasked matrix load, transpose, and unmasked output store. The Python wrapper
allocates only float16 input/output, and `typedKernel` preserves that storage
through the load and transpose without conversion. Its candidate commutes the
first natural-number input-address product. Both input strides, both output
strides, and both matrix dimensions remain symbolic.

`Correct.lean` proves both versions against independently declared windows:
output cell `(d,m)` reads `m * matrix_stridex + d * matrix_stridey` and writes
`d * out_stridex + m * out_stridey`. The identity formula on this transposed
input window is a matrix transpose, with successful flat execution and a frame
for every cell outside the output tile. Output-address injectivity is an explicit
necessary premise. No result values or successful execution are assumed.

`FPEquiv.lean` proves complete typed execution equality, including registers,
memory, and failure, using exact natural-number address arithmetic. It uses no
floating-point atom and requires no finite-value assumption. Source regressions
check actual float16 loads, transpose/store, and explicit input/output metadata
for both versions. Alias-based pointer metadata is supplied explicitly because
the DSL does not infer the final output region through pointer aliases.

The legacy real-valued declarations and docstrings are retained in the split.
Real contracts use a checked dtype erasure; the separate FP theorem preserves
the actual float16 source. The host's single-program launch, allocation, legal
Triton tile dimensions, fixed-width index overflow, and negative strides are
outside the per-kernel model. Other storage types are not wrapper profiles and
are not claimed. Independent comparator replay remains pending on this host.
