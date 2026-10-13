# square_matrix

`Kernels.lean` retains the complete masked row load, negative-infinity padding,
elementwise square and masked store from `square_matrix.py`. The source-facing
family covers matching fp32, fp16 and bf16 input/output storage. Its candidate
commutes only the integer row-index/stride multiplication in the input address.
The original eight declarations and their contracts survive the split unchanged.

`Correct.lean` proves the independent formula `xs i * xs i` for both typed
versions under their checked real projection. It covers successful flat-memory
execution, active-address safety and every cell outside active writes. Program
IDs, input/output row strides, column count and tails are symbolic; new typed
contracts require positive block size. A full logical row is covered when
`n_cols <= BLOCK_SIZE` and the input's inner stride is one; otherwise the contract
describes the actual contiguous-address tile prefix. The wrapper passes row
strides only, so arbitrary strided logical columns are not implemented by this source.

`FPEquiv.lean` proves equality of complete typed executions, including registers,
memory and failure. Exact natural-number index commutation needs no floating-point
atom or finite-operand premise. Floating multiplication and storage are unchanged.
`Proofs/Source.lean` checks projection, explicit input/output wiring and final
store precision for all six dtype/version combinations.

The alias-based DSL port declares input/output metadata explicitly because alias
tracking does not infer the final store's output region. The body is unchanged.
These are per-program contracts over natural-number offsets; host launch,
negative strides, fixed-width address overflow and other storage profiles remain
outside the current scope. Targeted Lean builds pass; independent comparator
replay remains pending on this host.
