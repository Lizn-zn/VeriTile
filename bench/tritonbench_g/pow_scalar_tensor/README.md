# pow_scalar_tensor

- Source file: `pow_scalar_tensor.py`
- Corpus: TritonBench-G v1
- Status: DONE — DSL port (`PowScalarTensor.lean`), genuine scalar-base power
  closed-form spec (`powSpec = Real.rpow val0 (in0[t·in0_stride0])`), and
  `ComputeCorrect.Realizes` summary
  (`pow_scalar_tensor_output_summary_general`) proven sorry-free.

`pow_func_scalar_tensor_kernel_rank_1` is the FlagGems pointwise-codegen
rank-1 elementwise kernel (same skeleton as `relu_strided_buffer`): block-ptr
load with `boundary_check`, the inlined `pow_func_scalar_tensor` helper
(`_pow(val0.to(tl.float32), in0)` — runtime scalar BASE, loaded tensor
EXPONENT), block-ptr store. Both `one_tile_per_cta` constexpr branches are
verified: the monolithic one-tile branch per-lane, and the grid-stride loop
end-to-end via the loop invariant `gs_loop_readback` (cross-iteration
disjointness + input-region preservation). The headline is fully
dimension-general (`val0`, `s0`, strides, `tile_size0`, `tiles_per_cta` all
universally quantified) with honest side-conditions `0 < out0_stride0`, and
for the grid-stride branch `in0_ptr ≠ out0_ptr` and `0 < numPids`.

Modeling boundary: `_pow` is `Op.pow` / Mathlib `Real.rpow` — exact for
`val0 > 0`; for a negative base with non-integer exponent `rpow` returns its
junk-value convention where CUDA `pow` returns NaN (see the `Op.pow` doc
comment in `VeriTile/Triton/Core/Ast.lean`). The `.to(tl.float32)` /
`.to(*_ptr.type.element_ty)` casts erase to the identity on the ℝ channel.

This directory is the per-kernel workspace for the TritonBench-G
full-formalization roadmap.
