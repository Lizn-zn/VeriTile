# TritonBench-G Tier 1 Review Checklist

Use this checklist to manually review the Tier 1 kernels that currently have a
VeriTile Lean port and public correctness theorem.

For each item, check:

- Python source and Lean DSL are aligned except for mechanical Lean syntax.
- The Lean theorem states the intended kernel behavior.
- The theorem surface uses `ComputeCorrect.*` or `ComputeRefine.*`.
- The file compiles with `lake env lean <Lean file>`.

## Ready To Review

- [x] `add_example.py`  
  Lean: `bench/tritonbench_g/add_example/AddExample.lean`

- [x] `add_value.py`  
  Lean: `bench/tritonbench_g/add_value/AddValue.lean`

- [x] `cosine_compute.py`  
  Lean: `bench/tritonbench_g/cosine_compute/CosineCompute.lean`

- [x] `kldiv_compute.py`  
  Lean: `bench/tritonbench_g/kldiv_compute/KldivCompute.lean`

- [x] `swiglu_fwd.py`  
  Lean: `bench/tritonbench_g/swiglu_fwd/SwigluFwd.lean`

- [x] `logsumexp_fwd.py`  
  Lean: `bench/tritonbench_g/logsumexp_fwd/LogsumexpFwd.lean`

- [x] `matrix_transpose.py`  
  Lean: `bench/tritonbench_g/matrix_transpose/MatrixTranspose.lean`

- [x] `max_reduction.py`  
  Lean: `bench/tritonbench_g/max_reduction/MaxReduction.lean`

- [ ] `mul_exponent_compensator.py`  
  Lean: `bench/tritonbench_g/mul_exponent_compensator/MulExponentCompensator.lean`

- [ ] `relu_triton_kernel.py`  
  Lean: `bench/tritonbench_g/relu_triton_kernel/ReluTritonKernel.lean`

- [ ] `sin_computation.py`  
  Lean: `bench/tritonbench_g/sin_computation/SinComputation.lean`

- [ ] `sin_kernel.py`  
  Lean: `bench/tritonbench_g/sin_kernel/SinKernel.lean`

- [ ] `softmax_triton1.py`  
  Lean: `bench/tritonbench_g/softmax_triton1/SoftmaxTriton1.lean`

- [ ] `softmax_triton2.py`  
  Lean: `bench/tritonbench_g/softmax_triton2/SoftmaxTriton2.lean`

- [ ] `square_matrix.py`  
  Lean: `bench/tritonbench_g/square_matrix/SquareMatrix.lean`

- [ ] `triton_mul2.py`  
  Lean: `bench/tritonbench_g/triton_mul2/TritonMul2.lean`

- [ ] `vector_addition.py`  
  Lean: `bench/tritonbench_g/vector_addition/VectorAddition.lean`

- [ ] `vector_addition_custom.py`  
  Lean: `bench/tritonbench_g/vector_addition_custom/VectorAdditionCustom.lean`

## Checked But Not Ported
