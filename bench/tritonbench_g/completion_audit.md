# TritonBench-G Completion Audit

Objective: check every `bench/tritonbench_g` problem against
`review_criteria.md`, keep Python and Lean kernel bodies faithful, and ensure
each completed port has a standard `ComputeCorrect.Realizes` correctness
surface.

## Evidence Checked

- Directory coverage: `find bench/tritonbench_g -mindepth 1 -maxdepth 1 -type d`
  reports 184 work directories. Of these, 141 currently contain a `.py` /
  `.lean` port pair; the remaining 43 are README-only scaffolds and are not
  counted as completed ports by this audit.
- Python/Lean file coverage: `find bench/tritonbench_g -maxdepth 2 -name '*.py'`
  and the matching Lean query both report 141 files.
- Build gate: `lake build` succeeds.
- Per-port source elaboration gate: `bench/check_ports.sh` reports
  `TritonBench-G ports: 141 ok, 0 fail`.
- Placeholder scan:
  `rg -n "True := by|trivial|sorry|admit" bench/tritonbench_g -g '*.lean'`
  currently reports only the three algorithm-layer blocker theorems below.
- Correctness-surface scan:
  every `bench/tritonbench_g/*/*.lean` file now contains a
  `ComputeCorrect.Realizes` target or theorem.

## Remaining Blockers

These files must not be counted complete yet:

- `mean_reduction/MeanReduction.lean`
  - Kernel body is faithful.
  - Target: `mean_dim_kernel_correct_target`.
  - Named algorithm postcondition: `mean_dim_kernel_alg_post`.
  - Compute-facing theorem is reduced to this algorithm-layer postcondition.
  - Current local proof infrastructure: `meanLoopInvariant`,
    `meanMaskedAccumulatorSpec`, `meanChunkLoadSpec`,
    `meanLoopInvariant_init_of_zero_reg`,
    `meanLoopInvariant_step_of_accumulator_update`, and
    `meanFromMaskedAccumulatorSpec_eq_meanSpec`.
  - Remaining proof: prove the concrete loop body produces the `_mean =
    old + chunkLoad` register update, instantiate `forRange_inv`, then connect
    the final readout/store to `mean_dim_kernel_alg_post`.

- `embedding_triton_kernel/EmbeddingTritonKernel.lean`
  - Kernel body is faithful.
  - Target: `embedding_kernel_correct_target`.
  - Named algorithm postcondition: `embedding_kernel_alg_post`.
  - Compute-facing theorem is reduced to this algorithm-layer postcondition.
  - Remaining proof: instantiate the per-chunk write invariant with
    `forRange_inv` under `outOffsetFull` injectivity.

- `diag_ssm_triton/DiagSsmTriton.lean`
  - Kernel body is faithful.
  - Target: `diag_ssm_forward_kernel_correct_target`.
  - Named algorithm postcondition: `diag_ssm_forward_kernel_alg_post`.
  - Compute-facing theorem is reduced to this algorithm-layer postcondition.
  - Remaining proof: instantiate the recurrence invariant with `forLoop_inv`
    under full `diagSsmForwardOutOffset` injectivity.

Passing `lake build` is not sufficient to close the objective while these
placeholder theorems remain.
