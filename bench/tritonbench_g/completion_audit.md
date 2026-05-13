# TritonBench-G Completion Audit

Objective: check every `bench/tritonbench_g` problem against
`review_criteria.md`, keep Python and Lean kernel bodies faithful, and ensure
each completed port has a standard `ComputeCorrect.Realizes` correctness
surface.

## Prompt-to-Artifact Checklist

| Requirement | Evidence | Current status |
|---|---|---|
| Check every `bench/tritonbench_g` problem. | 184 work directories are present; 141 currently have `.py` / `.lean` port pairs, and 43 are README-only scaffolds. | Covered for completed port pairs; scaffolds are not counted as completed ports. |
| Ensure every completed Python port has a Lean port. | Python/Lean file counts both report 141; `bench/audit_tritonbench_g.sh` enforces the count match. | Passing. |
| Ensure Lean ports compile. | `bench/check_ports.sh` reports `TritonBench-G ports: 141 ok, 0 fail`; the audit script reruns this gate. | Passing. |
| Apply `review_criteria.md` faithful-translation rules. | Mechanical gates check dtype-load additions, `keep_dims` substitutions, `+=` coverage, normalized pointer-update lhs, `rsqrt` preservation, Lean-only `tl.where`, `tl.*(...)` call set/order, kernel control-flow counts, and statement lhs order. | Mechanically covered for the listed must-fix patterns; still not a substitute for human line review of arbitrary arithmetic structure. |
| Fix Python/Lean mismatches found by the sweep. | Recent fixes restored faithful loop/tuple/statement surfaces and moved policy checks into `bench/audit_tritonbench_g.sh`; current audit passes. | No current mechanical mismatch. |
| Ensure completed ports expose a standard correctness surface. | Audit scans every `.lean` for `ComputeCorrect.Realizes`, `ComputeRefine.Realizes`, `ComputeCorrect.General`, or a named `correct_target`. | Passing. |
| Do not count placeholder proofs as complete. | Placeholder scan for `True := by`, `trivial`, `sorry`, and `admit` reports no matches. | Passing. |
| Do not close while algorithm-layer proof obligations remain. | Audit now checks that explicit `hAlg` blockers are exactly the documented set: `mean_reduction`, `embedding_triton_kernel`, and `diag_ssm_triton`. | Not complete. |

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
- Mechanical audit gate: `bench/audit_tritonbench_g.sh` reports
  `TritonBench-G audit gates passed`, covering Python/Lean count matching,
  port elaboration, placeholder-proof scanning, and correctness-surface
  scanning. It also checks that compiled ports do not still advertise README
  `TODO` status and that Python `.to(tl.float32)` casts missing from Lean are
  covered by an explicit documented slice/scope note. The same gate rejects
  unexpected algorithm-layer `hAlg` blockers outside the documented set of
  three remaining obligations. It rejects Lean-only `tl.load(..., dtype=...)`
  annotations and `keep_dims` reduction substitutions, both of which are
  must-fix deviations under
  `review_criteria.md`. It also flags Python `+=` statements missing from Lean
  unless the Lean port documents that the update is outside a proof slice or
  branch/surface specialization; this includes a normalized left-hand-side
  check that treats names like `a_ptr` and `A` as the same pointer. It checks
  that upstream `rsqrt` calls are preserved rather than rewritten as reciprocal
  square roots. It also rejects Lean-only `tl.where` statements, another
  must-fix "extra statement" pattern in `review_criteria.md`. Finally, it
  compares the Python and Lean `tl.*(...)` call surfaces and requires any
  missing or extra call to be covered by an explicit slice/specialization note.
  It also compares `for` / `while` / `if` counts inside the Python
  `@triton.jit` kernel body and Lean `triton { ... }` body to catch
  unannotated control-flow rewrites, and compares the ordered `tl.*(...)` call
  sequence to catch unannotated call reordering. A statement left-hand-side
  sequence scan covers top-level assignments, `+=` updates, annotated
  assignments, and tuple assignments, so added/removed/reordered non-call
  statements are also checked mechanically.
- Placeholder scan:
  `rg -n "True := by|trivial|sorry|admit" bench/tritonbench_g -g '*.lean'`
  currently reports no matches.
- Correctness-surface scan:
  every `bench/tritonbench_g/*/*.lean` file now contains a
  `ComputeCorrect.Realizes` target or theorem.

## Remaining Blockers

These files must not be counted complete yet:

- `mean_reduction/MeanReduction.lean`
  - Kernel body is faithful.
  - Target: `mean_dim_kernel_correct_target`.
  - Named algorithm postcondition: `mean_dim_kernel_alg_post`.
  - Public correctness theorem exposes this algorithm-layer postcondition as an
    explicit hypothesis; there is no `True` / `trivial` placeholder.
  - Current local proof infrastructure: `meanLoopInvariant`,
    `meanMaskedAccumulatorSpec`, `meanChunkLoadSpec`,
    `meanLoopInvariant_init_of_zero_reg`,
    `meanLoopInvariant_step_of_accumulator_update`,
    `meanLoopInvariant_register_reduceSum_to_meanSpec`, and
    `meanFromMaskedAccumulatorSpec_eq_meanSpec`. The masked load bridge is
    exposed by `meanChunkLoadSpec_active` and `meanChunkLoadSpec_inactive`;
    final masked scatter/readback is bridged by
    `meanStoreFromMaskedAccumulator_alg_post`, with
    `meanOutOffset_injective` and
    `meanStoreFromMaskedAccumulator_alg_post_default` packaging the row-output
    no-collision side; `meanLoopInvariant_to_scatter_alg_post` packages a final
    loop invariant plus scatter state into the algorithm postcondition. The
    concrete post-loop AST suffix is named `meanPostLoop`.
    `meanOutOffset_injective_col1` and
    `meanStoreFromExpandedMaskedAccumulator_alg_post` now cover the actual
    expanded `[BLOCK_M, 1]` store shape emitted after `mean[:, None]`.
    `meanPostLoop_step_alg_post` consumes the actual
    `stepStmts (meanPostLoop ...)` execution from the final accumulator and
    store-register assumptions. `meanPreLoop`, `meanLoopBody`, and
    `meanProjectedBody` name the projected algorithm body components, and
    `mean_dim_kernel_toAlg_body` proves that this split is the actual
    `toAlgKernel.body`. `meanPreLoop_step_regs` proves the concrete pre-loop
    execution facts: zero `_mean`, expanded `Mean` pointer tile, and
    `row_mask`.
  - Remaining proof: prove the concrete loop body produces the `_mean =
    old + chunkLoad` register update, instantiate `forRange_inv`, then feed
    the final loop invariant into
    `meanPostLoop_step_alg_post`.

- `embedding_triton_kernel/EmbeddingTritonKernel.lean`
  - Kernel body is faithful.
  - Target: `embedding_kernel_correct_target`.
  - Named algorithm postcondition: `embedding_kernel_alg_post`.
  - Public correctness theorem exposes this algorithm-layer postcondition as an
    explicit hypothesis; there is no `True` / `trivial` placeholder.
  - Current local proof infrastructure includes `embeddingLoopInvariant`,
  `embeddingLoopInvariant_zero`, `embeddingLoopInvariant_step_of_chunk_write`,
    `embeddingLoopInvariant_to_alg_post`, and
    `embeddingLoopInvariant_to_alg_post_of_final`. The final bridge now accepts
    the `forRange_inv` shape `BLOCK_N ≤ final` rather than requiring a
    definitionally exact final offset. Old-prefix/current-chunk disjointness is
    factored into `embeddingPrefixIndex_ne_currentChunk` and
    `embeddingOldPrefix_outOffset_ne_currentChunk`; aligned chunk coverage is
    captured by `embeddingChunkLane_lt_of_aligned_start` and packaged for the
    step theorem by `embeddingChunkLaneBound_of_aligned_start`; current-chunk
    masked scatter readback is bridged by `embeddingCurrentChunkScatter_write`, with
    `embeddingCurrentChunkNoCollision_of_full_injective` deriving its
    no-collision premise from full output injectivity plus chunk-bound coverage.
    `embeddingCurrentChunkScatter_preserve_old` covers old-prefix preservation
    for that current-chunk scatter, and
    `embeddingLoopInvariant_step_of_current_chunk_scatter` packages the scatter
    state into the next-prefix invariant.
  - Remaining proof: instantiate the per-chunk write invariant with
    `forRange_inv` under `outOffsetFull` injectivity, including the concrete
    loop-body store.

- `diag_ssm_triton/DiagSsmTriton.lean`
  - Kernel body is faithful.
  - Target: `diag_ssm_forward_kernel_correct_target`.
  - Named algorithm postcondition: `diag_ssm_forward_kernel_alg_post`.
  - Public correctness theorem exposes this algorithm-layer postcondition as an
    explicit hypothesis; there is no `True` / `trivial` placeholder.
  - Current local proof infrastructure includes
    `diagSsmForwardLoopInvariant`, `diagSsmForwardLoopInvariant_zero`,
    `diagSsmForwardLoopInvariant_step_of_time_write`, and
    `diagSsmForwardLoopInvariant_to_alg_post`. Old-time/current-time
    disjointness is factored into `diagSsmForwardIndex_ne_currentTime` and
    `diagSsmForwardOutOffset_ne_currentTime`; `diagSsmMaskedStateTile_succ`
    exposes the active-lane register update shape. Current-time offset/spec/
    active unfold lemmas are named `diagSsmForwardOutOffset_currentTime`,
    `diagSsmForwardSpecAt_currentTime`, and
    `diagSsmForwardActive_currentTime`; current-time masked scatter readback is
    bridged by `diagSsmForwardCurrentTimeScatter_write`, with
    `diagSsmForwardCurrentTimeNoCollision_of_out_injective` deriving its
    no-collision premise from the full output injectivity hypothesis.
    `BlockState.scatter_prop_masked_preserves_other_offset` and
    `diagSsmForwardCurrentTimeScatter_preserve_old` cover old-time preservation
    for that current-time scatter, and
    `diagSsmForwardLoopInvariant_step_of_current_time_scatter` packages the
    updated state register plus scatter state into the next loop invariant.
  - Remaining proof: instantiate the recurrence invariant with `forLoop_inv`
    under full `diagSsmForwardOutOffset` injectivity, including the concrete
    loop-body register/scatter shape.

Passing `lake build` is not sufficient to close the objective while these
algorithm-layer obligations remain.
