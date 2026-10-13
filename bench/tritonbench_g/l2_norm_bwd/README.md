# l2_norm_bwd

`Kernels.lean` contains the preserved historical transcription and a shared typed family for the complete Python backward kernel. X/DX share a storage dtype; DY independently selects fp16, bf16, fp32 or fp64, giving 16 storage pairs. Both masked loads convert to fp32. Runtime epsilon converts to fp32 at both source uses, and the output converts back to DX storage. X, DY and DX all use `stride_x_row`, exactly as the kernel does.

The candidate commutes the three natural-number row-base products. It changes no floating operation, reduction order or mask, and no speedup is claimed. `FPEquiv.lean` uses the default fp32 profile and states equality of complete executions, including failure, registers and all memory, with no domain sites or numerical FP atoms.

`Correct.lean` states independent real correctness for both versions: `dy*rstd - sum(dy*x)/(variance+eps)*rstd*x`, with `variance=sum(x*x)` over the active row and `rstd=1/sqrt(variance+eps)`. The complete flat-memory contract includes successful execution, exact bounds for both masked input windows and the output window, and preservation of every other memory cell. New full-row specifications require `0 < BLOCK_N` and `N ≤ BLOCK_N`; historical contracts keep their original statements, including empty-tile behavior. The real semantics use total square root/division and erase floating casts, so these theorems do not assert IEEE accuracy.

`Proofs/Execution.lean` separately proves successful numerical typed runs of both kernels with fp16 X/DX and bf16 DY. A nonzero row, stride 3, one active lane and one padded lane produce DX=7/32 from X=3, DY=2 and epsilon 7. Masked input sentinels do not enter the reductions, and the adjacent output remains 7. The witness backend uses exact real arithmetic and identity casts; it is not an IEEE rounding model.

The boundary is one kernel invocation with nonnegative natural strides. Host reshape/contiguity handling, allocation, launch grid and maximum fused-size checks remain outside the proof. In particular, the source uses X's row stride for DY even when the wrapper supplies a differently laid-out gradient. Flat allocations must be disjoint and cover the active addresses, with the standard zero undefined-memory launch state.

`Proofs/Real.lean` preserves the legacy full execution, reduction, safety and frame proofs. The new `Math`, `Source`, `Candidate`, `SourceChecks` and `FP` helpers isolate the independent formula, typed projection, full execution transport, precision checks and exact address rewrite.

The public `Correct` and `FPEquiv` entry dependency build passes (3445 jobs). Five public contracts are registered with validated source fingerprints and `test-gap` status. Independent comparator replay remains pending.
