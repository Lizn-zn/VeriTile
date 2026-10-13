# TritonBench-G Proof Completion Implementation Plan

> **For agentic workers:** Use subagent-driven-development for disjoint port implementations and independent review. Root owns common infrastructure and inventory integration.

**Goal:** Complete all TritonBench-G source, real correctness and FP equivalence obligations, including the presently unimported directories, with verified source correspondence and explicit trust boundaries.

**Architecture:** Preserve each original Python kernel and its precision, control flow and memory behavior in shared `Kernels.lean`. Prove independent mathematical specifications for original and candidate in `Correct.lean`, and kernel-level floating-point equivalence in `FPEquiv.lean`; keep supporting lemmas in `Proofs/`. Never count a precomputed-input slice, unsupported atomic admission, failed replay or a weaker substitute as completion of a missing full contract.

**Tech Stack:** Lean 4.29, mathlib, VeriTile typed Triton DSL, Python source/inventory checks, official comparator.

## Baseline and scope

- [ ] Rebuild `VeriTile` and validate existing migrated entries. Recheck the stale 59-complete/125-pending record against actual sources and proof results.
- [ ] Inventory all directories, including 11 source-import scaffolds. Retain `verification_round.md` as the per-directory ledger; do not limit the task to files containing `sorry`.
- [ ] Resolve unsupported original statements with explicit counterexamples or necessary, source-justified contracts. Do not silently alter existing statement signatures or substitute another optimization for a pending transformation.

## Initial parallel implementation batch

- [ ] `bench/tritonbench_g/bgmv_expand_slice/`: repair the original output-first FP derivation after dtype-preserving sum/promotion changes. Retain exact typed sources and the independent real proofs; expose unsupported precision admissions truthfully.
- [ ] `bench/tritonbench_g/fused_activation/`: inspect all Python JITs; create shared source/candidate, mathematical real contracts, full FP derivation and local helpers, preserving actual memory effects.
- [ ] `bench/tritonbench_g/relu_triton_kernel/` and `relu_strided_buffer/`: prove both original/candidate contracts for the actual launch branches/loops, masks and frames; use structural FP proofs when no numerical rule is needed.
- [ ] `bench/tritonbench_g/masked_add_cuda/`: retain the in-place/data-dependent mask and original real formula while adding a source candidate and guarded FP proof, then independent real correctness of the candidate.

## Per-port acceptance and remaining batches

For each directory in `verification_round.md`, including all pending rows:

1. Read the actual Python and all existing public contracts; record omissions and numerical/structural obligations separately.
2. Search existing semantic, execution, safety, frame and scalar lemmas before extending the library. Shared extensions belong to root to avoid concurrent edits.
3. Complete source/candidate, both mathematical contracts and the actual intended FP transformation. Keep shapes symbolic and necessary operand/intermediate domains visible. Prove nonempty execution/domain witnesses where conditions might hide vacuity.
4. Compile every changed helper and public entry. Inspect the actual assumption printer and axiom checks. Review source fidelity, statement scope, success, stores and full memory frame independently.
5. Run `bench/audit_trust.sh <port>` on stable snapshots once comparator infrastructure is available. A missing infrastructure prerequisite is pending verification, never a pass.
6. Integrate accepted modules into `lakefile.toml`, refresh source fingerprints and `proof_gap_manifest.tsv` using reviewed statements, and update specification sheets and `verification_round.md` with exact evidence.
7. Continue through elementwise, reductions, normalization, indexing/quantization, matrix and attention/recurrence families, then the source-import scaffolds and their required DSL support. Existing legacy slices remain pending until stronger requested contracts are proved.

## Final completion audit

- [ ] `lake build VeriTile VeriTileFull TritonBenchGVerified TritonBenchSpecExamples` passes.
- [ ] `bench/audit_tritonbench_g.sh --source-only`, coverage fingerprints, proof-gap manifest and source/layout regressions pass without exemptions hiding unfinished proofs.
- [ ] Full corpus trust audit and independent comparator replay pass for current snapshots.
- [ ] Every directory's original/candidate source, independent real formula, successful execution, actual IO/frame, FP precision/domain/assumptions and verification evidence are reviewed against its full required scope.
- [ ] All remaining blockers are genuinely discharged; no historical receipt, passing narrow theorem or manifest label is treated as evidence for unverified code.

The goal remains active until this audit succeeds. Work may continue across turns without shrinking the scope.
