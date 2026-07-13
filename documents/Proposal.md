# VeriTile: Verifiable Kernel Optimization

**Living proposal — skeleton.** This is the maintained academic story spine
for VeriTile v1, *not* a finished paper. It is updated continuously as work
progresses (academic convention: motivation → running example → high-level
idea → plan; details and results are added on demand). Companion slide
outline: [ProposalSlides.md](./ProposalSlides.md).

Charted by the wayfinder map
[VeriTile v1 — verifiable kernel optimization](https://github.com/Lizn-zn/VeriTile/issues/458);
skeleton drafted for
[Draft the maintained proposal + slides skeleton](https://github.com/Lizn-zn/VeriTile/issues/466).
`TBD` blocks mark content waiting on open map tickets — each names the
ticket that will fill it.

> **Supersession note.** The 2026-04 proposal
> ([archive/Proposal.md](./archive/Proposal.md)) pitched LLM-driven
> optimization + Lean equivalence proofs. This proposal reframes around
> *verifiable code optimization* — a verifier core + harness over user-level
> Triton **and Tilelang** code — per the locked v1 destination. Salvageable
> material (related-work pointers, trust-boundary diagram) is flagged where
> relevant.

---

## 1. Motivation

*The pain: optimizing a GPU kernel is easy to get wrong and expensive to
check.*

- Standard practice for kernel optimization is **golden-vs-optimized
  differential testing**: run the reference and the rewrite on sampled
  inputs, diff the outputs. Each iteration costs **hours to days of GPU
  time** for production-scale kernels.
  - `TBD:` quantified pain — GPU-hour figures, incident anecdotes,
    developer-survey citations. *(Fills from the continuous gap analysis +
    evaluation work.)*
- When the diff **fails, it does not localize**: a mismatched output tensor
  says nothing about which rewrite step, which block, or which boundary
  condition introduced the bug.
- When the diff **passes, it proves nothing**: testing cannot show the
  *absence* of bugs — masking/boundary paths, rare interleavings in
  async/overlapped kernels, and shape corners escape any finite sample.
- **Reframing: verifiable code optimization.** Replace "test the rewrite"
  with "verify the rewrite": a **verifier core** (Lean 4) that *proves* the
  optimized kernel consistent with the golden reference, wrapped in a
  **harness** that drives the golden-vs-optimized workflow and localizes
  failures when the proof does not go through.
- Non-goal honesty up front: the proof layer works over real-valued
  algorithm semantics; floating-point deviation is handled separately
  (§3.1) — we *warn* about FP-dangerous patterns rather than bound deltas.

## 2. Running examples

### 2.1 Sync capability: FlashAttention

The canonical hand-rewrite: attention recomputed block-wise with online
softmax so the `N × N` score matrix never materializes — semantically
equivalent in ℝ, structurally unrecognizable, and historically bug-prone.

- Already in-repo (the capability evidence): FA-1 forward fully proved
  (non-causal / causal / strided / 4D / boundary / score variants, ~16k
  lines), FA-1 + FA-2 backward, and the headline equivalences
  `fa1_eq_fa2_two_block_forward4D` and `fa1_backward_eq_fa2_backward(_4D)`.
  See [PLAN.md](../PLAN.md) §Status.
- The proposal walks this example end-to-end: naive attention (golden) →
  FlashAttention rewrite → what the verifier proves → what the harness
  reports on an injected bug.
- `TBD:` choose the exact narrative slice (FA-1 fwd single-block is the
  likely candidate — small enough to show on a slide).

### 2.2 Async/overlap capability: TileScale shift + AllGather-GEMM overlap

Selected by
[Select minimal async/parallel running examples](https://github.com/Lizn-zn/VeriTile/issues/460)
(full feasibility map: [AsyncRunningExamples.md](./AsyncRunningExamples.md)):

- **Example A — TileScale `simple_shift`** (ring `put`): the floor — the
  smallest kernel VeriTile cannot state today; the clean unit test for the
  multi-rank world layer + peer addressing.
- **Example B — TileScale `allgather_gemm_overlapped`**: the canonical
  compute–communication overlap — the property-P3 payload (a GEMM consuming
  tiles as remote data arrives).
- Kept as later targets, not running examples: DeepEP dispatch/combine
  (eventual async/distributed *validation* case) and Triton-distributed
  megakernels (scale stress test).
- The slot's job in the story: show the property class testing is *worst*
  at (interleaving-dependent bugs) and that the verifier covers it
  (property P3, §3.2).

## 3. High-level idea

### 3.1 Two-layer verification

The load-bearing design decision (locked since v0.2, see
[PLAN.md](../PLAN.md) §Verification architecture):

- **Algorithm layer (Lean proof):** proofs run over the erased ℝ/ℤ/ℕ view
  of the kernel — no IEEE rounding, NaN, or denormals.
- **Compute gap (external):** the FP-vs-ℝ difference is checked outside
  Lean (test-backed contracts); it is *never* bridged by a Lean theorem.
  IEEE-754 is permanently external by design.
- The `#447` rounding model captures *where* rounding events occur
  (structure, inside Lean) while magnitudes stay external — the substrate
  for property P4 below.

### 3.2 Four properties

v1 proves four equivalence properties over **user-level** kernel code:

| # | Property | Status today |
|---|---|---|
| P1 | Single-kernel correctness vs. a math spec | Largely exists (`ComputeCorrect.Realizes`, ~151 proven kernels, FA-1 fwd) |
| P2 | Fusion correctness of multiple kernels | Defined ([FusionCorrectness.md](./FusionCorrectness.md)): fused kernel refines the stage pipeline (`seqCompose`) on declared outputs at the ℝ layer, via the thin `ComputeRefine.FusionCorrect` wrapper; running example SwiGLU (+ FusedSiLU N-ary retrofit). Lean surface + example proofs pending (fog opened) |
| P3 | Parallel/async correctness | Abstract-only (`RefinesSequential`); operational interleaving is the research-grade gap ([#409](https://github.com/Lizn-zn/VeriTile/issues/409)). Running examples selected ([AsyncRunningExamples.md](./AsyncRunningExamples.md)); modeling approach `TBD` — [Async modeling: Lean state-machine vs TLA](https://github.com/Lizn-zn/VeriTile/issues/459) |
| P4 | FP-precision **warning patterns** (not tight bounds) | Rounding model exists (`#447`); pattern catalog designed ([FpWarningCatalog.md](./FpWarningCatalog.md)) — a static diff over rounding-event structure; checker implementation pending |

### 3.3 DSL-agnostic core, two frontends

- One **neutral core** — a single typed IR + operational semantics that all
  frontends lower onto (union policy: one closed operator type; frontends
  contribute syntax + lowering only, no semantic nodes). Vocabulary:
  [CONTEXT.md](../CONTEXT.md); code map:
  [ArchitectureHandoff.md](./ArchitectureHandoff.md). Phase-1 relocation of
  the neutral foundation out of `Triton/` is done
  ([#472](https://github.com/Lizn-zn/VeriTile/issues/472)).
- **Triton frontend:** exists (the `triton { ... }` embedding).
- **Tilelang frontend:** mapping settled — *Tilelang is a frontend, not a
  second core*: its tile intrinsics lower onto the same algorithm layer,
  with the memory hierarchy demoted to memory-space tags
  ([TilelangMapping.md](./TilelangMapping.md)). Frontend *implementation*
  pending (map fog, graduates next).

### 3.4 The harness

The verifier alone is a library of theorems; the *tool* is the harness:
golden-vs-optimized driver + failure localization/report around the Lean
core.

- `TBD:` component boundary (CLI, golden-runner, verifier-driver,
  localization/report) — [Design the harness boundary](https://github.com/Lizn-zn/VeriTile/issues/464).

## 4. Plan and milestones

Team-agreed ordering: **first model the user code, then support the v1
features.**

1. **M1 — model the user code.** Neutral-core relocation (Phase 1 ✓), the
   Tilelang mapping study (✓), async example selection (✓); remaining: the
   Tilelang frontend implementation and the Phase-2 core extensions, so
   user-level kernels from both DSLs are representable on the core.
2. **M2 — v1 features.** The four properties (P2–P4 to build on their
   design tickets), the Tilelang frontend, the harness.
3. **M3 — validation.** FlashAttention (sync) + the selected async cases;
   fold gap-analysis findings (FP-vs-ℝ, single-vs-multi-GPU/node, Tilelang
   IR-vs-memory-manipulation) back into this proposal.

The live route is the wayfinder map's frontier; this section tracks it at
milestone resolution only.

- Out of scope for v1 (explicit future work): compiler-layer and
  hardware-layer verification; tight FP delta bounds; internal IEEE-754
  formalization.

## 5. Evaluation plan *(stub)*

- `TBD:` case studies (FA end-to-end; async example end-to-end), kernel
  breadth (TritonBench-G ports), harness localization quality on injected
  bugs, proof-effort/automation metrics ([PLAN.md](../PLAN.md) §LLM
  benchmark protocol).

## 6. Related work *(stub)*

- `TBD:` refresh against 2026 literature. Seed pointers from the archived
  proposal: translation validation (Alive2), algorithm-level equivalence
  (ATL), compiler-internal SMT verification (TVM/Z3), LLM kernel-generation
  benchmarks; add Tilelang/TileScale line and GPU weak-memory verification
  for P3.

## 7. Results to date *(living ledger)*

- 2026-05: FA-1/FA-2 forward + backward closed; headline FA1≡FA2
  corollaries ([PLAN.md](../PLAN.md) §Status).
- 2026-07: neutral-core foundation scoped
  ([foundational-library audit](https://github.com/Lizn-zn/VeriTile/issues/465))
  and relocated out of `Triton/`
  ([Phase 1](https://github.com/Lizn-zn/VeriTile/issues/472)); rounding
  model landed (#447/#456); async running examples selected
  ([AsyncRunningExamples.md](./AsyncRunningExamples.md)); FP-warning
  catalog designed ([FpWarningCatalog.md](./FpWarningCatalog.md));
  Tilelang↔core mapping settled
  ([TilelangMapping.md](./TilelangMapping.md)); fusion correctness (P2)
  defined with SwiGLU as running example
  ([FusionCorrectness.md](./FusionCorrectness.md)).

## 8. Venue

SE / Systems framing; **venue TBD** — decided once the skeleton has
accumulated results (tracked in the map's *Not yet specified*).

---

## Changelog

- **2026-07-13** — P2 fusion correctness defined; SwiGLU chosen as running
  example ([FusionCorrectness.md](./FusionCorrectness.md),
  [ticket](https://github.com/Lizn-zn/VeriTile/issues/461)).
- **2026-07-07** — skeleton created
  ([ticket](https://github.com/Lizn-zn/VeriTile/issues/466)).
