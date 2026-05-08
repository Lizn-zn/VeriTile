# VeriTile — Project Plan

**English** | [中文](PLAN_zh.md)

VeriTile is a long-running project to bring real Triton kernel verification
into Lean 4. Not toys, not a hand-picked theorem set tuned to a deadline — the
goal is to let a Triton engineer take the production `.py` kernels they ship
every day (forward + backward + concurrency primitives + production-scale
layout / masking / autograd paths), put them through VeriTile's embedded DSL
with minimal modification, and get machine-checkable correctness guarantees.

This document supersedes the original 2026-04-26 brainstorm "PLDI 7-10 month
program plan" (redirected 2026-05-05; see §Decision log entries 8-9).

## North star

Real Triton verification requires four axes advancing together:

1. **DSL surface close to real Triton** — accept the ops, control flow,
   memory primitives, and concurrency primitives that production `.py`
   kernels actually use, not just the minimal subset needed for attention
   forward
2. **Algorithm-layer proofs cover production kernels** — full FA-1 forward
   + backward, FA-2, the fused-norm family, grouped GEMM, Mamba SSM, RoPE,
   GQA / MQA / MLA, etc., not 8 hand-picked theorems
3. **Triton-user friendliness** — `.py` paste-in delta drops from "rewrite
   into the embedding" toward "annotate spec / lift quantization points";
   ultimate goal is near-zero changes
4. **Real horizontal infra** — Algorithm/Compute split, Memory frame,
   Concurrency trace/refinement, ND-general framework (no ad-hoc
   dimension-specific shortcuts) all production-grade

This is an open roadmap, **not "deliver N theorems and stop"**. The §Status,
§In progress, and §Roadmap sections below organize work along these four axes.
The live issue-tracked roadmap is GitHub issue #91; this document records the
architecture, status, and decision log.

## Verification architecture (permanent, locked since v0.2)

VeriTile splits verification into **two independent and complementary** layers:

```
Algorithm layer (Lean proof)         Compute-gap layer (external testing)
────────────────────────────         ───────────────────────────────────
ProjectedCorrect ck spec             GapPolicy.require contract
  := ck.toAlgorithm? = ok ∧            := Python check of the
     Kernel.Correct ak spec               ComputeKernel behavior against
                                          the projected AlgKernel behavior
  Real / Int / Nat semantics.            within the contract tolerance.
  No IEEE rounding, no NaN,              Recorded in Lean as
  no fp exception flags.                 ExternalChecked contract.
```

The two layers are **not connected by an internal Lean theorem.** The numeric
gap between compute-facing syntax and projected algorithm syntax is empirical
(testing), by design. A VeriTile-certified compute theorem means:
> The projected algorithm structure is Lean-proved correct, **and** when
> `gap := .require contract` is used, the compute-to-algorithm gap named by the
> contract has been externally validated.

Both layers are required for full certification.

### Permanence

IEEE-754 formalization (NaN propagation, denormals, rounding modes, hardware
dot precision, fast-math, exception flags) is **permanently out of Lean's
proof scope**. This is not a deferral — there is no planned bridge theorem
connecting compute execution to Lean-internal algorithm execution.
Required compute gaps are and remain test-backed.

### Algorithm-layer details

1. The DSL accepts user-facing float kernels with `.fp32`/`.fp16`/`.bf16`
   dtype annotations as usual. Bit-level constants (e.g. a sentinel `1.0`
   written as `tl.bitcast(0x3f800000, tl.float32)`) are accepted via a
   single computable decoder; only finite-normal binary32 patterns are
   supported.
2. Algorithmic correctness is proved over algorithm-layer kernels obtained
   via `Kernel.eraseDType` (fp* → real, intN → int, uintN → nat).
3. The formal Lean layer uses `ComputeKernel.ComputeCorrect` /
   `ComputeKernel.ComputeRefine` with `gap := .ignore` by default, or
   `gap := .require contract` when an external compute-gap check should be
   recorded. Existing `Kernel.Correct` proofs continue to work over
   algorithm-side kernels.

### Compute-gap details (testing)

1. `ComputeKernel` faithfully preserves the user-written compute-facing AST
   (including `ComputeOp.bitcast`, fp/int width spellings, etc.).
2. The Python gap checker (separate workstream) exports or mirrors the
   `ComputeKernel` and its projected `AlgKernel`, generates sample inputs, and
   validates the contract relation such as exact equality or
   `|compute_output - algorithm_output| < epsilon`.
3. Epsilon policy is per-op / per-kernel and appears in
   `ComputeGapContract`; Lean records only `ExternalChecked contract`.

### User perspective

- Algorithm-structure bug: caught by `ComputeKernel.ComputeCorrect` /
  `ComputeKernel.ComputeRefine` (the Lean proof fails after projection to the
  internal `ProjectedCorrect` / `ProjectedRefine` obligation).
- IEEE-specific edge case bug (NaN, overflow, denormal handling, etc.):
  caught by the required compute-gap checker. **This layer has no internal
  Lean proof obligation by design.**
- "Full compute-facing certificate" means projected Lean proof plus required
  gap contract, when the theorem opts into `gap := .require contract`.

The trusted bridge from Real-algorithm correctness to floating computation
**is the external gap checker**, not a Lean theorem.

## Status (2026-05-05)

### Tier 1 — Loop-free kernel pairs ✅ (`v0.1-tier1`)

- `softmax_kernels_refinement` — naive ↔ numerically stable softmax
- `log_sum_exp_refinement` — direct LSE ↔ shift-trick LSE
- `softmax_reciprocal_refinement` — `y = e/s` element-wise division ↔
  `inv_s = 1/s; y = e * inv_s`
- math lemma `welford_eq_two_pass` (prep for Tier 2 Welford)

### Tier 2 — Streaming reductions ✅ (`v0.2-tier2`)

- `welford_kernels_refinement` — Welford ↔ two-pass variance
- `online_softmax_recurrence_eq_batch` — FlashAttention algorithmic core
- `layernorm_kernels_refinement` — fused single-pass ↔ two-pass LayerNorm
- Operational `forLoop_inv` (the master lemma for every loop kernel proof;
  binds the loop counter to an index register before the body, supports
  nested loops)
- Beyond original scope: Mask + Bool channel (masked load/store,
  `tl.load(p, mask=m, other=o)`, 6 comparison ops, `other=None`
  nondeterminism oracle); Typed Tile refactor (`Op : TileDType → TileShape →
  Type` with end-to-end typed `evalOp`/`stepStmt`, `RegFile` indexed by
  `(dtype, shape, name)`); `WithBot ℝ` carrier (`Op.negInf` lowers to true
  `⊥` rather than `-1e38` stand-in)

### Tier 3-A — FA-1 forward full coverage ✅ (`v0.3-tier3a`)

- `fa1_forward_correct` (non-causal, single-block reasoning)
- `fa1_forward_correct_strided` (arbitrary stride layout)
- `fa1_forward_correct_strided_causal`
- `fa1_forward_correct_4D` / `fa1_forward_correct_4D_causal` (`batch ×
  heads × seq × dim`)
- `Boundary.lean`: 8 boundary-mask / boundaryD variants (partial KV
  blocks, partial head dims)
- `ScoreVariants.lean`: softcap / ALiBi / sliding-window / 3-way
  composition, 4 forward variant theorems
- ~16k lines of FA-1 forward proof (Core + Boundary + ScoreVariants +
  Common)

### Horizontal infra ✅ (beyond original PLAN scope, in parallel with Tiers)

- **Two-layer architecture**: `ComputeKernel` / `AlgorithmKernel` bridged by
  `eraseDType`; `Kernel.Correct` / `Kernel.Refine` proven on the algorithm
  side; `ProjectedCorrect` / `ProjectedRefine` as the internal projection
  obligations; `ComputeKernel.ComputeCorrect` /
  `ComputeKernel.ComputeRefine` as the user-facing entries; bitcast routed
  through compute projection
- **Float dtype erasure**: `.fp32` / `.fp16` / `.bf16` → `ℝ`; bitcast via a
  single computable decoder; invalid bitcast bits rejected at macro
  expansion
- **Integer dtypes**: signed Int abstraction (typed integer load, Nat
  store coverage, Nat bitwise surface)
- **Memory subsystem**: Bounds (read out-of-bounds safety), Footprint
  (read/write region extraction), Frame (disjoint frame composition,
  unrelated-frame preservation), block-pointer boundary semantics, proof
  memory API via `readMem`
- **Concurrency framework**: Trace vocabulary, Refinement, async
  sequentialization contract, atomic-add proof surface, async copy / async
  wait / debug-barrier failure markers, producer-consumer discipline,
  projection-failure correctness lemmas, unsupported-atomic failure
  markers, generalized effect projection
- **ND-general path**: ND grid launch theorem, grid composition, generic
  `Fin shape.length` axis, `expand_dims` / `static_range` / shape-view
  surface
- **DSL surface expansion**: prefix scan, arg / sort, shape view, unary
  math, bitcast, expand_dims, static_range, minimum / maximum / abs,
  Boolean operators and comparisons
- **DSL modularization**: Triton core / DSL / semantics / float / memory
  typing module split, optional Triton well-formedness checker

### Numbers

- 111 `.lean` files under `VeriTile/` (~48.2k lines); 128 `.lean` files including `bench/` (~50.2k lines)
- Whole library 0 sorry
- `lake build` clean

## In progress

### FA-1 backward

The original PLAN listed backward as P3+ "never"; the redirect moves it
in-scope (see §Decision log entry 9).

`VeriTile/Examples/FlashAttention1/Backward.lean`:

- `streamingLSE_eq_lseReal` — the forward LSE store using `m_i + tl.log(l_i)`
  equals unshifted log-sum-exp (the key bridge for backward to reconstruct P)
- `attentionBackwardReal_eq_reverseMode` — closed-form reverse-mode FA-1
  backward
- `attentionBackwardRealMasked_allVisible` — generic mask-aware Real backward
  spec collapses to the existing non-masked spec under the all-visible mask
- `dQBlockContributionMasked_sum_eq_attentionBackwardRealMasked` — masked
  multi-block `dQ` contributions sum to the mask-aware backward spec
- `dQBlockContributionCausal_sum_eq_attentionBackwardRealCausal` — causal
  specialization of the masked multi-block `dQ` bridge
- `gridLaunchedAtomic_masked_dQ_correct` and
  `gridLaunchedAtomic_causal_dQ_correct` — launcher-facing masked/causal
  atomic `dQ` composition surfaces
- `fa1BackwardAtomicDQCausalKernel` — DSL kernel surface for causal
  block-partitioned backward with atomic `dQ`; trace extraction remains next
- `fa1BackwardAtomicDQCausalPreAtomicState` — pre-atomic boundary for the
  causal kernel prefix, matching the non-causal proof split
- `FA1BackwardAtomicDQCausalPreAtomicFacts` — proof target bundling the causal
  prefix obligations for `dQ_part`, `dK_block`, and `dV_block`
- `fa1BackwardAtomicDQCausalKernel_statefulTrace_blockContribution` — causal
  atomic `dQ` trace recomposition once prefix facts and tail execution are
  available
- `causalBackward_tile_bridges_complete` — causal `dQ/dK/dV` tile-dot bridge
  bundle for the prefix proof
- `causalBackward_block_tile_bridges_complete` — block-local causal
  `dQ_part`/`dK_block`/`dV_block` tile-dot bridge bundle for the atomic prefix
  proof
- Math-layer tile bridges: probability, `dP = dO · Vᵀ`, row correction
  `D_i = Σⱼ P_ij · dP_ij`, `dS = P * (dP - corr[:, None])`,
  `dV = Pᵀ · dO`, `dQ = dS · K · scale`, `dK = dSᵀ · Q · scale`
- `softmax_jvp_identity` — softmax JVP form
- `strippedBackward_tile_bridges_complete` — bundled math-layer surface
- `fa1BackwardStrippedKernel_correct` — stripped main theorem (no mask, no
  multi-block, single program-id)

Next: prove the causal kernel prefix facts against the concrete DSL prefix, then
connect that trace extraction to the launcher-facing surfaces above. Multi-block
atomic dQ composition itself is available through the grid-launched surface.

## Roadmap (priority-ordered, no fixed time windows)

### Near-term — Tier 3-A wrap-up and backward stage 1

- Maintain closed FA-1 backward stripped theorem
- Maintain `v0.3-tier3a` tag for forward full coverage
- Prove masked/causal FA-1 backward kernel trace extraction against the
  launcher-facing masked/causal `dQ` surfaces
- Wire FA-1 backward to multi-block (after mid-term multi-block semantics)

### Mid-term — Tier 3-B FA-2 forward + headline corollary

*Semantic extensions:*

- `multiBlockExec : Kernel → InitMem → Grid → FinalMem` model — runs every
  program_id (3D grid: `batch × heads × Q-blocks`), composing each
  program's writes
- Stride / layout model (per program_id row-major contiguous; no general
  layout system)
- Disjoint-writes lemma — for atomic-free pure-forward kernels, each
  program's output region is disjoint, so per-program local correctness
  composes

*Kernel and theorems:*

- FA-2 kernel embedded via `triton { ... }` (multi-axis program_id, delayed
  rescale path, mask-skip path)
- `fa2_delayed_rescale_sum_eq` and
  `fa2_delayed_rescale_weighted_sum_eq` — denominator/numerator algebra for
  block-local max normalization followed by merged-max rescaling
- `fa2_two_fragment_denominator_merge_eq_flat` and
  `fa2_two_fragment_numerator_merge_eq_flat` — two-fragment merge identities
  for FA-2 partitioned forward
- `fa2_two_fragment_attention_ratio_eq_flat` — pointwise output ratio identity
  for a two-fragment FA-2 merge
- `fa2_masked_sum_eq_zero_of_all_invisible` and
  `fa2_masked_weighted_sum_eq_zero_of_all_invisible` — full-mask block-skip
  denominator/numerator zero-contribution identities
- `fa2ScalarTwoFragmentMergeKernel_correct_view` — executable scalar
  merge-stage kernel surface for delayed-rescale fragment merge
- `fa_2_forward_correct` — like FA-1 forward, with: (a) sequence-length
  parallelization, (b) delayed rescaling, (c) full-mask block skip
  - Delayed rescale equivalence: `O_final / l_final` does not depend on
    whether intermediate `O` is rescaled each step (`delayed_rescale_eq`
    helper)
  - Block-skip correctness: full-mask blocks contribute 0 to the recurrence
  - Heavily reuses Tier 3-A single-program invariant tooling
- **Headline corollary `fa1_eq_fa2`** — ~30 lines via spec transitivity
  through `standardAttentionMath`

### Mid-term — Tier 3-C FA backward full coverage

- FA-1 backward + mask + causal + multi-block (after stripped single-block
  closes)
- FA-2 backward (reuses multi-block semantics)
- Headline corollary `fa1_backward_eq_fa2_backward` (dual to forward)

### Mid-term — Tier 4 production kernel batch 2

Ordered by real Triton-engineer usage frequency:

- **Grouped GEMM** — production attention serving, MoE routing, batched
  matmul critical path
- **Mamba / SSM** — state-space model, Triton-implemented production candidate
- **RoPE** — rotary position embedding, attention-serving necessity
- **Fused-norm family** — RMSNorm, QK-norm, LayerNorm with bias
- **Multi-head attention variants** — GQA, MQA, MLA (MoE-aware attention)

Every production kernel needs forward + backward + Lean equivalence to its
mathematical spec.

### Long-term — horizontal infra continuation

- **Concurrency primitives in the proof chain proper** — currently failure
  markers + projection boundaries. Next: atomic-add correctness main
  theorem, async-copy serialization equivalence theorem, end-to-end
  refinement of producer-consumer patterns
- **`.py` paste-in surface** — Python lifter prototype (the original P3+
  item, brought into the mid-/long-term roadmap); macro acceptance moves
  from "embedding-style" toward "real `.py` with near-zero changes"
- **ND-general closure** — eliminate residual 1D / 2D-specific paths;
  every op flows through the generic `Fin shape.length` axis
- **Effect framework polish** — compute-effect markerization, projection-
  failure correctness lemmas elevated into a full effect-aware proof stack
- **Block-pointer full coverage** — currently boundary semantics + partial
  layout; extend to stride manipulation, multi-D nested block pointers

## Explicitly outside Lean proof chain

After the v0.2 two-layer architecture lock-in, the following are
permanently outside Lean proof scope:

- **IEEE-754 bit-level behavior** — NaN propagation, denormals, rounding
  modes, hardware dot precision, fast-math, exception flags. Compute layer
  covers these via differential testing
- **Triton autotune configs / performance cost models** — autotune choice
  (`BLOCK_M` / `BLOCK_N` / `num_warps`) does not affect algorithmic
  equivalence
- **WGMMA / Hopper-specific instruction scheduling** — algorithmic
  equivalence is independent; pure performance layer
- **PTX / SASS backend faithfulness** — assumes the Triton compiler
  faithfully implements Triton semantics

Note: **Backward**, **Python lifter**, and **concurrency-primitive proofs**
have **moved out** of the original PLAN's P3+ list and into the roadmap
(see §Decision log entry 9).

## Cross-cutting

### Risk register

| Risk | Trigger signal | Mitigation |
|---|---|---|
| FA-1 backward mask or multi-block wiring stalls | proof split stalls ≥ 2 weeks | `/lean4:autoprove --deep=stuck`; if that fails, split into smaller stripped sublemmas |
| `multiBlockExec` formal model more complex than expected | drafting the disjoint-writes invariant | restrict to "per program_id row-major contiguous"; no general layout system |
| FA-2 multi-block abstraction more complex than expected | drafting `multiBlockExec` reveals hidden complexity | start with stripped FA-2 (single-program, no mask-skip); full version later |
| `tl.dot` over `Value.tile2D` simp behaves badly | FA proof stuck on simp | custom simp lemmas; or switch to `unfold + induction` style |
| Concurrency-primitive proofs introduce too much complexity | atomic-add main theorem makes no progress for ≥ 3 weeks | fall back to a sequential-consistency single-layer spec (no race model); full weak-memory left to long-term |
| Production-kernel batch 2 scope sprawl | Tier 4 list keeps growing | tackle 1-2 kernels at a time to closure + tag, not many lines in parallel |
| Python lifter and embedding macro acceptance mismatch | lifter output fails to type-check | extend macro acceptance first; lifter follows; not the reverse |
| `/lean4:autoprove` cannot handle long proofs (~300+ lines) | held-out close rate < 30% | cut into plugin-closable sublemmas + manual composition; benchmark report records human/machine split |
| Upstream `lean4` plugin breaks / API changes | `/lean4:autoprove` errors or output format changes | pin a specific plugin version (commit / version recorded in `CONTRIBUTING.md`); fall back to old cached version |
| Float-abstraction bridge feels too strong to external reviewers | review / audit demands IEEE-754 proof | two-layer architecture permanently locked: VeriTile proves the Real abstraction, exposes float-facing theorems via erasure, validates representative float behavior with tests |
| Diff-test numerical delta exceeds tolerance (representative kernel) | GPU output vs reference | yellow flag, not a gate — investigate IEEE-754 inherent delta, `.py` implementation bug, mismatched correspondence assertion. Formal proof remains valid |

### External validation

VeriTile keeps two external validation tracks separate:

1. **Compute-gap contracts** — required only when a theorem uses
   `gap := .require contract`. These checks compare the compute-facing
   `ComputeKernel` behavior with the projected `AlgKernel` behavior under the
   contract relation (for example exact equality or an epsilon bound). Lean
   records `ExternalChecked contract`; it does not prove IEEE / bit-level
   compute behavior internally.
2. **Runtime credibility tests** — optional smoke / differential tests that
   compare representative runnable Triton/Python implementations against
   PyTorch / `flash-attn` references. These tests help show that the embedded
   DSL corresponds to realistic runnable kernels, but they are not the Lean
   proof chain and do not replace `GapPolicy` contracts.

**What runtime credibility tests cannot solve** — hand-written `.py`
diff-testing does not validate the embedded operational semantics itself, nor
does it close the gap between the embedded AST and real `.py` source (that
requires the Python lifter; see §Long-term).

**Current runtime representative set** (grows along the roadmap):

- 1-2 Tier 1+2 kernels (e.g. `naiveSoftmaxKernel`, one streaming kernel)
- FA-1 forward (sample variants)
- Mid-term: FA-2 forward, FA-1 backward, FA-1 ↔ FA-2 cross-check
- Long-term: Tier 4 production kernel samples

**Tolerances**: Tier 1+2 ≤ 1e-5 element-wise absolute delta; FA-class
≤ 1e-3 (the order of `flash_attn` reference vs PyTorch).

**Runtime failure handling**: a runtime diff-test failure is a yellow flag,
not a gate — investigate IEEE-754 inherent delta (acceptable, document), `.py`
implementation bug (fix), mismatched correspondence assertion (cross-check
again). It does not invalidate the Lean theorem over the embedded semantics.
A required compute-gap contract failure is different: the theorem using
`gap := .require contract` should not be treated as fully certified until the
external certificate is regenerated and passes.

### LLM benchmark protocol

To make the close-rate metric reproducible:

- **Held-out set**: each new Tier reserves 5-10 lemmas in
  `bench/llm_eval/<tier>/` independent file structure; the main library
  does not depend on held-out files. **When to lock down: at the *start*
  of each Tier, *before* tuning that Tier's LLM tooling** — to prevent
  "tuning the tool on the eval"
- **Allowed context per held-out theorem**: the held-out file's imports +
  all theorems already proven in the file (before the held-out point) +
  Mathlib lemma name lookup + the tool's built-in retrieval (if any). The
  model sees the theorem statement, section context, and explicitly
  retrieved lemmas. **Not allowed**: rest of the project library (beyond
  imports), the held-out lemma's own proof, any human hint targeting the
  held-out lemma
- **Forbidden during eval**: human prompt iteration, human lemma-name
  hints, fine-tuning on the held-out lemma, editing LLM output and
  re-running. The eval is hands-off: tool runs end-to-end, success or
  failure, one data point
- **Trial count**: N=5 independent runs per held-out lemma with different
  random seeds. Close rate = (successes among (lemma, seed) pairs) /
  (5 × |held-out set|)
- **Cost report**: total API calls, tokens, wall-clock, dollar cost per
  Tier; includes failure retries
- **Cross-Tier prompt iteration is allowed**: after a Tier's eval
  completes, the prompt template may change; the modified tool is used on
  the *next Tier's* freshly-locked-down held-out set

### Open-source cadence

- Main repo: `github.com/Lizn-zn/VeriTile` (already public)
- Main branch always builds
- Tag on Tier closure (already: `v0.1-tier1`, `v0.2-tier2`; pending:
  `v0.3-tier3a`, plus more as the roadmap grows)
- Each release ships with release notes (new theorems, new semantics,
  benchmark data, scope changes)
- Bilingual README (English + Chinese) maintained
- `CONTRIBUTING.md` with a kernel-pair tutorial (using Tier 1
  `log_sum_exp` as worked example)
- `scripts/prove.sh` lives in-repo (not separately packaged); benchmark
  eval results ship with the project. The underlying `lean4` Claude Code
  plugin is upstream and not vendored — `CONTRIBUTING.md` records the
  pinned version

## Decision log

1. **2026-04-26 brainstorm** — Bet (b) "Verified kernel rewrites" + (e)
   "tooling usable", mix α (attention-headlined), Tier 3 framing T3-D
   (T3-A + T3-B both), sequencing approach 2 (vertical slices). See the
   original brainstorm record for details
2. **2026-04-26** — LLM tooling strategy: wrap the existing `lean4` Claude
   Code plugin (`/lean4:autoprove`) with a thin `scripts/prove.sh`, rather
   than build a Python proof tool from scratch. Reason: the plugin already
   provides LSP integration, multi-cycle iteration, deep-mode escalation,
   tactic cascade, and repair patterns — reinventing is unjustified
3. **2026-04-26** — Differential testing: a non-formal validation
   artifact, not a phase gate, not part of the trusted proof chain.
   Maintained only on a selected representative kernel set (~3-4 kernels
   total), as a credibility supplement
4. **2026-05-04** — Verification architecture permanently locked:
   Algorithm layer (Lean proof) + Compute layer (external testing); the
   two are not connected by a Lean theorem; the bridge is empirical
   (testing). IEEE-754 formalization is permanently outside Lean proof
   scope
5. **2026-05-04** — Memory subsystem landed: Bounds, Footprint, Frame,
   disjoint composition — prerequisite for Tier 3 multi-block
6. **2026-05-04** — Concurrency framework landed (failure markers +
   projection boundaries): scaffolding for long-term atomic / async main
   theorems, but not on the main proof chain right now
7. **2026-05-04** — Float dtype two-layer architecture landed:
   `ComputeKernel` / `AlgorithmKernel` bridged by `eraseDType`,
   float-facing theorems exposed via erasure
8. **2026-05-05** — **PLAN redirect.** From "7-10 month PLDI program plan
   + 8 hand-picked main correctness theorems + headline corollary" to "an
   open roadmap real-Triton-verification project."

   Reason:
   - The original PLAN's 8-main-theorem scope is in fact only the forward
     subset of Triton verification; production kernels real Triton
     engineers face (backward, concurrency, multi-variant attention,
     grouped GEMM, SSM) cover much more
   - Submission-driven deadline / scope-shrink rules (Gate A→B, B→C, C→D
     fallbacks to OOPSLA, etc.) do not match the project's actual
     long-term direction
   - Horizontal infra investment (float erasure two-layer, memory frame,
     concurrency framework, ND launch, score variants) already far
     exceeds the supporting work listed in the original PLAN, reflecting
     "real Triton verification" rather than "toy"
   - The original PLAN's "P3+ never" backward / Python lifter /
     concurrency primitives have already entered or are about to enter
     in-scope in actual project rhythm

   Shape of changes:
   - **Removed**: 7-10 month time window, Phase A/B/C/D structure, Gate
     A→B/B→C/C→D decision rules, PLDI / OOPSLA / ICSE submission framing,
     `v1.0-pldi` target tag
   - **Kept**: verification architecture (Algorithm/Compute split),
     differential testing, LLM benchmark protocol, decision log,
     open-source cadence
   - **Reshaped**: scope from "8 main theorems" to "Done + In progress +
     Roadmap" open layering

9. **2026-05-05** — **Backward pass, Python lifter, and concurrency-
   primitive proofs moved out of P3+ and into in-scope.**

   - **FA-1 backward** entered the near-term roadmap and now has no sorry
     debt in the library
   - **Python lifter** moved from P3+ to mid/long-term — Triton-user
     friendliness requires a paste-in surface, not just documentation
     correspondence
   - **Concurrency primitives** are currently failure markers + projection
     boundaries (already built); next: atomic-add correctness, async-copy
     serialization equivalence, etc., as main theorems

   Reason: these three are the dividing line between "real Triton
   verification" and "toy Triton verification". Triton engineers ship
   backward every day; production kernels heavily use atomic / async; a
   DSL that does not accept paste-in has no users.

## Implementation plans

Active execution is tracked in pinned roadmap issue #91 (layered) and the
linked per-task issues. Long-form design documents that survive an
implementation phase are filed under [`documents/`](./documents); closed-phase
design notes live in [`documents/archive/`](./documents/archive/).
