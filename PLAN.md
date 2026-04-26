# VeriTile — Program Plan

**English** | [中文](PLAN_zh.md)

A 7–10 month research program to deliver a "Verified kernel rewrites" paper through 8 main correctness theorems for production Triton kernels — including FlashAttention forward correctness — and the headline corollary `fa1_eq_fa2`.

This document is the program-level plan agreed in the brainstorm of 2026-04-26. Each phase below has a separate implementation plan (to be written via the `writing-plans` flow when the phase begins).

## Goal

**Primary** — submit a paper to PLDI (or OOPSLA-spring as fallback) with the central claim:

> 8 main correctness theorems for production Triton kernels — including FlashAttention forward ≡ standard attention denotation and FA-2 forward ≡ standard attention denotation, yielding the headline corollary FA-1 ↔ FA-2 — formally verified in Lean 4 against an embedded Triton operational semantics, using LLM-assisted proof drafting and GPU differential testing for credibility.

**Secondary** — toolchain (operational semantics, embedded `triton { ... }` DSL, LLM-assisted proof harness, differential testing) usable internally and openable to external contributors.

## Scope: 8 main correctness theorems + helpers + corollary

| Tier | Phase | Main theorems | Type |
|---|---|---|---|
| 1 | A | (#1) `softmax_kernels_refinement` (DONE) · (#2) `log_sum_exp_refinement` · (#3) `softmax_reciprocal_refinement` | kernel ↔ kernel × 3 |
| 2 | B | (#4) `welford_kernels_refinement` · (#5) `online_softmax_recurrence_eq_batch` · (#6) `layernorm_kernels_refinement` | kernel ↔ kernel × 3 |
| 3 | C, D | (#7) `fa_forward_correct` (T3-A, Phase C) · (#8) `fa_2_forward_correct` (T3-B, Phase D) | kernel ↔ math × 2 |

**Plus helpers** (not counted as main theorems):
- Math-only lemma `welford_eq_two_pass` (Phase A, prep for #4)
- Operational helper `forLoop_inv` (Phase B, used by every loop kernel proof)
- Mask-handling helper (`Phase C`; form depends on Phase A scouting outcome — see §Phase C)
- `delayed_rescale_eq` helper (Phase D, used inside #8)

**Plus headline corollary** (derived, not a 9th main theorem):
- **`fa1_eq_fa2 : Y_fa1 = Y_fa2`** — by spec transitivity through `standardAttentionMath` from #7 and #8 (~30 lines). **This is the sentence the paper foregrounds**, but it is *not* a separately-proved theorem; its content is fully discharged by #7 and #8.

### Tier 1 — Loop-free (Phase A, 3 pairs)

1. **`softmax_kernels_refinement`** — DONE — naive ↔ numerically-stable softmax
2. **`log_sum_exp_refinement`** — direct LSE kernel ↔ shift-trick LSE kernel (`logsumexp(x) = m + log(Σ exp(x − m))`, `m = max(x)`)
3. **`softmax_reciprocal_refinement`** — `y = e/s` per-element-divide kernel ↔ `inv_s = 1/s; y = e * inv_s` precomputed-reciprocal kernel (saves N−1 divisions; minor mathematically, demonstrates fused-multiply rewrite pattern)

Plus one math-only lemma `welford_eq_two_pass` (preparation for Phase B's Welford kernel theorem; not counted as a kernel-pair theorem).

### Tier 2 — Streaming reductions (Phase B, 3 pairs)

4. **Welford ↔ two-pass variance** at the kernel level — Welford uses `forLoop` over a tile to maintain `(M, S, n)`; two-pass uses `tl.sum` twice. Lifts the Phase A math lemma to actual kernels.
5. **Online softmax recurrence ≡ batch softmax** — *the FlashAttention algorithmic core, paper centerpiece.* Proves that the streaming `(m_new, l_new) = (max(m_old, max(x_block)), exp(m_old − m_new) · l_old + Σ exp(x_block − m_new))` recurrence produces the same `(m, l)` as the one-shot batch form, by induction over blocks.
6. **Fused single-pass LayerNorm ≡ two-pass LayerNorm** — composes the Welford lemma with the affine `(x − μ)/√(var + ε) · γ + β` transform. Demonstrates lemma reuse across kernel pairs.

### Tier 3 — Production attention (Phases C+D)

7. **`fa_forward_correct`** (T3-A, Phase C) — FlashAttention-1 forward kernel ≡ `standardAttentionMath Q K V causal` denotation. Single-block reasoning: each program_id processes one Q-block × full KV loop. Extends Phase B's `(m, l)` invariant to `(m, l, O)` for the running output accumulator. Output is per-element `(i, d)` of the (S × D) output matrix, *not* a scalar (`observeY2D` reads at `(i, d)`).

8. **`fa_2_forward_correct`** (T3-B, Phase D) — FlashAttention-2 forward kernel ≡ `standardAttentionMath` denotation. Same shape as #7 with three engineering differences: (a) sequence-length parallelism (FA-2 grid is 3D: `batch × heads × Q-blocks`), (b) delayed rescaling of `O`, (c) fully-masked-block skipping. Largely reuses Phase C's `(m, l, O)` invariant tooling on the single-program-id portion; the multi-block coordination is new in Phase D.

**Headline corollary** (derived, not a 9th theorem): `fa1_eq_fa2 : Y_fa1 = Y_fa2`, by spec transitivity through `standardAttentionMath` from #7 and #8 (~30 lines).

## Phase structure (~28–42 weeks total)

### Phase A — Tier 1 + LLM tool MVP + T3 scouting (~6–8 wk)

**Deliverables:**
- 3 Tier 1 kernel-pair theorems closed (#1 already done; close #2 and #3)
- 1 math-only lemma (`welford_eq_two_pass`)
- LLM proof tool MVP: ~200-line Python script, `prove.py file_path theorem_name` interface, prompt → API → write back → `lake env lean` verify → retry-on-error loop, N=5 retries
- T3 scouting document `Notes/T3_scouting.md` (~10–15 pages):
  - FA forward pseudocode + invariant sketch
  - lemma needs for B/C/D semantics extensions
  - explicit risk register for unforeseen formalization gaps
  - feasibility judgment: T3-B in window or fall back to T3-A only?

**Validation:**
- Create a held-out evaluation directory `bench/llm_eval/` separate from the main library; in it, replicate `softmax_naive_correct` with the proof body replaced by `sorry`. The main library copy stays intact and proven; the held-out copy is the LLM benchmark target. LLM MVP must close this held-out copy at close rate ≥ 1/3 across N=5 independent runs (see §LLM benchmark protocol).
- All P1 + Tier 1 builds clean

**Exit gate (`v0.1-tier1`):** all deliverables met; `lake build` clean.

### Phase B — forLoop semantics + Tier 2 (~8–12 wk)

**Deliverables:**

*Operational semantics:*
- Convert `stepStmt` and `stepStmts` to a `mutual` block to support nested loops
- `forLoop` operational semantics with explicit `termination_by (sizeOf body + 1, n − i)` lex measure
- `forLoop_inv` lemma (the workhorse for all Phase B/C/D loop proofs); each iteration binds the loop counter to the index register `idx` *before* stepping the body, so body code that reads `idx` sees the current iteration:

  ```
  ∀ (idx : RegName) (n : Nat) (body : List Stmt) (P : Nat → BlockState → Prop),
    P 0 s_init →
    (∀ i s, P i s →
      ∃ s', stepStmts body (s.setReg idx (Value.scalar (i : ℝ))) = some s'
            ∧ P (i+1) s') →
    ∃ s_final, exec_forLoop idx n body s_init = some s_final ∧ P n s_final
  ```

  The index binding is essential: Welford / online-softmax / FA bodies all reference `idx` as a register (e.g., to compute `offs := pid * BLOCK + idx * STRIDE`). Without binding `idx` before each `stepStmts body` call, the body's data dependency on `i` cannot be expressed in `BlockState`.

*Theorems:*
- 3 Tier 2 kernel-pair theorems (#4, #5, #6)
- The Tier 2 #5 (online softmax recurrence) proof is expected to be ~80–120 lines Lean; this is the paper's central technical contribution.

*LLM tool v0.2:*
- Lemma retrieval: vector / BM25 search over Mathlib + project lemmas; top-k injected into prompt
- Smarter retry: track failed-strategy set, exclude in subsequent prompts
- Structured output parsing: model outputs JSON `{tactic_block, reasoning}`; tactic block syntax-validated before file write

*Differential testing harness:*
- `tools/diff_test/` directory
- For each Tier 1 + Tier 2 kernel, produce a `.py` Triton implementation (sourced from open implementations such as unsloth, vllm, or hand-written following Triton tutorials)
- Compare both `.py` kernels' outputs to PyTorch reference on random inputs; compare to each other
- Tolerance: max element-wise absolute diff ≤ 1e-5

**Validation:**
- LLM v0.2 close rate ≥ 50% on the Phase B held-out set (per §LLM benchmark protocol)
- Differential testing passes for all 6 Tier 1+2 kernels

**Exit gate (`v0.2-tier2`):** all deliverables met; can submit to CGO/CC if Phase C is delayed.

### Phase C — `tl.dot` + masking + Tier 3-A (~8–12 wk)

**Deliverables:**

*Semantics extensions (broader than initially scoped — see Phase A scouting outcome):*
- `Value.tile2D : (m n : Nat) → (Fin m → Fin n → ℝ) → Value` constructor for 2D matrix tiles
- `Op.dot a b`: `(A · B)[i,k] = Σⱼ A[i,j] · B[j,k]` via Mathlib `Finset.sum`
- Comparison ops `Op.lt`, `Op.le` (returning `Value.tile` of 0-1 `ℝ` or boolean tile)
- `Op.where(cond, a, b)` for element-wise mask: branches on a predicate tile, *not* on a sentinel value
- Multi-axis broadcast `Op.broadcastTo`: promote a 1D tile to a 2D shape (e.g., a Q row promoted to a (Q-block × K-block) score tile)
- Causal indexing helpers: index arithmetic for "current Q-block start", "current KV-block start", with proven equivalences to `Fin` arithmetic
- 2D output addressing: `observeY2D : Option BlockState → Fin S → Fin D → Option ℝ` reads at `(s.pid_to_address (i, d))` (1D `observeY` was Tier 1+2 only)
- Single-block reasoning is sufficient for the *algorithmic* content here — multi-program-id grid coordination is Phase D

*Mask handling — open design choice (resolved by Phase A scouting):*

The naive claim "scores set to `-1e38` contribute 0 to softmax weights" is **false in `ℝ`** because `Real.exp(-1e38) ≠ 0`. Two viable resolutions; the choice between them is a Phase A scouting deliverable and the implementation lands in Phase C:

* **Option α — extended reals.** Score values use `WithBot ℝ` (or `EReal`); `Real.exp` extended so `⊥ ↦ 0`. Masked positions in the kernel produce `⊥` *semantically* (the float-level `-1e38` is a *concretization* relevant only to differential testing, not to the formal proof). Mathematically clean; engineering cost is non-trivial because `Op.dot`, `Op.where`, etc. now compose on extended reals.
* **Option β — mask-predicate denotation.** Spec uses `maskedSoftmax (x : Fin n → ℝ) (mask : Fin n → Bool)` directly: `maskedSoftmax x mask i = if mask i then 0 else exp(x i) / Σⱼ if mask j then 0 else exp(x j)`. The kernel embedding must thread an explicit Boolean mask register through `Op.where`, *not* a sentinel float. Avoids extended reals; requires the kernel embedding to expose mask predicates at the AST level.

Phase A scouting evaluates both for FA pseudocode and recommends one; Phase C implements it.

*Theorem:*
- `fa_forward_correct` — full Lean proof, no sorry
- Statement (sketch; assumes mask resolution from Phase A scouting):

  ```
  theorem fa_forward_correct
      (Q K V : Matrix (Fin S) (Fin D) ℝ) (causal : Bool)
      (s : BlockState) (h_inputs : InputsLoaded s Q K V) :
      ∀ (i : Fin S) (d : Fin D),
        observeY2D (exec FAForwardKernel s) i d
          = some ((standardAttentionMath Q K V causal) i d)
  ```

  where `standardAttentionMath Q K V causal : Matrix (Fin S) (Fin D) ℝ` is the math-level attention output (S × D matrix), and `observeY2D` reads cell `(i, d)` of the kernel's `Y` region. The theorem is per-element; row-level / matrix-level forms are corollaries.

- Proof strategy: `forLoop_inv` per Q block with invariant `(m_k, l_k, O_k)` extending Phase B's `(m, l)` recurrence to also track the running output `O_k : Fin D → ℝ`. Per-element correctness derives from `O_k / l_k` at the final iteration.

- **Estimate revised upward**: ~400–600 lines Lean total (semantics extensions ~150 lines, kernel theorem proof ~300 lines, masking-related lemmas ~50–150 lines depending on mask option). The earlier 200–300 line estimate did not account for comparison / broadcast / 2D-layout / causal-indexing semantics.

*LLM tool v0.3:*
- Proof-state interaction: capture intermediate Lean proof state (via Lean MCP server or direct `lean --server` interface), feed back to model on retry, enable step-wise tactic generation
- Required for ~200-line proofs (one-shot mode of v0.2 won't scale)

*Differential testing extends:*
- For each kernel embedded as `triton { ... }`, **maintain a hand-written corresponding Python Triton kernel** in `tools/diff_test/python/`. The two are *not* auto-generated from each other — correspondence is asserted by the human author and reviewed (a future Python lifter is P3+ work; see Out-of-scope).
- Run the hand-written Python kernel on GPU; compare its output to `flash_attn_func` from the `flash-attn` package
- Causal and non-causal both
- Tolerance ≤ 1e-3 (FA reference implementations have roughly this gap to PyTorch reference)
- **Correspondence claim** (must be made explicit in paper): side-by-side appendix table mapping each Lean AST instruction to its Python counterpart, plus a CI check that they share kernel signatures and shapes. Reviewers will ask "is the formal proof about the kernel you actually ran?" — the table + CI is the answer.

**Validation:**
- LLM v0.3 close rate ≥ 30% on the Phase C held-out set (per §LLM benchmark protocol; held-out lemmas are sub-steps of `fa_forward_correct`)
- Differential testing FA forward ↔ `flash_attn_func` passes

**Exit gate (`v0.3-tier3a`):** all deliverables met; can submit to OOPSLA if Phase D is delayed or descoped.

### Phase D — multi-block + FA-1 ↔ FA-2 + paper (~6–10 wk)

**Deliverables:**

*Semantics extensions:*
- `multiBlockExec : Kernel → InitMem → Grid → FinalMem` model — runs all program_ids over a (possibly multi-axis) grid (3D for FA-2: `batch × heads × Q-blocks`), combining the per-program writes
- Stride / layout model — minimal extension to express FA-2's per-`(batch, head, Q-block)` output region (row-major contiguous per program_id; full general layout system is out of scope)
- Disjoint-writes lemma: pure forward kernels (no atomics) with non-overlapping output regions per program_id can be analysed program-locally then composed
- **Estimate revised upward**: ~200–300 lines of semantics extension. The earlier 80-line estimate covered only the bare `multiBlockExec` definition; the disjoint-write proof + multi-axis program_id semantics + 3D-grid region-disjointness invariants are the real engineering. Deliberately avoids concurrency / atomics.

*Kernels and theorems:*
- FA-2 kernel embedded via `triton { ... }` macro (with multi-axis program_id, delayed rescale path, mask-skip path)
- **Main theorem #8 — `fa_2_forward_correct`** — analogous to Phase C's #7 but for FA-2's (a) sequence-length parallelism, (b) delayed rescaling, (c) fully-masked-block skipping:
  - Delayed-rescale equivalence: `O_final / l_final` is the same whether intermediate `O` values are rescaled per-step or only at the end (`delayed_rescale_eq` helper lemma)
  - Block-skipping correctness: fully-masked blocks contribute 0 to the recurrence (uses Phase C's mask-handling lemmas)
  - **Estimate revised upward**: ~250–400 new lines Lean (initial 150–200 estimate did not account for multi-block stride / layout reasoning); large reuse of Phase C single-program invariant tooling
- **Headline corollary `fa1_eq_fa2`** by spec transitivity (~30 lines): both #7 and #8 prove their kernel ≡ `standardAttentionMath`, so the kernels' outputs agree. **This is the paper's headline sentence but not a separately-proved theorem.**

*LLM tool v0.4 (release-ready):*
- Benchmark suite: all theorems from Tiers 1+2+3 packaged as evaluation set; auto-runs measure close rate, retry count, wall-clock, API cost
- Parallel sampling: multiple candidate tactics generated concurrently, first verified one wins
- Cost tracking: aggregate API tokens / cost / time over the entire experiment, report in paper
- CLI polish: `prove --theorem foo --max-retries 5 --strategy retrieval+stepwise` style interface
- Released as standalone package (`lean-llm-prover` PyPI, naming subject to availability check)

*Differential testing finalises:*
- Full differential testing table:
  - 6 Tier 1+2 kernel-pair tests, each kernel ↔ kernel pair tested against PyTorch reference + cross-comparison between the two kernels in the pair
  - FA-1 forward vs `flash_attn_func` / PyTorch reference
  - FA-2 forward vs `flash_attn_func` / PyTorch reference
  - FA-1 vs FA-2 cross-check: same input, observe identical output (to tolerance)

*Paper draft:*

Outline (10 sections):

1. Introduction — motivation: LLM-driven kernel rewriting + verification gap
2. Background — Triton, arm-in-lean, Vero, ATL
3. Embedded Triton subset and operational semantics
4. Algorithmic equivalence framework (refinement pattern, gather/scatter)
5. **Online softmax recurrence** (Phase B centerpiece)
6. **FA forward correctness** (Phase C)
7. **FA-1 ↔ FA-2 verified rewrite** (Phase D core)
8. LLM-assisted proof tooling (Phase A→D evolution + benchmark numbers)
9. Differential testing evaluation
10. Related work + Conclusion

**Validation:**
- All theorems closed, `lake build` clean
- LLM v0.4 benchmark numbers reproducible
- Full differential testing table passes (6 Tier 1+2 kernel pairs + FA-1 + FA-2 + FA-1 vs FA-2)
- Paper draft complete and submitted

**Exit gate (`v1.0-pldi`):** paper submitted; tag and release.

## Cross-cutting

### Risk register

| Risk | Phase | Detection | Mitigation |
|---|---|---|---|
| `forLoop_inv` interface design fails to support nested loops cleanly | B | Phase A scouting flags it during invariant sketches | Fall back to `Nat.iterate` form (less direct, equivalent semantics) |
| Online softmax recurrence Lean formalization stalls | B | ≥ 2 weeks no progress | Engage LLM tool intensively; if still stuck, simplify to stripped version (drop masking from this proof, reintroduce in Phase C) |
| `tl.dot` on `Value.tile2D` has poor `simp` behavior | C | `fa_forward` proof stuck on `simp` | Custom `simp` lemmas (P1 already required this for `Value.bop` on equal-length tiles); switch to `unfold + induction` style if simp won't cooperate |
| Mask sentinel-vs-real-zero gap (`Real.exp(-1e38) ≠ 0`) blocks naive mask formalization | A scouting / C | Phase A pseudocode pass flags it | Choose Option α (extended reals: `WithBot ℝ` / `EReal`) or Option β (mask-predicate denotation); commit during Phase A |
| 2D stride / layout formal model harder than expected | D | When drafting `multiBlockExec` invariants | Restrict to "row-major contiguous per program_id"; document the restriction in paper as an assumed layout |
| FA-2 multi-block abstraction unexpectedly complex | D | Phase C-end draft of `multiBlockExec` exposes hidden complexity | T3-D → T3-A only; downgrade target venue to OOPSLA |
| LLM tool can't handle ~200-line proofs | C | Phase B-end retained-lemma close rate < 30% | Honest report: LLM works on short proofs only; don't force on long ones; tool becomes a "proof-segment helper" rather than "full-proof generator" |
| Differential testing numerical gap > 1e-3 even after kernel implementation tightening | B–D | GPU output vs reference exceeds tolerance | Investigate three causes: IEEE-754 intrinsic difference (acceptable, document); kernel implementation bug (fix); our embedded semantics bug (fix and re-verify proofs) |
| Miss PLDI deadline (~Month 11) | D | Phase C-end timing review | Submit OOPSLA-spring (~Month 14) or ICSE-empirical |

### LLM benchmark protocol

To make close-rate metrics reproducible:

- **Held-out evaluation set**: 5–10 lemmas tagged in advance per Phase, committed to `bench/llm_eval/<phase>/` in a separate file structure; the main library does not depend on the held-out files. **Tagging happens at the *start* of each Phase, before any LLM-tool tuning for that Phase** — this prevents tool-tuning-on-the-eval. The Phase A held-out set is the `softmax_naive_correct` copy (above). Phase B and Phase C sets are picked by the engineer at the start of those Phases from sub-lemmas of #4–#6 and #7 respectively.
- **Allowed context** for the LLM, per held-out theorem: the held-out file's imports + every theorem already proven *in that file* before the held-out one + Mathlib lemma name lookups + retrieval over Mathlib if the tool implements it. The model sees the theorem statement, the section context, and any explicitly retrieved lemmas. **Not** allowed: the rest of the project library beyond imports, the proof of the held-out lemma, or any human hint specific to the held-out lemma.
- **Disallowed during eval**: manual prompt iteration, manual lemma-name hints, fine-tuning on held-out lemmas, any human edit to the LLM output before re-running. Eval is hands-off: tool runs end-to-end, succeeds or fails, that's the data point.
- **Trials**: N=5 independent runs per held-out lemma with random seeds varied. Close rate = (successful runs across all `(lemma, seed)` pairs) / (5 × |held-out set|).
- **Cost reporting**: total API calls, tokens, wall-clock, dollar cost — reported per Phase. Includes failed retries.
- **Allowed prompt iteration between Phases**: the prompt template can be revised after a Phase's eval; the v0.X+1 tool with revised prompts is then evaluated on a *fresh* held-out set tagged at the start of Phase X+1.

### Phase gate decision rules

**Gate A → B** — scouting must show T3-B feasible in the window. If NO, drop to T3-A only (lose #8), expand Tier 2 by 1–2 pairs (candidates: scan reordering, RoPE rearrangement), recalibrate timeline. Total main theorem count becomes 8–9 (3 + 4-or-5 + 1).

**Gate B → C** — online softmax recurrence proof must reach paper-quality. If NO, freeze at `v0.2-tier2`, submit CGO/CC with Tier 1+2, then resume Phase C post-submission as a new project arc.

**Gate C → D** — time-budget check. If Phase C ≤ 10 wk, full Phase D. If 11–14 wk, simplified Phase D (stripped FA-1 ↔ stripped FA-2: drop causal masking, drop block-skipping, simplified work partitioning). If > 14 wk, skip D, submit OOPSLA with T3-A only.

"Stripped" definition: kernels with the algorithmic core preserved (online softmax, `tl.dot`, output accumulator) but engineering details elided (no causal mask, no block-skip optimization, single-block-grid).

**Gate D → submit** — all theorems closed; LLM benchmark data complete; full diff-testing pass; paper draft complete. Submit. Tag `v1.0-pldi`.

### Open-source cadence

- Repo: `github.com/Lizn-zn/VeriTile` (already public)
- Main branch always builds; tag releases at each phase exit:
  - `v0.1-tier1`, `v0.2-tier2`, `v0.3-tier3a`, `v1.0-pldi`
- Each release includes release notes (new theorems, new semantics, benchmark numbers)
- README maintained bilingually (English + 中文)
- `CONTRIBUTING.md` with tutorial for adding a new kernel-pair equivalence (using Tier 1 log-sum-exp as worked example)
- LLM tool released as standalone PyPI package at Phase D end

### Out of scope (P3+, explicit non-goals)

To prevent scope creep and reviewer expectation mismatch:

- **Python lifter** (`.py` Triton → embedded `triton { ... }`): kernels are hand-embedded for this paper. Lifter is a follow-up empirical paper.
- **IEEE-754 fidelity**: floating-point modelled in `ℝ`. Differential testing is approximate, not bit-level. The kernel-level use of finite mask sentinels (e.g., `-1e38`) is a *concretization* relevant only to the runnable Python kernel; the formal proof uses extended reals or mask-predicate denotation per the Phase A choice.
- **Multi-stream / async copy**: not needed for FA-1/FA-2 forward.
- **Backward pass (FA-backward)**: forward only in this paper. FA-1 vs FA-2 backward is a follow-up.
- **Triton autotune configurations**: orthogonal to algorithmic equivalence.
- **WGMMA / Hopper-specific ops**: FA-1/FA-2 forward do not strictly depend on these.
- **Concurrency / atomics**: forward kernels are pure; atomic semantics deferred.

### Timeline (rough Gantt)

```
Month            1   2   3   4   5   6   7   8   9   10
Phase A        [====]
Phase B            [========]
Phase C                     [========]
Phase D                              [======]
LLM tool dev   [=============================]
DiffTest infra     [=========================]
Paper writing                            [====]

PLDI deadline (primary)                            ▲ (~Month 11)
OOPSLA-spring (fallback)                              ▲ (~Month 14)
```

## Decision log

Key decisions made during the brainstorm of 2026-04-26:

1. **Bet selection** — chose (b) "Verified kernel rewrites" + (e) "tools usable" over (a) Vero-for-Triton, (c) semantics-only paper, (d) bug study. Reason: highest probability of a tier-1 venue paper within the window, given current artifact state (P1 sorry-free, no LLM tool yet, no lifter).
2. **Family selection** — Mix α (Attention-headlined) over β (norm-fusion-heavy), γ (matmul-tiling). Reason: FlashAttention's online recurrence has genuine mathematical depth that differentiates from ATL/KaTen/TVM compiler-rewrite verification.
3. **Tier 3 framing** — T3-D (both T3-A and T3-B) over A-only / B-only / stripped-only. Reason: user committed to ambitious scope; resources flexible; T3-A serves as Phase C exit, T3-B as Phase D capstone, with descoping rules per Gate C → D.
4. **Sequencing** — Approach 2 (vertical slice) over 1 (bottom-up) and 3 (tracer bullet). Reason: every phase produces a publishable artifact; T3 scouting in Phase A absorbs the tracer-bullet benefit (early hard-problem engagement) without abandoning Tier 1's warmup value.
5. **LLM tool integration** — explicit deliverable across all phases (MVP → harness), not optional. Reason: user committed to using LLMs for proof drafting; tool is engineered for evolution and becomes a paper-secondary contribution by Phase D.
6. **Differential testing** — required across all phases as credibility infrastructure, but separated from formal proof. Reason: reviewers will ask "is this a real kernel?"; we answer by showing GPU outputs match references, while formal proof remains in `ℝ`.
7. **Out-of-scope decisions** — backward pass, IEEE-754 fidelity, Python lifter, atomics, autotune all deferred. Reason: each is potentially its own follow-up paper; bundling them creates an unfocused contribution.

## Per-phase implementation plans

This document is the program-level plan. Each phase requires its own implementation plan with concrete tasks and ordering. The implementation plans will be created via the `writing-plans` flow when each phase begins:

- Phase A implementation plan — to be written before starting Phase A
- Phase B implementation plan — written at Gate A → B
- Phase C implementation plan — written at Gate B → C
- Phase D implementation plan — written at Gate C → D

Each implementation plan should reference back to the relevant section of this document for context.
