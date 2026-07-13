# VeriTile — Proposal Slide Outline

**Living slide skeleton.** One `##` heading = one slide; bullets are the
slide's content gist, `[visual: …]` notes the intended figure. Maintained in
lock-step with [Proposal.md](./Proposal.md) — same spine, talk resolution.
This markdown outline is the source of truth; the deck renders from it
directly via [slides/build.sh](./slides/README.md) (Marp → HTML/PDF; verso
deprecated for slides), so the outline stays cheap to react to.

---

## 1. Title

- **VeriTile: Verifiable Kernel Optimization**
- Prove — don't just test — that your optimized GPU kernel matches the
  golden reference.
- Authors / affiliation `TBD`.

## 2. The pain: optimizing kernels is testing-bound

- Golden-vs-optimized differential testing burns **hours–days of GPU time**
  per iteration.
- A failing diff **doesn't localize** the cause.
- A passing diff **proves nothing** — absence of bugs is out of testing's
  reach (boundaries, masks, interleavings).
- [visual: today's loop — rewrite → GPU test farm → diff → shrug]

## 3. Reframe: verifiable code optimization

- Verifier core (Lean 4) *proves* golden ≡ optimized over user-level code.
- Harness drives the workflow and **localizes** failures.
- [visual: same loop with the GPU test farm replaced by verifier + harness;
  salvage the trust-boundary diagram idea from the archived proposal]

## 4. Running example: FlashAttention

- Naive attention (golden) vs. FlashAttention (blockwise + online softmax)
  — ℝ-equivalent, structurally unrecognizable, historically bug-prone.
- Already proved in VeriTile: FA-1 fwd/bwd, FA-2, FA1 ≡ FA2 headline
  corollaries.
- [visual: side-by-side kernel snippets + the proved equivalence statement]

## 5. Running example 2: async/overlap

- TileScale `simple_shift` (the floor: one ring `put`) and
  `allgather_gemm_overlapped` (canonical compute–comm overlap) — selected
  in [AsyncRunningExamples.md](./AsyncRunningExamples.md); DeepEP kept as
  the eventual validation target.
- Story job: the bug class testing is worst at — interleaving-dependent.
- [visual: overlap timeline — GEMM tiles consumed as remote data arrives]

## 6. High-level idea: two-layer verification

- Lean proves the **ℝ-algorithm layer**; the **FP gap** is externally
  checked, never bridged in Lean (IEEE-754 permanently external).
- Rounding model: where rounding happens (in Lean) vs. how big (external).
- [visual: two-layer diagram with the explicit non-bridge]

## 7. Four properties

- P1 single-kernel correctness · P2 fusion (fused kernel ≡ stage pipeline
  on outputs, ℝ layer; SwiGLU) · P3 async/parallel · P4 FP-warning
  patterns (not bounds).
- [visual: 4-quadrant status card — exists / substrate / gap, from
  Proposal.md §3.2 table]

## 8. One core, two frontends

- DSL-agnostic neutral core (union policy); Triton + Tilelang frontends =
  syntax + lowering only.
- [visual: hourglass — two DSLs above, one core, one proof stack below]

## 9. The harness

- CLI · golden-runner · verifier-driver · failure localization/report —
  boundary `TBD`
  ([harness design](https://github.com/Lizn-zn/VeriTile/issues/464)).
- [visual: developer's-eye workflow, verifier as a black box inside]

## 10. Where we stand

- ~151 proven kernels, FA-1/FA-2 fwd+bwd closed, 0 `sorry`; rounding model
  landed; neutral core relocated; async examples selected; FP-warning
  catalog + Tilelang mapping + fusion correctness designed.
- [visual: repo-status strip or the §3.2 status table]

## 11. Plan

- M1 model the user code → M2 v1 features → M3 validation (FA + async
  cases); compiler/hardware layers = future work.
- [visual: milestone timeline, ticket-linked]

## 12. Ask / discussion

- Venue: SE vs. Systems — `TBD` until results accumulate.
- Feedback wanted on: framing (§2–3), example choice (§4–5), evaluation
  plan.
