# Async Modeling Substrate — State-Machine vs TLA (veil / Lentil / TLA-in-Lean)

Research/decision asset for **[#459](https://github.com/Lizn-zn/VeriTile/issues/459)**
(parent map **[#458](https://github.com/Lizn-zn/VeriTile/issues/458)** Property 3;
gated on **[#460](https://github.com/Lizn-zn/VeriTile/issues/460)**; resolves the
modeling-substrate half of #409's "step 3 is research-grade" framing).

## The question

> Do we need TLA (or a TLA-style temporal-logic layer) to verify async/parallel
> correctness, or does a plain Lean state-machine / interleaving model suffice —
> and if a temporal layer is needed, do we adopt an existing tool or build it in
> Lean?

Three candidate substrates for Property 3 (parallel/asynchronous correctness):

1. **Bespoke Lean state-machine** — a small-step interleaving semantics over
   `BlockState` + a memory-consistency (release-acquire / happens-before)
   relation, in plain Lean. *The team default.*
2. **Veil** ([verse-lab/veil](https://github.com/verse-lab/veil)) — an
   Ivy/mypyvy-style automated transition-system verifier embedded in Lean 4;
   VCs in first-order logic discharged by SMT, with model checking + a
   counterexample-to-induction (CTI) invariant-discovery loop.
3. **Lentil** ([verse-lab/Lentil](https://github.com/verse-lab/Lentil)) — a
   formalization of **TLA (Temporal Logic of Actions) in Lean 4** (definitions
   ported from `coq-tla`, proofs in Lean tactics). This *is* the "TLA-in-Lean"
   option.

## TL;DR — recommendation

**No, v1 does not need TLA or any temporal-logic layer. Adopt the team default:
a bespoke small-step interleaving semantics + a release-acquire / happens-before
relation, written in plain Lean, extending the placeholders that already live in
`VeriTile/Concurrency/`.**

- **Both #460 running examples are safety properties**, and neither needs
  liveness or fairness (established construct-by-construct below). Temporal
  operators (`◇`/`□`/`WF`/`SF`) — the *only* thing a TLA layer adds over a plain
  next-state relation — buy nothing for safety: safety is an inductive invariant
  over the transition relation, statable and provable without them.
- **Do not adopt Lentil (TLA-in-Lean) for v1.** Its distinctive value is
  liveness; v1 uses none. It would also impose a *behaviour = infinite sequence
  + temporal formula* modeling style that is a poor fit for the existing
  *relational, state-to-state* correctness surface (`Realizes` / `ComputeCorrect`
  / `RefinesSequential`). Keep it as a **watch-item pointer** for the day a
  future validation target genuinely needs eventual-delivery/termination
  (e.g. DeepEP progress) — that is the only scenario that would reopen this
  decision.
- **Do not adopt Veil as the proof substrate — but borrow its ideas.** Veil's
  metatheory is *first-order transition systems discharged to SMT*; VeriTile's
  hard part (Property 1/2) is *real-analysis over ℝ* (dot products, reductions,
  transcendentals) proved in Mathlib — the whole two-layer design exists because
  that math is **not** SMT-friendly. Routing the concurrency skeleton through
  Veil-SMT while the compute stays in Mathlib creates a two-metatheory seam for
  no payoff on the part that is actually hard. What *is* worth borrowing is
  Veil's **inductive-invariant + refinement framing** (identical to our safety
  proof shape) and, as an *external oracle*, its **model-checker + CTI generator
  to discover the interleaving invariant before we prove it in Lean** — without
  adopting its SMT-first proof pipeline.

Net: **safety over interleaved traces, single Lean/Mathlib metatheory, no
temporal logic, no external verifier in the trusted path.**

---

## 1. What each candidate actually offers

| | Veil | Lentil (TLA-in-Lean) | Bespoke Lean state-machine |
|---|---|---|---|
| **What it is** | Automated + interactive verifier for state-transition systems, embedded in Lean 4 (CAV'25; POPL'26 tutorial). Ivy/mypyvy lineage. | A Lean 4 library formalizing TLA (temporal logic of actions); `coq-tla` port. | Not a tool — a modeling *style*: small-step operational semantics + a happens-before relation, in ordinary Lean/Mathlib. |
| **Reasoning power** | Safety via inductive invariants; concrete + symbolic model checking; CTI-driven invariant discovery. Liveness = future work. | Full temporal reasoning: safety **and** liveness (`◇`, `□`, weak/strong fairness) over infinite behaviours. | Whatever you prove by hand. Naturally: safety/invariants + refinement. Liveness only if you build the temporal scaffolding yourself. |
| **Discharge** | VCs → first-order logic → **SMT** (multiple solvers), with foundational Lean soundness on top. | Lean tactic proofs over the temporal DSL. | Lean tactic proofs; reuses Mathlib real analysis directly. |
| **State model** | Its own imperative transition-system language (relations over FO state). | Behaviours as functions `ℕ → State`, temporal formulas over them. | The existing `BlockState` / `Kernel.exec` / `Op` GADT, extended with a world layer + comm nodes + an `EffectTrace`. |
| **Maturity** | Young (2025), active, published, real distributed-protocol case studies. | Minimal-scope by the authors' own statement; ~small library, actively developed, Apache-2.0. | N/A — but the placeholder substrate already exists in-repo (see §3). |
| **Integration cost for VeriTile** | High: a **second metatheory** and SMT toolchain in the trusted path; state must be re-encoded out of `BlockState`; the ℝ-compute proofs cannot cross into it. | Medium: a temporal DSL layered over a relational correctness surface that doesn't speak temporal; modeling-style mismatch. | Low–medium: extends `Concurrency/`'s existing `HappensBefore` / `AsyncVisibility` / `RefinesSequential`; **one** metatheory throughout. |

---

## 2. Do the v1 examples need liveness, or only safety?

Straight from the #460 feasibility map
([`AsyncRunningExamples.md`](./AsyncRunningExamples.md) §7). The two chosen
running examples bracket the question:

| Example | Correctness property | Class | Needs liveness/fairness? |
|---|---|---|---|
| **`simple_shift`** (ring `put`) | After the shift, each rank's `B` equals its predecessor's `A` — **equal final memory**. | Safety (final-state). Exactly what `RefinesSequential` gestures at. | **No.** |
| **`allgather_gemm_overlapped`** (AllGather→GEMM overlap) | The GEMM on tile `k` reads *the version of the remote tile the matching `signal` released* — a **visibility / happens-before** claim over the interleaving. | Safety **over interleaved traces** (release-acquire). Not final-state-only, but still safety. | **No.** |

The critical observation: `allgather_gemm_overlapped` *does* need more than
equal-final-memory — it needs an **ordering** fact (the acquiring `wait` observes
the releasing `put+signal`). But an ordering fact is a **safety** property: "no
read observes a value before the release edge that publishes it." It is a
predicate on *finite* interleaved prefixes, provable as an inductive invariant.

What it is **not** is a *liveness* property. Neither example asserts:

- *eventual* delivery ("the `put` is eventually visible" — the kernels assume the
  `wait` returns because the hardware/NVSHMEM guarantees it; v1 verifies the
  *conditional* "**if** the wait returns, the value read is the released one");
- termination/progress under fairness ("the scheduler eventually runs the
  producer");
- absence of starvation / weak-or-strong fairness on any action.

Those `◇`/`WF`/`SF`-shaped obligations are precisely the fragment a temporal
logic (TLA / Lentil) exists to serve — and **v1 touches none of them.** This is
the #460 prediction confirmed: *"v1 likely needs safety over interleaved traces,
not liveness/fairness."*

> **The liveness carve-out is explicit.** VeriTile is a *golden-vs-optimized
> equivalence* verifier: it asks "does the optimized kernel compute the same ℝ
> result as the reference," **assuming the concurrency completes**. Progress /
> deadlock-freedom of the NVSHMEM protocol is the platform's contract, out of
> v1 scope (consistent with #458's "hardware layer is future work"). If a later
> target makes *progress itself* the property under test, revisit — see §5.

---

## 3. Why a bespoke Lean state-machine (and why the substrate is half-built)

The team default is not a greenfield build. `VeriTile/Concurrency/` already
contains the *vocabulary* for exactly this substrate, deliberately shipped as
un-wired placeholders (per [`ConcurrencySemantics.md`](./ConcurrencySemantics.md)
§"Trace Vocabulary" and the L1–L4 roadmap):

- **`RefinesSequential`** (`Concurrency/Refinement.lean`) — the final-state
  safety surface: *every successful concurrent run has a successful sequential
  run with the same final memory.* This is `simple_shift`'s property verbatim,
  and the skeleton of `allgather_gemm`'s.
- **`EffectTrace` / `OccursBefore` / `HappensBefore`** (`Concurrency/Trace.lean`)
  — an explicit-order happens-before relation with a documented extension point:
  *"future discipline layers can strengthen this with program-order, async-wait,
  and barrier edges without changing the consumer theorem surface."* This is the
  release-acquire hook `allgather_gemm_overlapped` needs.
- **`AsyncVisibility` / `BarrierDiscipline`** (`Concurrency/Discipline.lean`) —
  discipline predicates already stating *"every async `wait` is justified by a
  prior async `copy` that happens-before it"* and the barrier analogue. This is
  the acquire-pairs-with-release obligation, one rank-generalization away from
  the cross-rank `signal`/`signal_wait_until` pair.
- **`PermissionModel` / `OwnershipMap`** — abstract ownership for race-freedom
  arguments, already parameterized (not hard-wired to warp ownership).

So the recommended path is *finish wiring what exists*, not *import a new
paradigm*. Decisive advantages:

1. **One metatheory.** The interleaving/ordering skeleton and the ℝ-compute proof
   (`Op.dot` correctness, tiled-reduction identities, the FA-1 chain) live in the
   *same* Lean/Mathlib world. The safety theorem for `allgather_gemm_overlapped`
   *reduces the interleaved execution to the existing sequential `Realizes`
   result*: given the release-acquire discipline holds, every read of an `A_ag`
   tile observes the value the matching signalled `put` wrote, so the interleaved
   run is equivalent to a sequential all-gather-then-GEMM — and *that* is already
   a proven compute kernel. Veil/Lentil cannot cross this seam; a bespoke
   relation is *on the same side of it*.
2. **Reuses the neutral core.** The world layer (rank → per-rank memory), the
   `put`/`get`/`signal`/`wait` comm nodes, and the happens-before edges land on
   `BlockState` / the `Op` GADT under the **union policy** (#465): one closed
   GADT, extended, not forked. Aligns with the already-agreed foundation shape.
3. **Right-sized to v1.** Two examples, bounded ranks and tiles. The proof
   obligation is a hand-tractable invariant, not a search that needs SMT
   automation to be feasible.
4. **No external tool in the trusted path.** The verifier stays a
   self-contained Lean artifact (matters for the artifact contract / axiom
   audit the repo already enforces).

### Cost/benefit vs building TLA-in-Lean ourselves

The issue also floats *implementing TLA-in-Lean ourselves*. This is strictly
dominated: it is more work than the bespoke state-machine (a temporal-operator
DSL + its metatheory) to obtain a capability (liveness) v1 does not use, and if
we ever *did* want it, **Lentil already exists** and we would adopt it rather
than re-derive `coq-tla`. So DIY-TLA is off the table in both directions:
unnecessary now, and pre-empted by Lentil later.

---

## 4. Why not Veil, and why not Lentil (the sharper case)

**Veil** is the most tempting to over-adopt, because its *sweet spot — automated
safety of distributed protocols via inductive invariants — literally names our
problem.* The reasons it is nonetheless the wrong *substrate*:

- **Metatheory mismatch on the hard half.** Veil discharges to SMT/FOL. Property
  1/2 correctness is real-analysis (transcendentals, tiled-reduction
  associativity over ℝ) — the reason the two-layer ℝ-erasure architecture exists
  is that this is *not* expressible as decidable FO/SMT. Veil helps only with the
  finite interleaving skeleton, which in v1 is the *small* part. We'd pay a
  two-metatheory seam to automate the easy half.
- **State re-encoding.** Veil wants the system stated in *its* transition
  language, not our `BlockState`/`Kernel.exec`/`Op` GADT. That either duplicates
  the model or abandons the neutral core the rest of v1 is built on.
- **Trusted-path weight.** An SMT toolchain enters the verifier's trust story and
  the artifact/axiom audit.

But Veil is **not dismissed** — it is *reclassified* from substrate to tool:

- Its **inductive-invariant + refinement discipline** is the exact shape our
  bespoke safety proof takes; use it as the design template.
- Its **model checker + CTI generator** are a genuinely useful *external,
  untrusted oracle*: run the interleaved model in Veil to *discover* the
  happens-before invariant and find missing-edge bugs in seconds, then *transcribe
  and prove* the invariant in Lean. This gets the automation benefit without
  putting SMT in the trusted path. **Watch-item, not a v1 dependency.**

**Lentil** is the cleanest reject for v1: it is a *TLA formalization*, and TLA's
entire reason to exist over a plain next-state relation is temporal/liveness
reasoning. With v1 provably safety-only (§2), the temporal machinery is dead
weight, and its behaviour-centric modeling style clashes with the relational
`Realizes`/`RefinesSequential` surface. Correct disposition: **keep a pointer**;
if a future property is genuinely liveness (§5), *adopt Lentil* rather than
build TLA-in-Lean — but do not pull it into v1.

---

## 5. Decision + what the async-impl tickets build on

**Decision (Property 3 substrate): bespoke small-step interleaving semantics +
release-acquire / happens-before relation, in plain Lean, extending
`VeriTile/Concurrency/`. No TLA layer. Veil = optional external invariant-finding
oracle. Lentil = deferred pointer for a future liveness need.**

Concrete substrate the implementation tickets inherit (staged per #460 §7's
sequencing hint, each rung independently checkable):

1. **World layer + `put`** (unlocks `simple_shift`, *safety only*): rank →
   per-rank memory over `BlockState`; a peer-addressing map; a `put` comm node.
   Theorem shape: a `RefinesSequential`-style equal-final-memory statement.
2. **`signal`/`wait` + happens-before edges** (unlocks non-overlapped
   `allgather_gemm`): cross-rank `signal`/`signal_wait_until` nodes; strengthen
   `HappensBefore` with the release-acquire edge; generalize `AsyncVisibility`
   from single-program async-copy to cross-rank signal/wait. Theorem shape: *any
   read across a matched wait observes the released value* (an inductive
   invariant on trace prefixes) → reduces to the sequential `Realizes`.
3. **Interleaving scheduler + double-buffering** (unlocks
   `allgather_gemm_overlapped`): an executable interleaving relation over the
   `EffectTrace`; async-token/pipeline-stage state; memory-space tags (the
   Phase-2 `shared` tag from #465). The overlap safety theorem is: *under the
   release-acquire discipline, every interleaving of put(k+1)‖gemm(k) yields the
   same C as the sequential schedule* — safety over interleaved traces, still no
   liveness.

**Re-open trigger (the single condition that flips this decision):** a target
whose *property under test is progress/liveness itself* — eventual delivery,
deadlock-freedom, fairness-dependent termination (plausibly DeepEP's
dispatch/combine progress, a deferred validation target in #458). If that
graduates from validation-target to verification-goal, adopt **Lentil** for that
fragment (layered above the same state-machine, not replacing it). Nothing in the
v1 running examples reaches that trigger.

---

## 6. What this feeds

- **Resolves the #459 modeling-substrate decision** and unblocks the
  **Async/parallel implementation stack** fog in #458's "Not yet specified":
  the four #409 layers (world/symmetric heap, comm primitives in the AST,
  interleaving scheduler + happens-before, trace-level correctness predicates)
  now have a chosen substrate and a staged rung order to slice into tickets.
- **Confirms the #460 bracket**: `simple_shift` = safety floor,
  `allgather_gemm_overlapped` = safety-over-traces ceiling; the decision holds
  because the ceiling is still safety, not liveness.
- **Folds into the proposal** (#466): Property 3's "high-level idea" is now
  *interleaving state-machine + release-acquire, single Lean metatheory with the
  ℝ-compute proofs; TLA deferred* — a concrete plan slot, not a `TBD`.

## 7. Sources

- Veil — [veil.dev](https://veil.dev/),
  [verse-lab/veil](https://github.com/verse-lab/veil), CAV'25 paper
  *"Veil: A Framework for Automated and Interactive Verification of Transition
  Systems."*
- Lentil — [verse-lab/Lentil](https://github.com/verse-lab/Lentil) (TLA
  formalization in Lean 4; `coq-tla`-ported definitions).
- In-repo anchors — `VeriTile/Concurrency/{Refinement,Trace,Discipline}.lean`;
  [`ConcurrencySemantics.md`](./ConcurrencySemantics.md);
  [`AsyncRunningExamples.md`](./AsyncRunningExamples.md) (#460);
  [#409](https://github.com/Lizn-zn/VeriTile/issues/409) gap table.
