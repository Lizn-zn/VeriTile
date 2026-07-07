# FP-Precision Warning Pattern Catalog (v1)

Research / design summary for [#462](https://github.com/Lizn-zn/VeriTile/issues/462),
under the v1 map [#458](https://github.com/Lizn-zn/VeriTile/issues/458) —
**Property 4**. Output of this ticket: the pattern catalog + a detection /
surfacing plan. It graduates the map's *"FP-precision warning checker"* fog
into a concrete rule set that a later implementation ticket builds.

## 1. Scope — what Property 4 is, and is not

VeriTile keeps a permanent **two-layer split** (`ArchitectureHandoff.md` §1):
proofs run on an erased **ℝ-algorithm** layer; the FP/compute gap is checked
externally and never bridged inside Lean. Property 4 does **not** attempt to
close that gap. In particular it is *not*:

- a tight/accurate FP delta bound — explicitly **out of scope** on the map
  ("bounds are usually too loose to help"); and
- an internal IEEE-754 numeric semantics — **permanently external** by design.

Property 4 is instead a **pattern-based warning layer**: a static, mostly
syntactic check over kernel *structure* that flags recurring code shapes known
to introduce or amplify precision bias, and surfaces them to the developer in
the harness report. A warning is advisory — it never changes the ℝ correctness
verdict. It says *"this shape is a known precision hazard; the ℝ proof cannot
see it because ℝ is associative and lossless."*

The layer is built **on top of** the `#447` rounding model — it is a
static-AST / golden-vs-optimized *diff* over rounding-event structure, **not** a
new numeric semantics.

## 2. Foundation reused (do not rebuild)

### 2.1 The `#447` rounding model — rounding-event *structure*

`Triton/Float/{RoundingModel,EvalOpR,StepR,Refine}.lean` already captures
*where* and *how many* rounding events a kernel incurs, without asserting their
*magnitude* (magnitude stays the external gap). Two facts make it the substrate:

1. **There are exactly two rounding-event sites** (`RoundingModel.lean`):
   - `RoundingModel.cast` — a float cast, rounds at the destination dtype
     (`evalOpR`'s `Op.castFloat` arm);
   - `RoundingModel.storeValue` — a narrowing store, rounds at the buffer dtype
     (`StepR`'s `writeMemAsR`).
   Every quantization a kernel can suffer is one of these two. The warning layer
   never has to discover *new* rounding sites; it classifies the ones `#447`
   already enumerates.

2. **The realization spec encodes the event structure symbolically.**
   `ComputeRefine.RealizesR` (`Refine.lean`) states correctness against
   `expected : RoundingModel → ι → α` — a spec that is a *function of the
   model*. Its syntax **is** the rounding-event inventory of the observed
   dataflow (`Refine.lean` doc-comment):
   - `expected R i = core i`   — no `R` occurs → the path is **rounding-free**;
   - `expected R i = R.round dt (core i)` — exactly **one** final quantization;
   - nested / iterated `R.round` — **one event per occurrence**.
   The named shapes `RoundFree` and `SingleRounding` are the two v1 endpoints
   of this inventory.

This gives the warning layer a *sound, proof-backed* handle: a kernel proven
`RoundFree` **provably has zero rounding events** on its observed path, so no
narrowing-cast warning (W3, below) is even possible for it; a `SingleRounding`
kernel has provably exactly one. A discharged `#447` proof is therefore
*evidence of absence* of a class of warnings, not merely a place to raise them.

### 2.2 The stable / naive math pairs

`Triton/Math/` already carries the *proven* ℝ-equalities between a naive shape
and its numerically-stable counterpart — precisely the pairs whose FP behavior
diverges even though their ℝ forms are equal:

- `Softmax.naive_eq_stable` — `naiveSoftmaxMath = stableSoftmaxMath` (max-shift);
- `LogSumExp.stableLSE_eq_LSE` / `log_sum_exp_shift_invariant` — max-shifted LSE;
- `Reduction.welfordVar` / `layerNorm` — two-pass mean/variance (vs a naive
  running `∑x²` fused form).

Each such identity is exactly a case where **the ℝ proof passes but FP bias
differs by construction**. These are the ground truth for the diff-based
patterns (W2, W4): the *golden* side references the stable member, the
*optimized* side may have degraded to the naive member.

### 2.3 The Op GADT — the static match surface

`Core/Ast.lean` gives the typed node vocabulary the static patterns match on:
`Op.sub`, `Op.add`, `Op.exp`/`exp2`, `Op.log`/`log2`, `Op.rsqrt`/`sqrt`,
`Op.reduceSum`/`reduceMax`, `Op.castFloat`, and the store statements in
`StepR`. Static patterns are structural matches on this tree; dtype tags
(`FloatDType`) are carried on every node, so narrowing is decidable at match
time.

## 3. The pattern catalog

Five patterns ship in v1. For each: the hazard, the **detection signal** (which
of the three signal kinds — §4 — fires it), and how it **surfaces**.

### W1 — Catastrophic cancellation (subtracting near-equal quantities)

- **Hazard.** `Op.sub a b` where `a` and `b` are close in magnitude loses the
  leading significant digits; relative error explodes.
- **Signal.** *Static AST* (kind A), heuristic + provenance. Flag `Op.sub`
  (and `add` of oppositely-signed operands) where both operands trace, through
  the dataflow, to a **common large-magnitude source** — e.g. `x - reduceMax x`
  is *benign* (that is the intended shift), but `exp(a) - exp(b)` for correlated
  `a,b`, or `(x*x) - mean²` (the fused-variance anti-pattern) is flagged.
  Because full magnitude reasoning is external, W1 is a *heuristic* over
  provenance, tuned to low false positives; it is the softest signal in the set.
- **Surface.** `severity = info` unless the subtraction feeds a `div`/`rsqrt`
  (then `warn`): cancellation followed by division is the classic amplifier.

### W2 — Large unstable accumulation (naive vs compensated / online)

- **Hazard.** A long reduction (`reduceSum`, or a `forLoop` accumulator) without
  a max-shift / Welford / compensated (Kahan) structure accumulates rounding
  bias that grows with the reduction length.
- **Signal.** Two complementary signals:
  - *Golden-vs-optimized diff* (kind B) — the **primary** signal. The golden
    kernel's dataflow references a stable member from `Triton/Math/`
    (`stableSoftmaxMath`, `stableLSE`, `welfordVar` two-pass); the optimized
    kernel has collapsed it to the naive member (`naiveSoftmaxMath`, running
    `∑exp`, fused `∑x²`). The `naive_eq_stable` / `stableLSE_eq_LSE` identities
    guarantee the ℝ proof still closes — so *only* the structural diff can catch
    it.
  - *Static AST* (kind A) — as a single-kernel fallback: a `reduceSum` over
    `Op.exp` with **no** preceding `reduceMax`-subtraction on the same lane, or
    a variance computed as `mean(x²) − mean(x)²` rather than two-pass.
- **Surface.** `severity = warn`; report includes which stable pattern the
  golden used and where the optimized kernel dropped it (statement / `Op` node).

### W3 — Narrowing cast / reduced-dtype store (the two `#447` sites)

- **Hazard.** `RoundingModel.cast src dst` into a narrower grid, or
  `RoundingModel.storeValue` at a narrow buffer dtype, quantizes; doing so
  *inside* an accumulation (rather than only at the final output) compounds.
- **Signal.** *Rounding-event count / site inventory* (kind C) — the **tightest,
  most principled** signal, read directly off `#447`:
  - enumerate the `R.round` occurrences in the kernel's `expected`
    (`RealizesR`) shape, or equivalently the `castFloat`-to-narrower and
    narrowing-store nodes in its AST;
  - `RoundFree` ⇒ 0 events ⇒ **no W3** (proof-backed absence);
  - `SingleRounding` ⇒ 1 event at the output ⇒ **minimal, expected** — `info`;
  - `n > 1` events, or any event **on the accumulation path** (not just the
    final store) ⇒ `warn`;
  - a **count/site delta vs golden** (the optimization introduced a narrowing
    the golden did not have) ⇒ `warn`, with the delta reported.
- **Surface.** `severity` scales with event count and placement as above; the
  report cites each rounding site by dtype (`bf16`/`fp16`/…) and node. `#447`'s
  `GridNested` mixin lets an *upcast* (`.to(float32)` on a narrow value) be
  recognized as a no-op and **not** warned.

### W4 — Reassociation of a non-associative FP sum

- **Hazard.** An optimization reorders / regroups a sum (or product): the ℝ
  result is identical (ℝ `+` is associative), but the FP result differs by the
  reassociation error.
- **Signal.** *Golden-vs-optimized diff* (kind B). Compare the **reduction-tree
  shape** of the two kernels: same multiset of leaves, different association
  structure (e.g. sequential vs pairwise-tree `reduceSum`, or a hoisted partial
  sum). The `ComputeRefine.Realizes` ℝ-equality proof is what *licenses* the
  optimization, so — as with W2 — only the structural diff surfaces the FP
  hazard the proof is blind to.
- **Surface.** `severity = info` by default (reassociation is often intentional
  and mild), escalating to `warn` when combined with W2 (long reduction) or a
  reduced dtype (W3) on the same accumulator.

### W5 — Overflow / underflow-prone transcendental without a guard

- **Hazard.** `Op.exp`/`exp2` on an unshifted argument overflows for large
  inputs; `Op.rsqrt`/`sqrt`/`div` by a near-zero denominator (variance without
  `+ ε`) underflows / blows up; `Op.log` near zero.
- **Signal.** *Static AST* (kind A). Match:
  - `exp(t)` where `t` is **not** of the form `(… − max…)` on its lane
    (the softmax/LSE guard) — reuses the same shift-detection as W2;
  - `rsqrt(t)` / `_ / sqrt(t)` where `t` is a variance/sum-of-squares **not**
    of the form `(… + ε)` (compare `Reduction.layerNorm`, which carries `+ ε`);
  - `log(t)` where `t` is a sum that can reach `0`.
- **Surface.** `severity = warn`; report names the missing guard (max-shift /
  epsilon) so the fix is actionable.

### Catalog summary

| ID | Pattern | Primary signal (kind) | Default severity |
|----|---------|-----------------------|------------------|
| W1 | Catastrophic cancellation | A static AST + provenance | info → warn |
| W2 | Unstable accumulation (naive vs stable) | B golden-vs-optimized diff (A fallback) | warn |
| W3 | Narrowing cast / reduced-dtype store | C `#447` rounding-event count | info → warn |
| W4 | FP-sum reassociation | B golden-vs-optimized diff | info → warn |
| W5 | Unguarded exp / rsqrt / log | A static AST | warn |

## 4. Detection-signal taxonomy

Every pattern is fired by one of three signal kinds; the checker is organized as
three passes.

- **(A) Static AST pattern** — a structural match over a **single** kernel's
  `Op` GADT tree (+ dtype tags + local dataflow provenance). Cheap, runs on
  every kernel independently of a golden. Fires W1, W5, and the single-kernel
  fallback of W2.
- **(B) Golden-vs-optimized structure diff** — needs **both** kernels; compares
  their dataflow / reduction-tree shape (and, where available, their `expected`
  spec shapes). Catches exactly the hazards the ℝ-equality proof is *designed*
  to hide: stable→naive degradation (W2) and reassociation (W4). This is the
  signal that most directly serves the harness's golden-vs-optimized workflow.
- **(C) Rounding-event count / site delta** — reads the `#447` structure: the
  `R.round` occurrences in a `RealizesR` `expected` shape, or the
  narrowing-`castFloat` / narrowing-store nodes in the AST; and, for a diff, the
  **count/site delta** against the golden. The only signal with a *proof-backed
  lower bound* (`RoundFree`/`SingleRounding` ⇒ provably 0/1 events). Fires W3 and
  reinforces the severity of W2/W4.

Kinds B and C both fit the harness's golden-vs-optimized driver; kind A also
runs standalone on a lone kernel.

## 5. Surfacing plan

Warnings ride **alongside** the correctness verdict in the harness report
(map §"Harness component breakdown", [#464](https://github.com/Lizn-zn/VeriTile/issues/464));
they never gate the ℝ proof. Proposed record schema (one row per finding):

```
FpWarning {
  pattern   : W1 | W2 | W3 | W4 | W5      -- catalog id
  severity  : info | warn                 -- §3 rules
  kernel    : name                        -- optimized kernel under test
  site      : statement / Op node ref     -- where the shape occurs
  signal    : staticAST | goldenDiff | roundingCount   -- kind A/B/C
  rationale : text                        -- e.g. "reduceSum over exp with no max-shift"
  golden    : optional { what the golden did differently }   -- kinds B/C only
  events    : optional { count, dtypes }  -- kind C only, from #447
}
```

Reporting rules:
- **Non-blocking.** A verified kernel with warnings is still *verified*; the
  report lists warnings in a separate section, ranked `warn` before `info`.
- **Proof-discharged absence is stated positively.** When a kernel is proven
  `RoundFree` / `SingleRounding`, the report says so ("0 / 1 rounding events,
  proof-backed") instead of staying silent — this is the payoff of building on
  `#447` and distinguishes "checked, clean" from "not checked".
- **Actionable rationale.** Each `warn` names the missing structure (max-shift,
  epsilon, compensation, or the dropped stable-pattern reference) so the
  developer knows the fix.

## 6. What this opens / hand-off to implementation

This catalog **closes the pattern-catalog research** and opens the map's
*"FP warning checker"* execution fog. The follow-up implementation ticket
should:

1. Implement the **three passes** (§4) — the static-AST matcher over `Op`
   (A), the golden-vs-optimized structural diff (B), and the `#447`
   rounding-event reader over `expected` / AST (C).
2. Emit the **`FpWarning` records** (§5) and wire them into the harness report
   ([#464](https://github.com/Lizn-zn/VeriTile/issues/464)); gate on nothing.
3. Seed the golden/stable side of B and C from the existing
   `Triton/Math/` identities (§2.2), so W2/W4 have ground truth from day one.

Per the map's sequencing this stays a **thin static layer on `#447`**; it adds
no numeric semantics and touches none of the existing evaluator/stepper.

## 7. Open questions / gap analysis (fold back into the proposal)

- **W1 provenance precision.** Cancellation without magnitude is inherently
  heuristic; the false-positive rate needs calibration on the TritonBench-G
  ports before W1 is promoted above `info`. Magnitude remains the external gap
  by design — W1 must not pretend otherwise.
- **B needs a canonical kernel-normal-form** to diff reduction-tree shape
  reliably (α-renaming, commutative-operand ordering). Reuse of the DSL's
  lowering (`DSL/Expansion/`) is the likely source.
- **`expected`-shape availability.** Signal C is sharpest when the kernel has a
  `RealizesR` proof (the `expected` shape is then explicit). For kernels with
  only a `ComputeCorrect` proof, C falls back to counting AST narrowing nodes —
  sound but coarser. Encouraging `RealizesR` proofs for reduced-dtype kernels
  is a proposal-level recommendation.
- **atomicRMW carve-out.** `#447` still evaluates `atomicRMW` inputs under
  `.triv` (`ArchitectureHandoff.md` §8 follow-up); until that closes, C
  under-counts rounding events inside atomic accumulations — the checker should
  emit a *coverage caveat* rather than a false "0 events" for such kernels.
