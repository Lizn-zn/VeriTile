/-
VeriTile — Architecture overview, written as a Verso `Page`.

The page renders to HTML via Verso's blog/site machinery. Custom block
components produce semantic-class wrappers (`card card-blue`, `cols`, …);
slide-style presentation is handled by `static_files/slides.css` +
`static_files/slides.js`, which the theme (`VeriTileSite.lean`) loads.
-/

import VersoBlog

open Verso Genre Blog

section
open Verso Doc Elab ArgParse
open Lean
open Verso Output Html
open Template
open scoped Lean.Doc.Syntax

set_option pp.rawOnError true

/-! Layout building blocks. Each one wraps its body in a typed div whose CSS
class is consumed by `static_files/slides.css`. -/

block_component +directive cardBlue where
  toHtml id _ _ goB contents := do
    pure {{<div class="card card-blue" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cardOrange where
  toHtml id _ _ goB contents := do
    pure {{<div class="card card-orange" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cardGreen where
  toHtml id _ _ goB contents := do
    pure {{<div class="card card-green" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cardPurple where
  toHtml id _ _ goB contents := do
    pure {{<div class="card card-purple" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cardMuted where
  toHtml id _ _ goB contents := do
    pure {{<div class="card card-muted" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cols where
  toHtml id _ _ goB contents := do
    pure {{<div class="cols cols-2" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive cols3 where
  toHtml id _ _ goB contents := do
    pure {{<div class="cols cols-3" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive pipeline where
  toHtml id _ _ goB contents := do
    pure {{<div class="pipeline" id={{id}}>{{← contents.mapM goB}}</div>}}

block_component +directive numgrid where
  toHtml id _ _ goB contents := do
    pure {{<div class="numgrid" id={{id}}>{{← contents.mapM goB}}</div>}}

end


#doc (Page) "VeriTile : Triton, Proven" =>

— Automated Verification for Triton GPU kernels in Lean 4


# Worked example — stable softmax

`Triton source  →  embedded in Lean 4 (region-polymorphic DSL)  →  theorem proved by an LLM.`

::::cols3

:::cardBlue
*1 · Triton source*

```
# Naive form is y = exp(x) / sum(exp(x)),
# but exp(large) overflows, so subtract
# the row max first ("stable softmax").
@triton.jit
def softmax(x, y, N: tl.constexpr):
  pid  = tl.program_id(0)              # row id
  offs = pid * N + tl.arange(0, N)     # row offsets
  a    = tl.load(x + offs)             # load row
  m    = tl.max(a, axis=0)             # row max
  e    = tl.exp(a - m)                 # shifted exp
  s    = tl.sum(e, axis=0)             # row sum
  tl.store(y + offs, e / s)            # write
```

The actual `.py` kernel a Triton user ships.

:::

:::cardOrange
*2 · Embedded in VeriTile (Lean 4 DSL)*

```
-- Same kernel, embedded inside Lean 4.
-- `triton { … }` desugars to a typed
--   Op : TileDType → TileShape → Type
-- AST. `$(xReg)` / `$(yReg)` keeps the
-- kernel region-polymorphic.
def stable (xReg yReg : RegionName) (N : Nat)
    : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  a    := tl.load($(xReg) + offs)        -- load row
  m    := tl.max(a)                       -- row max
  e    := tl.exp(a - m)                   -- shifted
  s    := tl.sum(e)                       -- row sum
  y    := e / s                           -- output
  tl.store($(yReg) + offs, y)
}
```

Mirrors the `.py` line-for-line. Typed end-to-end — no dynamic tag.

:::

:::cardGreen
*3 · Theorem + LLM-generated proof*

```
-- `softmaxSpec` is the math reference;
-- `scripts/prove.sh` → `/lean4:autoprove`
-- closed the whole proof end-to-end.
theorem stable_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h : TensorView.loaded s
           (programTileView s xReg N)
           (fun idx => xs idx.1)) :
    ∀ idx,
      TensorView.observe
        (exec (stable xReg yReg N) s)
        (programTileView s yReg N) idx
        = softmaxSpec xs idx := by
  intro idx
  have h := stable_eq_spec xs (tileMax hN xs)
  exact congrFun h ⟨idx.1, by omega⟩
```

User wrote zero tactics — agent did multi-cycle iteration + deep-mode escalation.

:::

::::


# What VeriTile sets out to do

::::cols
:::cardBlue
*The problem*

Triton powers production attention, GEMM, normalization, SSM, RoPE, MoE
routing — and yet there is no machine-checked way to certify that an
optimized Triton kernel computes the same thing as the naive reference.

Engineers re-derive the algebra by hand for every variant, and a single
off-by-one (`mask=`, `other=`, partial KV block, D-tail) can silently
change results.

:::

:::cardOrange
*Our approach*

* Embed a Triton-style DSL inside Lean 4.
* Give the embedded kernel formal operational semantics.
* Prove kernel-pair refinement and correctness against math specs.
* Use an LLM agent (`scripts/prove.sh` → `/lean4:autoprove`) to drive proofs.

:::

::::

:::cardGreen
*What success looks like.* A Triton engineer pastes a real `.py` kernel,
annotates a math spec, and VeriTile + LLM closes the proof. Output: a Lean
theorem stating that the kernel refines its specification — for every
input, every `program_id`, every tile.

:::


# Four axes that must advance together

::::cols
:::cardBlue
*1 · Real DSL surface*

Accept the ops, control flow, memory and concurrency primitives that
production `.py` kernels actually use — not a minimal subset.

:::

:::cardOrange
*2 · Production-scale algorithms*

Full FA-1 forward + backward, FA-2, fused-norm family, grouped GEMM, SSM,
RoPE, GQA / MQA / MLA — not 8 hand-picked theorems.

:::

::::

::::cols
:::cardGreen
*3 · Triton-user friendliness*

Move the user's delta from "rewrite into the embedding" toward "paste-in
`.py`" with near-zero changes (Python lifter is roadmap).

:::

:::cardPurple
*4 · Production horizontal infra*

Algorithm/Compute split, Memory frame, Concurrency refinement, ND-general
framework — no ad-hoc dimension shortcuts.

:::

::::


# Two-layer architecture: Lean proof  ⟂  external testing

::::cols
:::cardBlue
*Algorithm layer · Lean proof*

`Kernel.AlgorithmCorrect ck post`

Real / Int / Nat semantics. No IEEE rounding, no NaN.

* `eraseDType`: `fp32 → ℝ`, `int32 → ℤ`, `uint32 → ℕ`.
* Computable bit-pattern decoder for `fp32` constants.
* Existing `Kernel.Correct` proofs reused on the algorithm side.

Catches *algorithm-structure bugs* — every wrong recurrence, wrong
masking, wrong reduction is a failed Lean proof.

:::

:::cardOrange
*Compute layer · External testing*

Differential testing on a representative kernel set.

Hand-written `.py` Triton ↔ PyTorch / `flash-attn`, ε-bound match.

* 1–2 Tier-1/2 kernels, FA-1 forward, mid-term FA-2 / FA-1 backward.
* Tolerance: Tier 1+2 ≤ 1e-5; FA-class ≤ 1e-3.
* *Not a phase gate*, *not a Lean theorem* — a credibility artifact.

Catches *IEEE-754 edge cases* (NaN, denormal, overflow, fast-math).

:::

::::

:::cardMuted
The bridge between the two layers is empirical, by design. IEEE-754
formalization is permanently outside Lean proof scope. *VeriTile-certified*
= both layers green.

:::


# End-to-end pipeline

The user's `triton { ... }` source flows through seven typed stages before
becoming a Lean theorem:

:::pipeline
* `triton { … }` — surface syntax
* DSL macro — elaboration + typing
* `ComputeKernel` — fp32 / int32 / uint32 + bitcast
* `toAlgorithm?` — `Except EraseDTypeError`
* `AlgKernel` — ℝ / ℤ / ℕ AST
* `exec` — `evalOp` · `stepStmt` · `forLoop_inv`
* Theorem — `Correct` / `Refine`

:::

Cross-cutting subsystems plug into the same `BlockState`:

::::cols
:::cardBlue
*Memory* — `Bounds` · `Footprint` · `Frame` · `Typing` · `View`

:::

:::cardOrange
*Concurrency* — `Trace` · `Refinement` · `Atomic` · `Discipline`

:::

::::

::::cols
:::cardGreen
*Launch* — `Grid` (ND `program_id`) · `Composition` (disjoint merge)

:::

:::cardPurple
*LoopInvariant* — `forLoop_inv` master induction lemma

:::

::::


# DSL layer — accept Triton-like surface syntax

`VeriTile/Triton/DSL/`

::::cols3
:::cardBlue
*Input*

* User-written `triton { ... }` blocks.
* Sugars: `tl.load / tl.store / tl.dot / tl.where / tl.sum / tl.max / tl.exp / tl.arange / tl.program_id`.
* Kwargs: `mask=`, `other=`, `axis=`, `keep_dims=`.
* Region-polymorphic kernels via `$(xReg)`, `$(yReg)`.

:::

:::cardOrange
*What we did*

* `Syntax` + `Expansion`: macro elaborator desugars into typed `Op / Stmt`.
* `Typing`: infers `TileDType × TileShape`, inserts `NumericDType / Broadcast` evidence.
* `Metadata`: kwargs, axis, `keep_dims`.
* Optional well-formedness checker.

:::

:::cardGreen
*Output*

* Typed AST: `Op : TileDType → TileShape → Type`.
* `Stmt` list (assigns, stores, forLoops).
* A `Kernel` record `{inputs, outputs, body}`.
* Same kernel re-instantiable on any region names.

:::

::::

```
def naiveSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid  := tl.program_id(0)
  offs := pid * $(N) + tl.arange($(N))
  x    := tl.load($(xReg) + offs)
  e    := tl.exp(x);  s := tl.sum(e);  y := e / s
  tl.store($(yReg) + offs, y)
}
```


# Core AST — types, shapes, and the typed `Op`

`VeriTile/Triton/Core/`

::::cols3
:::cardBlue
*Input*

* AST nodes from the DSL macro.
* User-supplied dtype: `.real / .nat / .int / .bool`.
* User-supplied shape: `.scalar / .vec n / .mat m n / ND`.

:::

:::cardOrange
*What we did*

* `TileDType / TileShape / TileIndex / TileCarrier`.
* `Tile` = `TileIndex shape → TileCarrier dtype`.
* `Op : TileDType → TileShape → Type` indexed by both — no existential wrapper.
* `Stmt` is the existential boundary (assign / store / forLoop).

:::

:::cardGreen
*Output*

* Fully-typed kernel structure used by every layer.
* Smart constructors: `Tile.scalar / Tile.vec / Tile.mat`.
* `bop / uop / cop` with `NumericDType` + `Broadcast` evidence.
* `Kernel = { inputs, outputs, body : List Stmt }`.

:::

::::

```
inductive Op : TileDType → TileShape → Type
  | const     (x : ℝ)                            : Op .real .scalar
  | arange    (n : Nat)                          : Op .nat  (.vec n)
  | add  {α o} (h : NumericDType α)              : Op α o → Op α o → Op α o
  | load      (region) (addr : Op .nat shape)    : Op .real shape
  | dot       …
  -- Type indices flow through every layer; no dynamic tag, no recovery step.
```


# `ComputeKernel` — proof-stating surface, projects to `AlgKernel`

`VeriTile/Triton/Float/` + `VeriTile/Triton/Core/Ast.lean`

::::cols3
:::cardBlue
*Input*

* A typed kernel that may carry `.fp32 / .int32 / .uint32` and bit-level `.const` / `.bitcast`.
* Algorithm-side statements lift via `ComputeStmt.alg`.

:::

:::cardOrange
*What we did*

* `ComputeOp` constructors: `.alg`, `.const`, `.bitcast` (width-preserving).
* Computable `Float32Bits.decodeRat` for finite-normal binary32.
* `ck.toAlgorithm?` traverses the body via `ComputeStmt.listToAlgorithm?`.
* `GapPolicy` records whether an externally-checked compute gap is required.

:::

:::cardGreen
*Output*

* `AlgKernel` (or `Except EraseDTypeError`).
* `ComputeKernel.AlgorithmCorrect`, `ComputeRefine`, `ExecCorrect`.
* Failed projection ⇒ `False` — explicit, not silent.
* User-facing theorems ride this surface; proofs run on the `AlgKernel` side.

:::

::::

```
def AlgorithmCorrect (ck : ComputeKernel) (post : AlgSpec) : Prop :=
  match ck.toAlgorithm? with
  | .ok ak  => Kernel.Correct ak post
  | .error _ => False

def ComputeCorrect (ck) (post) (gap : GapPolicy := .ignore) : Prop :=
  GapPolicy.Holds gap ∧ ProjectedCorrect ck post
  -- gap = .require contract → externally checked
```

:::cardMuted
`ComputeKernel` is the *proof-stating* Lean AST — it lets users state
theorems on fp/int kernels and have the proof flow through `toAlgorithm?` to
the algorithm side. It is *not* the differential-testing pipeline; that is
external (PLAN.md §Differential testing).

:::


# Operational semantics — `evalOp`, `stepStmt`, `exec`

`VeriTile/Triton/Semantics/`

::::cols3
:::cardBlue
*Input*

* `AlgKernel` + initial `BlockState`.
* `BlockState = mem · regs · pid · undef`.
* `regs : RegFile` dependently typed by `(dtype, shape, name)`.

:::

:::cardOrange
*What we did*

* `evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)`.
* `stepStmt`, `stepStmts`, `stepForLoopAux` — mutual.
* `TileOps`: scalar / vec / mat, broadcasting, reductions.
* Mask + bool channel; `tl.load(p, mask, other)` lowers to separate AST.
* `other = None` modeled by a per-state `undef` oracle.

:::

:::cardGreen
*Output*

* `exec : AlgKernel → BlockState → Option BlockState`.
* Deterministic except for the explicit `undef` channel.
* Single source of truth used by every theorem in the project.

:::

::::

:::cardMuted
*Notable design choice — `WithBot ℝ` carrier.* `tl.full((), -inf)` lowers
to true `⊥` — kernels using `-inf` as a max-accumulator seed need *no
input-range hypothesis*. No `-1e38` stand-in.

:::


# `forLoop_inv` — the master induction lemma

`VeriTile/Triton/LoopInvariant.lean`

::::cols3
:::cardBlue
*Input*

* A loop body of type `List Stmt`.
* A loop counter range `[0, n)`.
* An invariant `Inv : Nat → BlockState → Prop`.
* A step lemma:  `Inv k s → Inv (k+1) (run body s_with_k)`.

:::

:::cardOrange
*What we did*

* Bind the loop counter to an index register before each iteration — matches Triton's induction var.
* Support nested loops and early-exit `None` propagation.
* Pre-bake simp / cases lemmas to absorb the four known pitfalls (`forLoop_unfold` eagerness, `stepStmts []` non-rfl, `if_neg` in `∧`, match cascades).

:::

:::cardGreen
*Output*

* A reusable `forLoop_inv` theorem — the *single entry* every loop-bearing kernel proof goes through.
* Drives FA-1 forward / boundary, Welford, online-softmax, fused LayerNorm, ScoreVariants, …

:::

::::

:::cardMuted
*Why a master lemma.* Every Triton kernel of interest contains at least one
streaming reduction. Without a single, well-tested loop-induction principle,
every proof would re-invent its own recursion, and every example would
re-discover the same four pitfalls. `forLoop_inv` makes the induction step the
only thing the LLM agent has to discover.

:::


# Memory subsystem — bounds · footprint · frame · typing · view

`VeriTile/Triton/Memory/`

::::cols3
:::cardBlue
*Input*

* Kernel execution traces over a flat memory.
* Per-program region offsets and tile shapes.
* Region bounds (`RegionBounds`).

:::

:::cardOrange
*What we did*

* *Bounds* — read/write OOB safety (Layer-1).
* *Footprint* — per-kernel read/write footprints.
* *Frame* — write-footprint frame contracts; unrelated-region preservation; disjoint composition (Layer-2a).
* *Typing* — typed memory loads (int / nat / fp regions).
* *View* — `TensorView` / `programTileView` API for tile-level theorems.

:::

:::cardGreen
*Output*

* Reusable memory-safety lemmas plugged into refinement proofs.
* `TensorView.observe / loaded` for stating theorems at tile level rather than at flat addresses.
* Direct, masked, and block-pointer footprints unified.

:::

::::

```
namespace ComputeKernel
def MemorySafe (bounds : RegionBounds) (ck : ComputeKernel) : Prop := …
def ExecWritesWithin (ck) (s : BlockState) (P : WriteFootprint) : Prop := …
end ComputeKernel
```


# Concurrency framework — trace · refinement · discipline

`VeriTile/Triton/Concurrency/`

::::cols3
:::cardBlue
*Input*

* Kernels with atomic-add, async-copy, async-wait, debug-barrier, producer-consumer patterns.

:::

:::cardOrange
*What we did*

* *Trace* — vocabulary for observable concurrency events.
* *Refinement* — behavioural refinement; sequential-consistency single-layer spec.
* *Atomic* — atomic-add proof surface.
* *Discipline / ProducerConsumer* — failure markers and projection-failure correctness lemmas.

:::

:::cardGreen
*Output*

* Generalized effect projection — concurrency primitives manifest as failure markers on the projection boundary.
* Algorithmic theorems are unaffected by reorderings under the documented discipline.

:::

::::

:::cardMuted
*Current scope vs roadmap.* We currently support concurrency primitives via
*projection boundaries* and *failure markers*. The atomic-add main theorem
and async-copy serialization equivalence are roadmap items: they will move
primitives *into* the proof chain proper rather than parking them at the
boundary.

:::


# Launch — ND grid + disjoint-writes composition

`VeriTile/Triton/Launch/`

::::cols3
:::cardBlue
*Input*

* A per-program-id kernel + an ND grid shape.
* Per-program write footprints (from `Memory/Frame`).

:::

:::cardOrange
*What we did*

* *Grid* — launch every `program_id` over an ND index space.
* *Composition* — assemble per-program writes into a final whole-grid memory, given a disjoint-writes proof.
* Generic `Fin shape.length` axis — no 1D / 2D shortcuts.

:::

:::cardGreen
*Output*

* Whole-grid output state expressed as the disjoint union of per-program contributions.
* Lifts a single-program correctness proof into a full-grid correctness proof — the bridge to multi-block FA-2.

:::

::::

:::cardMuted
*Worked example.* `Examples/GridComposition.lean` exercises the Layer-2b
disjoint merge end to end, on the same `programTileView` API used by
single-program proofs.

:::


# Theorem library — what we actually prove

`VeriTile/Examples/`

::::cols
:::cardBlue
*Loop-free pairs · Tier 1*

* `softmax_kernels_refinement` (naive ↔ stable)
* `log_sum_exp_refinement` (direct ↔ shift trick)
* `softmax_reciprocal_refinement` (e/s ↔ `inv_s` · e)
* `vector_add` (multi-buffer · math correctness)

:::

:::cardOrange
*Streaming reductions · Tier 2*

* `online_softmax_recurrence_eq_batch` *(FA core)*
* `welford_kernels_refinement`
* `layernorm_kernels_refinement`
* `forLoop_inv`, mask + bool channel, typed Tile, `WithBot ℝ`

:::

::::

::::cols
:::cardGreen
*FA-1 forward · Tier 3-A*

* `fa1_forward_correct` (non-causal)
* `…_strided` / `…_strided_causal`
* `…_4D_views` / `…_4D_causal_views`
* `V1Boundary`: 8 boundary / D-tail variants
* `ScoreVariants`: ALiBi · sliding · softcap
* ~16k lines, V0 + V1Boundary + ScoreVariants + Common

:::

:::cardPurple
*Cross-cutting*

* `MemorySafety`, `MemoryFrame`
* `GridComposition` (Layer-2b)
* `FusedSiLU` (kernel-pair refinement)
* `FA-1 backward stripped` *(in progress, 1 sorry)*

:::

::::


# Proof workflow

::::cols
:::cardBlue
*1 · Write two kernels*

Original Triton kernel + an optimized rewrite, both inside the embedded
`triton { ... }` DSL.

:::

:::cardOrange
*2 · State the theorem*

Spec-side equivalence: for every tile index,
`observe(exec(kernel₁, s)) = observe(exec(kernel₂, s))`.

:::

::::

::::cols
:::cardGreen
*3 · Run `scripts/prove.sh`*

Wraps `/lean4:autoprove` (Claude Code plugin). Multi-cycle iteration,
deep-mode escalation, tactic cascades, repair patterns.

:::

:::cardPurple
*4 · CI gate: `lake build`*

Whole library 0 sorry (except FA-1 backward × 1). Artifact gate: `lake build`,
no sorry, axiom whitelist, surface check.

:::

::::

:::cardMuted
*Held-out LLM bench.* `bench/llm_eval/<tier>/` is locked *before* each
Tier — measures close-rate without tuning the tool on the eval set. N = 5
independent runs per held-out lemma, different random seeds. Cost report per
Tier: API calls, tokens, wall-clock, dollars.

:::


# Status — 2026-05-05

:::numgrid
* *80* — `.lean` files
* *~42.7k* — lines of Lean
* *0* — sorry (lib core)
* *3* — Tier milestones closed

:::

*Tiers closed.*

* *Tier 1* · `v0.1-tier1` — Loop-free kernel pairs × 3 + `welford_eq_two_pass`.
* *Tier 2* · `v0.2-tier2` — Streaming reductions × 3, `forLoop_inv`, mask + bool channel, typed Tile refactor, `WithBot ℝ` carrier.
* *Tier 3-A* · `v0.3-tier3a` (pending tag) — FA-1 forward full coverage: non-causal, strided, causal, 4D, boundary, boundaryD, score variants — ~16k lines.
* *Horizontal infra* — Algorithm/Compute split, float erasure, Memory subsystem, Concurrency framework, ND grid + composition, DSL expansion.


# Roadmap

::::cols3
:::cardBlue
*Near-term*

* Close FA-1 backward stripped (last sorry).
* Tag `v0.3-tier3a`.
* Add mask to FA-1 backward.
* Wire FA-1 backward to multi-block.

:::

:::cardOrange
*Mid-term*

* Tier 3-B FA-2 forward: `multiBlockExec`, delayed rescale, mask-skip — headline `fa1_eq_fa2` corollary.
* Tier 3-C FA backward full coverage.
* Tier 4 production batch: grouped GEMM, Mamba/SSM, RoPE, fused-norm family, GQA / MQA / MLA.

:::

:::cardGreen
*Long-term*

* Concurrency primitives in the proof chain proper (atomic-add, async-copy serialization).
* `.py` paste-in surface — Python lifter prototype.
* ND-general closure (eliminate residual 1D/2D paths).
* Effect framework polish; full block-pointer coverage.

:::

::::

:::cardMuted
No fixed time windows — VeriTile advances by Tier; main branch always builds.

:::


# Why the architecture is now stable

::::cols
:::cardBlue
*End-to-end typed pipeline*

`TileDType × TileShape` index every `Op`, `Tile`, `RegFile`, `evalOp`.
No dynamic tag, no existential recovery step. The type system *is* the
soundness backbone.

:::

:::cardOrange
*Two-layer split is permanent*

Algorithm correctness in Lean (`Kernel.Correct`); IEEE-754 fidelity via
external testing. The two never merge by Lean theorem — by design.
`ComputeKernel` is the in-Lean proof-stating bridge to the algorithm side
via `toAlgorithm?`.

:::

::::

::::cols
:::cardGreen
*Single shared interpreter*

`evalOp / stepStmt / forLoop_inv` are reused by every theorem.
Adding a new op or a new kernel has bounded blast radius.

:::

:::cardPurple
*Cross-cutting subsystems plug in cleanly*

Memory · Concurrency · Launch attach to the same `BlockState`.
New primitives extend the system without rewriting existing proofs.

:::

::::

:::cardMuted
*Architecture locked since v0.2* — what changes from here is theorem
coverage, not foundations.

:::
