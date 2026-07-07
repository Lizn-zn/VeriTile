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

/- Embed a static asset (lives in `static_files/`) as a slide-width image.
   Use this for diagrams authored in Figma / Excalidraw / hand-written SVG,
   instead of recreating the visuals as CSS-positioned divs. -/
block_component +directive image (src : String) (alt : String) where
  toHtml id _ _ _ _ := do
    pure {{<img id={{id}} class="diagram" src={{"static/" ++ src}} alt={{alt}}/>}}

/- A horizontal pipeline. The `flow` directive only carries the rail label
   and its colour; everything inside is a sequence of `:::stage` directives,
   each fully describing its own block (border, body colour, shadow). The
   stage's body is regular markdown — backticks for `<code>`, etc.
   `static_files/slides.css` only owns the layout (flex row, arrows). -/
block_component +directive flow (label : String) (railColor : String) where
  toHtml id _ _ goB contents := do
    let style := s!"--flow-rail-color: {railColor};"
    pure {{<div class="flow" id={{id}} style={{style}}>
            <div class="rail">{{label}}</div>
            {{← contents.mapM goB}}
          </div>}}

/- `width` is a CSS flex-basis value (`"auto"`, `"12vmin"`, `"120px"`, …) —
   each stage sets its own width independently. `accent` overrides only the
   top border (e.g. `"0.4vmin solid var(--blue)"`); pass `""` to leave the
   top border equal to the rest. `weight` is a CSS font-weight ("400", ...). -/
block_component +directive stage
    (width : String) (border : String) (color : String) (shadow : String)
    (weight : String) (accent : String) where
  toHtml id _ _ goB contents := do
    let base := s!"flex: 0 0 {width}; border: {border}; color: {color}; box-shadow: {shadow}; font-weight: {weight};"
    let style := if accent.isEmpty then base else s!"{base} border-top: {accent};"
    pure {{<div class="stage" id={{id}} style={{style}}>{{← contents.mapM goB}}</div>}}

end


#doc (Page) "VeriTile : Triton, Proven" =>

— Automated Verification for Triton GPU kernels in Lean 4


# Worked Example — Stable Softmax

`Triton source  →  embedded in Lean 4  →  theorem proved by an LLM.`

::::cols3

:::cardBlue
*1 · Triton source*

```
"""
Stable Softmax (row-wise, masked).
  x, y  : input/output ptr (N floats)
  N     : runtime row length
  BLOCK : compile-time block size (≥ N)
  out   : y[i] = exp(x[i]) / Σ exp(x[j])
    (via `y`, no return — GPU kernel)
"""
@triton.jit
def softmax(x, y, N, BLOCK: tl.constexpr):
  pid  = tl.program_id(0)
  cols = tl.arange(0, BLOCK)
  mask = cols < N
  offs = pid * N + cols
  a    = tl.load(x + offs,
    mask=(cols < N), other=-float("inf"))
  m    = tl.max(a, axis=0)
  e    = tl.exp(a - m)
  s    = tl.sum(e, axis=0)
  tl.store(y + offs, e / s, mask=mask)
```

The actual `.py` kernel a Triton user ships.

:::

:::cardOrange
*2 · Embed the kernel · State the spec*

```
/- Embed in Lean 4 — `triton { ... }` -/
def stable_sm (xReg yReg : RegionName)
    (N BLOCK : Nat) : Kernel := triton {
  <- paste the .py body here ->
}

/- Spec — if xReg holds the input xs,
   then after running stable_sm,
   yReg[i] equals softmax(xs)[i]
   at every row index `i`. -/
theorem stable_sm_correct
  (xReg yReg : RegionName) (N BLOCK : Nat)
  (hN : 0 < N) (hBN : N ≤ BLOCK)
  (s : BlockState) (xs : Fin N → ℝ)
  (h : InputLoadedAt s xReg N xs) :
  ∀ i : Fin N,
    observeAt
    (exec (stable_sm xReg yReg N BLOCK) s)
    yReg N s.pid i
    = some (softmaxSpec xs i)
```

Mechanical translation, fully automated.

:::

:::cardGreen
*3 · Theorem + LLM-generated proof*

```
/- Prove the correctness of stable sm.
   The produced formal proof is
   checked by Lean 4 theorem prover -/
theorem stable_sm_correct (...) := by
  -- 0. Open the recursion on `0 < N`.
  obtain ⟨n, rfl⟩ :=
    Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  -- 1. Reduce the masked kernel exec at
  --    row index `i` to its per-row spec.
  have hRed :=
    stable_sm_reduce_at xReg yReg
      (n+1) BLOCK hN hBN s xs h i
  rw [hRed]
  -- 2. Unfold the spec helpers.
  unfold stableSpec softmaxSpec
  -- 3. Apply the math identity:
  --    stable = softmax (after row-max).
  have hEq := stable_eq_softmax xs hN
  exact congrArg some (congrFun hEq i)
```

Agent generates the proof, fully automated.

:::

::::


# The problem with optimizing Triton kernels today

`Performance up · correctness on a prayer.`

:::image "today-vs-veritile.svg" "Today's workflow vs VeriTile's workflow"
:::

:::cardMuted
Engineers iterate fast — naive → stable → online → fused — but each
rewrite ships on a sample-test prayer. A single `mask=` / `other=` slip,
partial KV block, or wrong reduction axis can pass sampling and quietly
break the production hot path. We see this as a verification gap, and
VeriTile is our attempt at it.

:::


# Features — what we do, what we don't

`Bounded by what we prove · intentional in what we leave to testing.`

::::cols
:::cardBlue
*We do this*

* *Proving-ensured correctness* — every optimization is a Lean theorem,
  mechanically checked against its math reference.
* *Triton-faithful semantics* — full `mask=` / `other=`, ND broadcasting,
  reductions, `forLoop_inv` master induction.
* *Triton-user friendly DSL* — `triton { ... }` reads like `.py`,
  region-polymorphic, kwargs supported.
* *LLM-driven proofs* — `scripts/prove.sh` → `/lean4:autoprove`,
  multi-cycle iteration, deep-mode escalation.

:::

:::cardMuted
*We explicitly don't*

* Trust differential testing alone — sampling can't cover the algorithm.
* Cherry-pick a "simplified Triton" subset — fidelity to real Triton
  is mandatory.
* Hand-write proofs for every kernel — the LLM does it.
* Formalize IEEE-754 — out of Lean scope by design; the compute layer
  validates fp behavior empirically.

:::

::::

*What we do is bounded; what we don't is intentional.*


# Two-layer architecture: Lean proof  ⟂  external testing

::::cols
:::cardBlue
*Algorithm layer · Lean proof*

`ProjectedCorrect ck post`

Real / Int / Nat semantics. No IEEE rounding, no NaN.

* `eraseDType`: `fp32 → ℝ`, `int32 → ℤ`, `uint32 → ℕ`.
* Computable bit-pattern decoder for `fp32` constants.
* Existing `Kernel.Correct` proofs reused on the algorithm side.

Catches *algorithm-structure bugs* — every wrong recurrence, wrong
masking, wrong reduction is a failed Lean proof.

:::

:::cardOrange
*Compute layer · External testing*

Compute-gap checking for required contracts.

`ComputeKernel` behavior ↔ projected `AlgKernel` behavior, ε-bound match.

* Per-op / per-kernel `ComputeGapContract`.
* Recorded in Lean through `GapPolicy`.
* *Not a Lean theorem* — an externally checked bridge by design.

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

`VeriTile/Frontend/Triton/DSL/`

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

`VeriTile/Core/`

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

`VeriTile/Float/` + `VeriTile/Core/Ast.lean`

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
* `ComputeKernel.ComputeCorrect`, `ComputeKernel.ComputeRefine`, `ExecCorrect`.
* Failed projection ⇒ `False` — explicit, not silent.
* User-facing theorems ride this surface; proofs run on the `AlgKernel` side.

:::

::::

```
def ProjectedCorrect (ck : ComputeKernel) (post : AlgSpec) : Prop :=
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
external (PLAN.md §Compute-gap details).

:::


# Operational semantics — `evalOp`, `stepStmt`, `exec`

`VeriTile/Semantics/`

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

`VeriTile/LoopInvariant.lean`

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

`VeriTile/Memory/`

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

`VeriTile/Concurrency/`

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


# How async fits — three rooms, one final-state contract

`Triton async is not reducible to pure sequential — we model it without sequentializing it.`

::::cols3
:::cardBlue
*Front desk · DSL accepts*

`tl.async_copy(...)` / `tl.async_wait()` / `tl.debug_barrier()` parse and elaborate like any other op.

Each becomes a `ComputeStmt.effectMarker "tl.async_copy"` — opaque, named, carried on the AST. Triton users paste `.py` unchanged.

:::

:::cardOrange
*Kitchen · algorithm layer rejects*

`toAlgorithm?` on an `effectMarker` returns `Except.error (.requiresEffectProjection op)`.

The pure-sequential proof chain refuses to execute async — silently sequentializing would lie about Triton's semantics. *Fidelity over cleanliness.*

:::

:::cardGreen
*Side door · concurrency framework*

`VeriTile/Concurrency/` builds an `EffectTrace` (memory ∪ async ∪ barrier), derives `HappensBefore`, and states correctness as `RefinesSequential concurrent sequential` on final memory.

:::

::::

:::cardMuted
*Four discipline conditions unlock the side door* —
`AsyncVisibility` (every `wait` follows a `copy`),
`BarrierDiscipline` (every `barrier.wait` follows an `arrive` of the same name),
`OwnershipTransfer` (footprint permissions transfer cleanly),
`NoRaceOnOwnedCells` (no concurrent writers on owned cells).
Once these are proved, *concurrent ≡ sequential on final memory* follows by refinement.

:::


# Where the async approach comes from

`Each piece is textbook — the value is composing them on production Triton kernels in Lean 4.`

::::cols
:::cardBlue
*Building blocks · standard names*

* *Concurrent Separation Logic* — O'Hearn / Brookes 2004. Permissions, ownership transfer, no-race.
* *Fractional permissions* — Boyland 2003. Generalizes our `ExclusivePerm` to shared-read.
* *Lamport happens-before* — 1978. Trace-order + async / barrier edges.
* *DRF-SC theorem* — Adve 1990. Race-free programs admit a sequential-consistency reading — the shape of our `RefinesSequential`.
* *Reduction theorems* — Lipton 1975, Cohen-Lamport 1998. `async_copy + async_wait` ↔ synchronous load/store as movers.

:::

:::cardOrange
*GPU verification predecessors*

* *GPUVerify* (Betts / Chong / Donaldson) — two-thread reduction + barrier invariants for CUDA.
* *VerCors* — permission-based GPU reasoning.
* *PUG* — CUDA verification in Boogie.
* *IronFleet · CertiKOS · Iris-based OS verification* — algorithm layer pure-sequential, concurrency attached via refinement: the same shape we use.

:::

::::

:::cardMuted
*Compact label.* "Permission-based Concurrent Separation Logic + DRF-style sequentialization refinement." VeriTile's contribution is not the underlying logic — it is *applying this stack* to *real production Triton kernels* in *Lean 4*. Prior work used the same parts on toy CUDA in Boogie / Why3.

:::


# Launch — ND grid + disjoint-writes composition

`VeriTile/Launch/`

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
* `Boundary`: 8 boundary / D-tail variants
* `ScoreVariants`: ALiBi · sliding · softcap
* ~16k lines, Core + Boundary + ScoreVariants + Common

:::

:::cardPurple
*Cross-cutting*

* `MemorySafety`, `MemoryFrame`
* `GridComposition` (Layer-2b)
* `FusedSiLU` (kernel-pair refinement)
* `FA-1 backward stripped` + atomic dQ surface

:::

::::


# User-facing theorem surface

::::cols
:::cardBlue
*Correctness · one kernel*

`ComputeCorrect.Realizes` states the observable write map of one kernel:

```
write : ι → Option MemCellAddr
expected : ι → α
```

`some addr` checks a written cell; `none` is inactive.

:::

:::cardOrange
*Refinement · two kernels*

`ComputeRefine.Realizes` compares two write maps from the same logical output
index:

```
lhsWrite rhsWrite : ι → Option MemCellAddr
relation : ι → α → β → Prop
```

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

Use `ComputeCorrect.Realizes` for a kernel ↔ spec theorem, or
`ComputeRefine.Realizes` for a kernel-pair relation. Most masked outputs use
`WriteMap.writeIf mask addr`.

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

Whole library 0 sorry. Artifact gate: `lake build`,
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
* *111* — `.lean` files under `VeriTile/`
* *~48.2k* — lines of Lean under `VeriTile/`
* *0* — sorry
* *3* — Tier milestones closed

:::

*Tiers closed.*

* *Tier 1* · `v0.1-tier1` — Loop-free kernel pairs × 3 + `welford_eq_two_pass`.
* *Tier 2* · `v0.2-tier2` — Streaming reductions × 3, `forLoop_inv`, mask + bool channel, typed Tile refactor, `WithBot ℝ` carrier.
* *Tier 3-A* · `v0.3-tier3a` — FA-1 forward full coverage: non-causal, strided, causal, 4D, boundary, boundaryD, score variants — ~16k lines.
* *Horizontal infra* — Algorithm/Compute split, float erasure, Memory subsystem, Concurrency framework, ND grid + composition, DSL expansion.


# Roadmap

::::cols3
:::cardBlue
*Near-term*

* Maintain `v0.3-tier3a` as the Tier 3-A checkpoint tag.
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
