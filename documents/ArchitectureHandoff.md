# VeriTile — Whole-Repo Architecture Handoff

A fine-grained map of the entire codebase as it stands on `main` (tip `bbb31d2a`,
2026-07-06), written for a collaborator picking the project up. It covers the existing
implemented architecture *and* the newest work (the `#447` rounding model). Names are the
real Lean identifiers — grep for them.

This synthesizes and cross-links the existing design docs
([`CodeOrganization`](./CodeOrganization.md), [`CorrectnessSurfaces`](./CorrectnessSurfaces.md),
[`EraseDType`](./EraseDType.md), [`MemorySafety`](./MemorySafety.md),
[`ConcurrencySemantics`](./ConcurrencySemantics.md), [`KernelManifest`](./KernelManifest.md),
[`ProofConventions`](./ProofConventions.md)) plus [`PLAN.md`](../PLAN.md); read those for
depth on any one axis.

---

## 0. What VeriTile is, in one paragraph

VeriTile embeds a typed Triton-style kernel DSL in Lean 4 (`triton { ... }`), gives it an
operational semantics over a typed operator GADT `Op : TileDType → TileShape → Type`, and
proves each kernel either **correct** against a mathematical spec or **refining** another
kernel — via the `ComputeCorrect.*` / `ComputeRefine.*` theorem surfaces. Proofs run on an
erased **real-valued** algorithm layer; concrete floating-point numeric behavior is handled
separately (see §1).

## 1. The one load-bearing design decision: two-layer verification

Everything downstream follows from this split (locked since v0.2, see `PLAN.md`):

```
Algorithm layer  (Lean proof)              Compute-gap layer (external testing)
─────────────────────────────              ────────────────────────────────────
ProjectedCorrect ck spec                   GapPolicy.require contract
  := ck.toAlgorithm? = ok ak                 := Python check of ComputeKernel vs
   ∧ Kernel.Correct ak spec                     projected AlgKernel within tolerance
  over ℝ / ℤ / ℕ. No IEEE                     recorded in Lean as
  rounding, NaN, denormals.                    `ExternalChecked contract`.
```

**The two layers are deliberately *not* connected by an internal Lean theorem.** A
`ComputeCorrect` certificate means "the projected algorithm structure over ℝ/ℤ/ℕ is
Lean-proved correct", plus (if `gap := .require`) "the named numeric gap was externally
validated". IEEE-754 formalization is *permanently* out of Lean scope by design.

> **Where `#447` fits.** The new rounding model (§8) is the *first internal* handle on
> rounding — but it stays consistent with the above by being **abstract**: it captures the
> *structure* of rounding events (how many, where) without asserting anything about their
> *magnitude*. Magnitude remains the external gap's job. See §8.

## 2. Layered architecture (bottom-up; each layer depends only on those below)

```
  bench/tritonbench_g/<k>/, VeriTile/Examples/   per-kernel glue: triton{…}, *Spec, theorem
  ─────────────────────────────────────────────
  Float/Correctness.lean                         PUBLIC theorem surfaces (Realizes, …)
  Float/{RoundingModel,EvalOpR,StepR,Refine}     #447 rounding-invariant surfaces  ← newest
  Memory/{View,Frame,Typing,Bounds,Footprint}    memory *reasoning* (framing, safety, views)
  Launch/*, Concurrency/*                         grid launch + atomics/async gap
  ─────────────────────────────────────────────
  Semantics/{EvalOp,Step,State,…}                evalOp / stepStmt / exec + BlockState
  DSL/{Syntax,Expansion/*,Typing,Inference}      triton{…} macro → ComputeKernel
  Core/{Types,Shape,Ast}                          the typed AST: Op / Stmt / Kernel
  Triton/Math/*, VeriTile/Math/*                  pure ℝ operators + Mathlib identities
```

The **three-layer discipline** (from `CodeOrganization.md`) governs where new code goes:
pure `(Fin N → ℝ) → …` math → `Triton/Math/`; anything generic over a kernel's
`(load, mask)` touching `Tile`/`WithBot` → `Semantics/` (named by *mechanism*, not
operator); anything touching `BlockState`/`RegionName`/layout → per-kernel glue.

---

## 3. Core AST — `Triton/Core/`

The typed substrate. A well-typed expression is a term of `Op dtype shape`, so illegal
dtype/shape combinations are unrepresentable.

**`Core/Types.lean` — dtypes & carriers**
- `TileDType` — closed dtype universe: `.real .fp32 .fp16 .bf16 .int .nat .bool .ptr .blockPtr`.
- `TileCarrier : TileDType → Type` — the Lean value per dtype. All four float channels carry
  `WithBot ℝ` (so `-∞`/`tl.full(-inf)` is representable and propagates through `max`); `.int→Int`,
  `.nat→Nat`, `.bool→Bool`, `.ptr→RegionName×Nat`, `.blockPtr→BlockPtr`.
- `Region (d)` / `RegionName := Region .real` — named buffer with phantom element dtype; `RegName := String`.
- `BlockPtr` — layout-explicit block pointer (`region, baseOffset, parentShape, blockShape, strides, offsets`) with `.address`, `.inBounds`, `.advance`.
- `FloatDType` (the 4 float tags) with `toTileDType`, `ofReal`, `storeValue`, `cast`.
- `NumericDType / IntegralDType / ComparableDType` — witness GADTs gating which dtypes may be added / floor-divided / compared.

**`Core/Shape.lean` — shapes, indices, tiles**
- `TileShape := List Nat` (outermost-first; scalar = `[]`). `TileIndex : TileShape → Type` (dependent index).
- `Tile dtype shape := { data : TileIndex shape → TileCarrier dtype }` — a tile is just its indexing function; `@[ext]` pointwise. Constructors `Tile.scalar/vec/mat`.
- Shape algebra (`axisDim, eraseAxis, reduceShape, allIndices, linearIndex, …`) and `Broadcast a b out` — the GADT of legal binary broadcasts, with `leftIndex/rightIndex` back-projections.

**`Core/Ast.lean` — the operator GADT + statements + kernels**
- `Op : TileDType → TileShape → Type` — **doubly-indexed GADT** (`mutual` with `MemAccess`, `MaskOpt`); the index *is* the typing. Families: constants/nullary (`const, constFloat, programId, ref, arange`), shape ops (`broadcast, full, reshape, remap, join, split, expandDim, transpose`), elementwise numeric carrying a witness + `Broadcast` proof (`add/sub/mul/div/floorDiv/mod`, bitwise), unary real math (`exp, log, sigmoid, sqrt, rsqrt, tanh, erf, …`), comparisons/select (`lt…ne, where, ite, max2`), reductions/scans (`reduceMax, reduceSum, scan, argMax, sort`), `dot`, casts (`castFloat, castNatToInt, natToReal, …`), and pointers/memory (`ptrAdd, makeBlockPtr[Dyn], advanceBlockPtr, load`).
  - `MemAccess d shape` — `.region`/`.ptr`/`.blockPtr … boundaryCheck` address forms.
  - `MaskOpt` — `.none`/`.mask`/`.maskOther`.
- `Stmt` (non-dependent, effectful): `assign, store, atomicAdd (NumericDType), atomicRMW (RMWOp = add/max/min/xchg/cas), forLoop, forRange, forRangeDyn, ifThen, ifThenElse`.
- `Kernel` (= `AlgKernel`) — `{ inputs outputs : List RegionName, body : List Stmt }`. **Memory is a single global map at runtime**; `inputs/outputs` are metadata.
- Also here: the compute wrapper (`ComputeDType, ComputeOp, ComputeExpr, ComputeStmt, ComputeKernel`, `EraseDTypeError`, bit structs `UInt32Bits/Int32Bits/Float32Bits`) — see §5.

## 4. Semantics — `Triton/Semantics/` (umbrella `Semantics.lean`)

The operational model. Everything is `Option`-valued (⟂ = undefined read/out-of-bounds).

- **`State.lean` — `BlockState`**, the execution state: `mem : RegionName → Nat → MemCell` (total, dynamically-typed cells; `MemCell := Sigma TileCarrier`), `regs : RegFile` (dtype+shape-indexed typed register file — a name is only observable at the dtype/shape it was written), `pids : Nat → Nat` (per-axis `program_id`; `pid := pids 0`), `undef` (oracle for `mask=`/`other=None` loads). Primitives: `setReg`, `writeMem/writeMemAs/writeMemTyped/writeCell`, `readMem/readMemAs/readMemTyped/readMemValue`, `atomicRMWAt`.
- **`EvalOp.lean` — `evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)`** — the denotational expression evaluator (noncomputable).
- **`Step.lean` — `stepStmt`, `stepStmts`, loop aux, `exec`.** `stepStmt` maps `assign→setReg`; `store`/`atomicAdd` fold over `TileShape.allIndices` writing active lanes via `writeMemTyped`; loops unroll via `stepForLoopAux`/`stepForRangeAux` (well-founded on `stop-cur`); `if` delegates to `stepStmts`. **`exec k s := stepStmts k.body s`** is the per-`program_id` runner.
- Mechanism helper files (named by *mechanism*, per `CodeOrganization.md`): `TiledIndexing` (program/lane→tile index), `MaskedReduction` (masked-lane reduction bridges), `AtomicReduction`, `StreamingAccumulator` (online-softmax `(m,l,O)` recurrence), `BlockPtrEval`, `BroadcastReshape` (+ the `tile_elementwise` simp attribute), `Offset`, `TileOps`, `Scalar`.

## 5. Compute surface & the projection — `Core/Ast.lean` + `EraseDType.md`

Two kernel surfaces and a projection between them:

- **`Kernel` / `AlgKernel`** — the algorithm layer (§3); what `exec` runs and proofs reason about.
- **`ComputeKernel.mk inputs outputs body`** — the compute-facing surface the `triton{…}` macro always emits. `body : List ComputeStmt` mixes `ComputeStmt.alg (Stmt)` (pure algorithm statements) with compute-only nodes (`ComputeExpr`, bit-level `ComputeOp`/`bitcast`, `llrint`, `effectMarker`, `opaque`). `ComputeDType = uint32/int32/fp32`.
- **`ComputeKernel.toAlgorithm? : ComputeKernel → Except EraseDTypeError AlgKernel`** — the projection. Recursively lowers each `ComputeStmt`/`ComputeExpr`; erases bit dtypes (`uint32→nat, int32→int, fp32→real`), decoding constant bit-payloads through the single authoritative decoder `Float32Bits.decodeRat`; genuinely compute-only nodes (runtime `bitcast`, `effectMarker`, `opaque`) **fail** with `requiresComputeSemantics` rather than inventing algorithm meaning.
- `ComputeKernel.eval s := match toAlgorithm? with | ok ak => exec ak s | error => none`. `eval_eq_exec_of_toAlgorithm?` is a *definitional unfold*, **not** a soundness theorem — there is by design no Lean bridge from bit-level compute execution to algorithm execution.
- The `EOut`/`computeTerm` mechanism keeps concrete `fp32` tags on the compute view while the algorithm view uses `.real` (for projectable accumulator/typed-load cases). Wider promotion, fp16/bf16 rounding, and overflow remain compute-gap contracts.

## 6. Correctness surfaces — `Float/Correctness.lean` (the public API)

The predicates live here (not in `Kernel/`). Bottom to top:
- `Kernel.Correct k post := ∀ s s', exec k s = some s' → post s s'` — postcondition-style.
- `Kernel.Refine lhs rhs rel` — two kernels from the same `s`; finals satisfy `rel`.
- `ComputeKernel.{AlgorithmCorrect, ProjectedCorrect, ComputeCorrect}` (+ `…Refine`) — dispatch on `toAlgorithm?` (`ok → Kernel.Correct`, `error → False`); `ComputeCorrect` adds the `GapPolicy` obligation (`.ignore` default, `.require contract` for test-backed gaps).
- **Ergonomic user surfaces** (`CorrectnessSurfaces.md` is the pick-list): `ComputeCorrect.Realizes` (one kernel ↔ output spec via a `WriteMap ι := ι → Option MemCellAddr` + `OutputReadable`), `ComputeRefine.Realizes` (kernel pair), plus `OutputScalar/OutputArray/OutputTile/OutputPair[Where]`, `Post`, `General`. `WriteMap.writeIf` states masked stores; `realizes_writeIf_iff` is the standard unfolding step.
- Transition lemmas `computeCorrect_of_toAlgorithm_eq` / `computeCorrect_of_toAlgKernel rfl` discharge the projection so proofs proceed on `Kernel.Correct`.
- `ComputeCorrectAt?`/`…At?` — ε-bounded observation predicates reserved for future test-backed IEEE claims; deliberately unused by the ℝ proofs.

## 7. Memory reasoning — `Triton/Memory/`

Primitives live in `Semantics/State.lean` (§4); `Memory/` is the *reasoning* layer on top.
- **`Memory/View.lean`** — `TensorView shape := { region, base, strides }` with `offset` (strided address map), `loaded`/`InputAt` ("region contains tensor `xs` under this map"), `Valid` (non-overlap), `offset_injective`; address-map builders `Offset.strided/contig/rowMajor2D` + injectivity lemmas; `IndirectView`/`pagedAddr` for gather/paged-KV.
- **`Memory/Frame/`** — framing: `WriteFootprint := MemCellAddr → Prop`; `WriteWithin P s₀ s₁` ("`s₁` differs from `s₀` only inside `P`"); the frame lemma library (`mem_eq_of_not_written`, `writeWithin_trans`, `foldl_*_writeWithin`); per-statement bounds (`stepStmt_store_writeWithin`); `Kernel.ExecFrame`/`ExecWritesWithin` and the `ComputeKernel.ExecWritesWithin` projection.
- **`Memory/Typing/`** — region typing: executable checker `Op.check`/`Stmt.check`/`check`/`checkStrict : … → Except CheckError` with pointer `provenance`, and the propositional counterpart `*.RespectsRegionTyping` the checker soundness targets.
- **`Memory/Bounds.lean` / `Footprint.lean`** — `MemorySafe` (in-bounds access) for `Op`/`MemAccess`/`Stmt`/`Kernel`/`ComputeKernel`; `tileImage`/`activeAddrTileImage` connecting a tile's address image to a `WriteFootprint` (the disjointness input for grid composition, §9).

See `MemorySafety.md` and `GpuMemoryModel.md` for the intended discipline.

## 8. `#447` Float rounding model — `Triton/Float/` (the newest work, PR #456)

The latest body of work on `main`. Adds a rounding-invariance layer *alongside* (never
modifying) the existing evaluator/stepper. Full deep-dive: it is worth reading the four
files directly; the essentials:

- **`RoundingModel.lean`** — `structure RoundingModel { round : FloatDType → ℝ → ℝ, round_real : round .real = id }`: one opaque rounding fn per float dtype, constrained *only* on the mathematical `.real` channel. `RoundingModel.triv` (every `round = id`) **is** today's semantics. Two — and only two — rounding-event sites: `RoundingModel.cast` (float cast, rounds at destination; used by `evalOpR`'s `castFloat`) and `RoundingModel.storeValue` (narrowing store, rounds at buffer dtype; used by `StepR`'s `writeMemAsR`). Opt-in property mixins as typeclasses, never global axioms: `IdemRounding`, `MonoRounding`, `OddRounding`, `GridNested` — both `.triv` and real round-to-nearest-even satisfy all four.
- **`EvalOpR.lean`** — `evalOpR R` = `evalOp` threading `R`, changing exactly the `Op.castFloat` arm. Degeneration `evalOpR_triv : evalOpR .triv = evalOp`.
- **`StepR.lean`** — `stepStmtR R`/`execR R` = `stepStmt`/`exec` with expressions through `evalOpR R` and stores through `writeMemTypedR R`. Degeneration chain `stepStmtR_triv`/`execR_triv`. (Scope carve-out: `atomicRMW` inputs still evaluate under `.triv` — flagged follow-up.)
- **`Refine.lean`** — the surfaces: `Kernel.CorrectR`/`ComputeKernel.{evalR, ComputeCorrectR, ExecCorrectR}` (each mirrors a §6 surface with `exec→execR`, each with a `_triv_iff` degeneration), `RoundingModel.Representable` (= "rounding fixes it"), and the headline `ComputeRefine.RealizesR` — realization against a spec `expected : RoundingModel → ι → α` that is a *function of the model*, so the spec's syntax encodes the rounding-event structure. The single bridge `RealizesR.toRealizes` instantiates `R := .triv` and rides the degeneration chain to recover the ℝ theorem for free. Named shapes: `RoundFree` (gating-GEMM), `SingleRounding` (fused-SwiGLU).

**Payoff:** every existing `ComputeCorrect.*` theorem is the `.triv` shadow of a stronger,
rounding-invariant `ComputeRefine.*R` theorem — gained without rewriting proofs and without
a concrete IEEE encoding. (Consumers like the SwiGLU pilot live on branch
`feat/447-swiglu-pilot`, **not yet on `main`**.)

## 9. Launch & Concurrency — the grid + the compute-to-algorithm gap

**`Triton/Launch/`** — `Grid := { dims }`, `GridIndex g` (typed ND program index) with
`toPids`; `BlockState.withGridIndex` instantiates a program's pids. Grid theorems today use
`Kernel.ForAllProgramsSome` (per-program-local: every grid index succeeds + postcondition).
`Composition.lean` composes disjoint programs: `GridWritesDisjoint` (from disjoint
`tileImage`s), `mergeFrames`, `GridLaunchedOrdinary`, and the general `LaunchCorrect`.
There is **no whole-grid `launchExec` and no `num_programs` def yet** — known gaps.

**`Triton/Concurrency/`** (intentionally the weaker "first" surface, `#81`) —
`Trace.lean`: the event vocabulary (`MemoryEvent`, `TraceEvent`, `AsyncEvent`,
`BarrierEvent`, `EffectTrace`, `HappensBefore`, `PermissionModel`/`OwnershipMap`).
`Atomic.lean`: atomics proven at the *algorithm* level — `trace_atomicAdd_real_correct`,
and the key concurrent composition `mergeFramesWithAtomic_atomicAdd_eq_finsetSum` (merge
per-program frames with atomic accumulation into one final state).
`Refinement.lean`: `ConcurrentTrace.RefinesSequential` — "every concurrent run has a
sequential run with equal final memory" (explicitly weaker than bisimulation).
`Discipline.lean`/`ProducerConsumer.lean`: async/barrier *discipline* predicates
(`OwnershipValid`, `NoRaceOnOwnedCells`, `BarrierDiscipline`) and a one-tile
double-buffered-copy-refines-sequential example. **No operational interleaving semantics or
general bisimulation exists yet** — async/barrier concurrency is modeled abstractly and
connected to semantics only through `RefinesSequential`. See `ConcurrencySemantics.md`.

## 10. DSL — the `triton { ... }` macro — `Triton/DSL/`

Compile-time lowering pipeline: parse → infer → expand → build.
- **`Syntax.lean`** — `declare_syntax_cat tritonExpr/tritonStmt/…` parser categories; the single user entry `syntax "triton" "{" tritonStmt* "}" : term`.
- **`Expansion/Main.lean`** (macro core) — `elab_rules` for `triton{…}` is THE entry point: runs pre-passes, `expandStmts`, emits `ComputeKernel.mk`. `expandExpr : Env → tritonExpr → MacroM EOut` is the recursive lowerer, dispatching into `Expansion/{Compute,Memory,Control}.lean` (one handler per `tl.*` family: `expandArith/expandReduce/expandDot/expandLoad/expandStore/expandAtomicAdd/expandProgramId/expandArange/…`).
- **`Expansion/Common.lean`** — `EOut` (carries both algorithm `term` and optional fp32 `computeTerm`), `CInfo` compute-dtype tag, coercion helpers.
- **`Typing.lean`/`Inference.lean`/`Metadata.lean`** — macro-time `DInfo`/`SInfo` dtype+shape tags and `Env`; pre-passes that infer which identifiers are `.nat` offsets vs `.real` data (so `tl.load` gets the right default dtype) and auto-populate `inputs`/`outputs` by scanning for `tl.load`/`tl.store`.

## 11. Pure math & proof-reuse libraries

- **`Triton/Math/`** — operator-named pure ℝ files (`Activation, Reduction, Softmax, LogSumExp, L2Norm, Loss, …`): `(Fin N → ℝ) → …` definitions + non-trivial identities (e.g. naive = stable softmax, Welford = two-pass), depending only on Mathlib. Inclusion rule: ≥2 callers and mathematically substantive.
- **`VeriTile/Math/`** — heavier certified analysis (e.g. `GeluTaylor20*` polynomial bound certificate chain, `RealErf`, `Tanh`).
- **`Triton/Kernel/`** — *proof-reuse* lemma libraries (NOT kernel definitions): `LoopInvariant.lean` (`forLoop_inv` master induction lemma), `Matmul.lean`, `OffsetInjective.lean`, `ScatterStore.lean`, `EvalHelpers.lean` (`cdiv` + `evalOp` unfold lemmas).

## 12. Verification harness — `bench/`, `scripts/`, CI

- **`bench/tritonbench_g/<kernel>/`** — TritonBench-G ports (≈184 kernel dirs on `main`), each with the `triton{…}` transcription, kernel-local `*Spec`/`*Load`/`*Offset`, and the correctness theorem. Regenerated human-readable summaries live in **`bench/tritonbench_g/_spec-sheets/`** (renamed from `spec-sheets/` in #446).
- **`scripts/kernel-manifest.tsv`** — the canonical registry (≈151 proven rows). Schema `id·file·theorem·kind·status·source·source_ref·config·label·notes`; `kind ∈ {correct refine math launch trace safety frame}`, `status ∈ {proven projected test-gap blocked smoke}`. See `KernelManifest.md`.
- **CI gate** — `lake build` + `scripts/check-artifact.sh` (**no `sorry`**, axiom whitelist, manifest schema, doc-term drift) + `bench/check_ports.sh` (per-port build). Bench-audit now runs on a self-hosted 96-core runner (#455).
- **LLM proof wrapper** — `scripts/prove.sh` automates the standard proof loop (`computeCorrect_of_toAlgKernel rfl` → `simp` reduce `exec` → close on a Mathlib identity).
- **External gap checker** — the Python side validating `GapPolicy.require` contracts (compute vs projected-algorithm within tolerance); this is the other half of §1, not in Lean.

---

## 13. Reading path for a new collaborator

1. `PLAN.md` §"Verification architecture" — the two-layer split (§1 here). Non-negotiable.
2. `Core/Types.lean` + `Core/Ast.lean` — the `Op` GADT; internalize that types encode typing.
3. `Semantics/State.lean` + `Step.lean` — `BlockState` and `exec`.
4. `Float/Correctness.lean` + `CorrectnessSurfaces.md` — the theorem surfaces you'll actually state.
5. One worked kernel end-to-end, e.g. `VeriTile/Examples/VectorAdd.lean` (correct) and `SoftmaxEq.lean` (refine).
6. `EraseDType.md` — the compute→algorithm projection, once you hit an fp32/bitcast case.
7. `Float/{RoundingModel,Refine}.lean` — the newest layer (§8), if you touch rounding.

## 14. Known gaps / active follow-ups (flagged in-source or in `PLAN.md`)

- Whole-grid `launchExec` and `num_programs` do not exist yet (§9).
- Concurrency has no operational interleaving / general bisimulation — only `RefinesSequential` (§9).
- `#447` rounding-aware `atomicRMW` inputs are follow-up; SwiGLU consumers not yet on `main` (§8).
- IEEE-754 numeric semantics are permanently external (test-backed), by design (§1).
- README "15 kernels" / "Repository Layout" counts are stale — the manifest (§12) is the source of truth.
