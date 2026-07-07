# Mapping Tilelang's Top-Most IR onto VeriTile's Core Semantics

**Status:** Research/design summary (issue #463, parent map #458).
**Date:** 2026-07-07.
**Question:** How does Tilelang's top-most (user-facing) IR map onto VeriTile's
neutral core — where do the two align, where is the *top-IR-vs-memory-manipulation*
gap, and do we lower Tilelang to the **same** algorithm layer for DSL-agnostic proofs?

This is a *mapping design*, not an implementation. It opens the "Tilelang frontend
implementation" fog named in [#458](https://github.com/Lizn-zn/VeriTile/issues/458)
and is governed by the v1 foundation decisions from
[#465](https://github.com/Lizn-zn/VeriTile/issues/465) (neutral core, **union policy**,
frontend = *syntax + lowering only*, memory-space tags, world layer — see
[`../CONTEXT.md`](../CONTEXT.md)). Read [`ArchitectureHandoff.md`](./ArchitectureHandoff.md)
(§3 the `Op`/`Stmt` GADT, §9 launch/concurrency, §10 the `triton{…}` frontend) and
[`TritonSubset.md`](./TritonSubset.md) first — the argument here is the second point on
the line those two docs start.

---

## TL;DR

1. **Tilelang is a frontend, not a second core.** Its top-most IR is a TVM
   TensorIR `PrimFunc` decorated with tile intrinsics (`T.copy`, `T.gemm`,
   `T.reduce_*`, `T.Pipelined`, explicit `T.alloc_shared`/`T.alloc_fragment`).
   Nearly every primitive that has a **clean algorithm-layer (ℝ) meaning already
   has a neutral-core `Op`/`Stmt` node** — the same nodes the Triton frontend
   lowers onto. By the union policy this is expected: the core is DSL-agnostic, so
   a second DSL that computes the same maths reuses the same nodes.
2. **The "explicit memory hierarchy" that looks like extra top-level code erases
   to value-preserving data movement.** `alloc_shared`/`alloc_fragment` + `T.copy`
   across scopes is, at the ℝ layer, *the same tile moved between buffers* — it
   lowers to the `load` / `assign` / `store` VeriTile already has, with the
   shared/fragment/global distinction demoted to a **memory-space tag** (metadata,
   Phase-2 foundation extension). The genuine gap (thread→element layouts, async
   pipelining, tensor-core precision) is exactly the **compiler/hardware layer**
   that #458 already rules future / permanently-external. So Tilelang does **not**
   widen VeriTile's semantic surface beyond what the existing roadmap anticipates.
3. **Yes — lower Tilelang to the same algorithm layer.** `T{…}` → `ComputeKernel`
   → `toAlgorithm?` → `AlgKernel`, reusing the exact
   `ComputeCorrect`/`ComputeRefine`/`…R` surfaces. Two kernels (one Triton, one
   Tilelang) that lower to the same `AlgKernel` are *literally the same proof
   obligation*; fusion (#461) and FP-warning (#462) surfaces are inherited for
   free. The one non-trivial frontend job is **raising per-index `T.Parallel`
   nests back to whole-tile `Op`s** — a lowering concern, not a semantics one.

---

## 1. What "Tilelang's top-most IR" actually is

Tilelang (`tile-ai/tilelang`) is a Python-embedded DSL. A kernel is a **TVM
TensorIR (`@T.prim_func`) function** written with a `T.*` tile API and JIT-compiled
to CUDA/HIP/LLVM. The project exposes *three interleavable levels* on one lowering
pipeline:

```
Tile Program  ──►  (+ Tile Library)  ──►  (+ Thread Primitives)  ──►  IRModule  ──►  C/CUDA/HIP/LLVM
 beginner            developer                expert                  (TIR)          hardware
```

- **Developer / Tile-Library level is the "top-most IR" this study targets:**
  `T.copy`, `T.gemm`, `T.reduce_*`, `T.Pipelined`, `T.Parallel`, explicit
  `T.alloc_shared`/`T.alloc_fragment`. This is where real FlashAttention / GEMM
  kernels are written.
- **Expert level** (raw `threadIdx`, warp shuffles, mbarriers, TMA, WGMMA,
  register control) is below the boundary v1 cares about.

Two load-bearing facts from Tilelang's own docs shape the mapping:

- **The memory hierarchy is user-written, not compiler-inferred.** "TileLang
  exposes user-facing intrinsics that map directly to physical memory spaces"
  rather than leaving placement to an opaque pass. So `alloc_shared` + `T.copy`
  *is* top-level user code — squarely the "memory-manipulation view" the issue
  names.
- **What the compiler *does* infer** is the register-fragment **layout** (the
  thread→element map for a `T.alloc_fragment` tile, via the Layout Inference pass),
  plus coalescing, vectorization, bank-swizzle, pipeline commit/wait, and sync
  barrier insertion. These are precisely the parts VeriTile does *not* model
  (it reasons over whole tiles, never per-thread layout).

Canonical GEMM skeleton (verbatim from Tilelang docs), the shape every mapping
claim below is checked against:

```python
with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
    A_s = T.alloc_shared((BM, BK), 'float16')
    B_s = T.alloc_shared((BK, BN), 'float16')
    C_f = T.alloc_fragment((BM, BN), 'float32')
    T.clear(C_f)
    for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
        T.copy(A[by*BM, ko*BK], A_s)     # global → shared
        T.copy(B[ko*BK, bx*BN], B_s)     # global → shared
        T.gemm(A_s, B_s, C_f)            # shared × shared → fragment (accumulate)
    T.copy(C_f, C[by*BM, bx*BN])         # fragment → global
```

---

## 2. Q1 — Which constructs map onto existing `Op`/`Stmt`, which need new core nodes?

The neutral core (`Triton/Core/Ast.lean`, §3 of the handoff) already carries
constants, shape ops, ND elementwise numeric/boolean, unary ℝ math,
comparisons/`where`/`ite`, `reduceMax`/`reduceSum`/`scan`/`argMax`/`sort`, `dot`,
casts, pointers/`load`, and the `Stmt` set `assign / store / atomicAdd / atomicRMW /
forLoop / forRange / forRangeDyn / ifThen / ifThenElse`. Against that:

### 2a. Direct alignment — existing nodes, no change

| Tilelang top primitive | VeriTile neutral-core node | Notes |
|---|---|---|
| `T.Kernel(gx, gy, gz, threads=)` → `(bx, by, bz)` | Launch grid: `Grid` / `GridIndex` / `Op.programId axis`; block idx = per-axis `program_id` | `threads` (block dim) is a scheduling fact with no algorithm meaning; erased. Same surface as Triton `tl.program_id`. |
| `T.ceildiv(a, b)` | `Op` cdiv (`(a+b-1)/b`), the existing `tl.cdiv` lowering | — |
| `T.serial` / `T.unroll` loops | `Stmt.forRange` / `Stmt.forLoop` | Unroll count is a schedule hint; erased (cf. Triton `static_range`). |
| `T.Pipelined(range, num_stages=)` **value semantics** | `Stmt.forRange` | `num_stages` = async producer/consumer overlap; its *value* result equals the sequential loop. Overlap correctness is the concurrency layer (§4 Tier B). |
| `T.if_then_else(c, a, b)` / Python `if/else` | `Op.where` / `Op.ite`; control `if` → `Stmt.ifThenElse` | — |
| `T.gemm(A, B, C, transpose_B=, policy=)` | `C + Op.dot(A, B)` (+ `Op.transpose` for `transpose_B`) | Exactly Triton's `tl.dot(a, b, acc)` accumulator form. `policy`/tensor-core = hardware; erased. |
| `T.reduce_sum/max/min(src, dst, dim=)` | `Op.reduceSum` / `Op.reduceMax` (axis = `dim`) | `reduce_min` via negate; `reduce_abssum` = `reduceSum ∘ abs`. |
| `T.cumsum` / `T.cummax` | `Op.scan` (sum/max, forward) | — |
| `T.exp/exp2/log/rsqrt/sigmoid/abs/max/min` | `Op.exp/exp2/log/rsqrt/sigmoid/…/max2` | ℝ elementwise math already present. |
| `T.fill(buf, v)` / `T.clear(buf)` | `Op.full shape v` written by `Stmt.assign` (`clear` = fill `0`) | Fragment/shared init = assign a constant tile to a register. |
| `T.atomic_add/max/min` | `Stmt.atomicAdd` / `Stmt.atomicRMW` | Already the Triton-side markers (single-program + grid-merge theorem). |
| `T.reshape` / `T.view` (no-copy) | `Op.reshape` / `Op.remap` | — |

### 2b. The crux — `T.copy` and `alloc_*` decompose, they do **not** add nodes

`T.copy(src, dst)` moves a tile between *any* memory scopes (global↔shared↔fragment).
At the algorithm layer a copy is **identity on the tile's mathematical content** —
it changes *where* a value lives, never *what* it is. So it lowers, by the scope of
`src`/`dst`, onto nodes VeriTile already has:

| `T.copy` direction | Lowers to | 
|---|---|
| global → shared / global → fragment | `Op.load` (region → register tile) |
| shared → fragment / fragment → shared / reg → reg | `Stmt.assign` (register tile ↔ register tile) |
| fragment → global / shared → global | `Stmt.store` (register tile → region) |

Likewise `T.alloc_shared` / `T.alloc_fragment` / `T.alloc_local` / `T.alloc_var`
allocate a **scratch `RegionName` or register-file tile**; the shared/fragment/global
distinction becomes a **memory-space tag** on that region (metadata per #465 —
visibility/safety rules may consult it; the semantic memory stays flat). A fragment
accumulator carried across a `T.Pipelined` loop is exactly a **register tile threaded
through `Stmt.forRange`** — the same shape as Triton's streaming-softmax `(m, l, O)`
accumulator, so the existing `StreamingAccumulator` mechanism (handoff §4) is reused
verbatim for FlashAttention-in-Tilelang.

**Conclusion:** no new *semantic* `Op`/`Stmt` node is required for the tile-library
surface. `T.copy`/`alloc_*` are **frontend lowering** over `load`/`assign`/`store` +
tagged regions. This is the whole content of the "top-IR vs memory-manipulation" gap
resolving in VeriTile's favour: the extra top-level memory code collapses to the
value moves the core already expresses.

### 2c. Genuine gaps

| Tilelang feature | Verdict | Where it lands |
|---|---|---|
| `T.infinity(dtype)` **+∞** | Small real gap | `WithBot ℝ` carries only −∞ (`Op.negInf` = `⊥`). FlashAttention uses **−inf only**, so covered in practice; a `+inf`-consuming kernel needs `WithTop`/`WithBot (WithTop ℝ)`. Minor, deferrable. |
| `T.async_copy` / `T.tma_copy` / `T.Pipelined` **overlap** / barriers / mbarriers / WGMMA / `T.sync_*` | Out (v1 future) | The **same async/concurrency gap already documented for Triton** (`ConcurrencySemantics.md`, #12/#409). Value semantics are in scope via the sequential loop; async overlap is the world/interleaving layer. |
| Layout/schedule annotations: `T.annotate_layout`, `T.use_swizzle`, `set_max_nreg`, warp `policy`, `coalesced_width`, eviction/L2 hints | Out (erased) | Pure performance hints with **no algorithm-layer meaning** — erased exactly like Triton `cache_modifier`/`eviction_policy`. Compiler/hardware layer, out of scope by #458. |
| `T.dp4a`, `T.gemm_sp`, tensor-core `T.ieee_*` / fast-math `T.__exp` | Out / compute-gap | Hardware instruction fidelity; belongs to the external compute-gap contract (like Triton libdevice), not an ℝ node. |

The union-policy verdict: the *only* candidate genuinely-new core node Tilelang
motivates over Triton is a `+∞` carrier, and even that is optional. Everything else
is either an existing node, frontend sugar, or a documented future/hardware layer.

---

## 3. Q2 — The named gap: user code (in scope) vs compiler/hardware (future)

Tilelang exposes explicit memory-hierarchy manipulation at the top level, where
VeriTile abstracts memory to flat regions. Split that manipulation into three tiers:

- **Tier A — *which logical tile goes where* (USER CODE, IN SCOPE).**
  `alloc_shared`/`alloc_fragment` + `T.copy` choose the staging path
  global ↔ shared ↔ fragment. At the ℝ layer this is *value-preserving data
  movement*: the copy is identity on the tile. VeriTile models it as
  `load`/`assign`/`store` between flat regions/registers, with scope as a
  **memory-space tag**. This is the same load/store the Triton frontend already
  lowers — Tilelang merely *names the intermediate buffers explicitly* where
  Triton leaves them implicit in pointer arithmetic. **In scope, no new semantics,
  frontend lowering + tags only.**

- **Tier B — *how a tile is laid out across threads / how movement overlaps*
  (COMPILER, FUTURE).** Register-fragment thread→element layout (Layout Inference),
  coalescing, vectorization, bank-swizzle, pipeline commit/wait, sync-barrier
  insertion. VeriTile deliberately reasons over **whole tiles**, never per-thread
  layout, so none of this is visible at the algorithm layer. This is the
  **compiler-layer verification** #458 lists as explicit future work; the async
  slice overlaps #12/#409's world/interleaving layer.

- **Tier C — *hardware instruction/precision semantics* (PERMANENTLY EXTERNAL /
  OUT).** Tensor-core / WGMMA accumulation precision, TMA descriptors, mbarrier
  parity, bank conflicts, register pressure. Same permanent-external stance as
  IEEE-754 (`GpuMemoryModel.md` "Not Modeled"): checked externally, never bridged
  in Lean.

**The load-bearing finding:** the *top-IR-vs-memory-manipulation* gap is almost
entirely **Tier B/C**, not Tier A. The user-visible memory-hierarchy code (alloc +
copy) is Tier A and collapses to value-preserving moves already in the core. The
part that genuinely does not fit (layouts, async overlap, tensor-core precision) is
exactly the compiler/hardware boundary #458 already fixed as future / permanently
external. Tilelang therefore **confirms** VeriTile's flat-region abstraction rather
than straining it — modulo the Phase-2 memory-space-tag extension already named on
the map.

---

## 4. Q3 — Lower to the SAME algorithm layer (DSL-agnostic proofs)?

**Yes, and this is the design's payoff.** Per the #465 union policy, a **frontend
contributes only surface syntax + lowering and owns no semantic node and no proof
surface**. So the Tilelang frontend mirrors the `triton{…}` pipeline (handoff §10):

```
T{ … }  ──elab──►  ComputeKernel.mk inputs outputs body
                        │  (same compute surface the triton{…} macro emits)
                        ▼  toAlgorithm?
                    AlgKernel  ──exec / execR──►  BlockState
                        │
                        ▼  reuse verbatim
   ComputeCorrect.Realizes / ComputeRefine.Realizes / …R   (Float/Correctness.lean, Float/Refine.lean)
```

Consequences:

- **DSL-agnostic proofs are free.** A Triton kernel and a Tilelang kernel that
  lower to the *same* `AlgKernel` present the *same* `Kernel.Correct`/`Refine` goal —
  one proof, two surfaces. The `#447` rounding-invariant `…R` surfaces, fusion
  (#461, on `ComputeRefine`), and FP-warning (#462, on the rounding model) all apply
  to Tilelang kernels with **zero additional proof machinery**.
- **The single non-trivial frontend job: raise per-index nests to whole-tile `Op`s.**
  Tilelang writes elementwise/fragment work as index loops —
  `for i, j in T.Parallel(M, N): C[i, j] = f(A[i, j], B[i, j])` — whereas VeriTile's
  `Op` is **whole-tile-valued**. The lowering must recognize a `T.Parallel` nest
  whose body is a pointwise map over fragment/shared tiles and fold it into a
  tile-shaped `Op` (`add`/`mul`/`exp`/`where`/…) — the *inverse* of what the
  Tilelang compiler does when it maps a tile op onto threads. Loops that are
  genuinely sequential (a reduction accumulator, a `T.Pipelined` K-loop) stay as
  `Stmt.forRange` with a register-tile accumulator. This raise-to-tile step is the
  frontend's main work and is a lowering concern, not a change to semantics.
- **FlashAttention reuses existing machinery.** The FA-fwd example maps cleanly:
  3-D `T.Kernel` → grid; `Q/K/V_shared`/`acc_*` fragments → scratch regions/register
  tiles; `T.gemm(Q,K,transpose_B=True)` → `Op.dot`+`Op.transpose`; the online-softmax
  `scores_max`/`logsum`/`acc_o` recurrence across `T.Pipelined` → the
  `StreamingAccumulator` `(m, l, O)` recurrence already proven for Triton FA-1.
  (`T.exp2 · scale` with `scale = log2(e)/√d` is `exp` in disguise — a frontend
  rewrite, `Op.exp`.)

---

## 5. Coverage checklist (design-time)

`Core` = existing neutral-core node reused. `Lower` = frontend sugar over existing
nodes (no new semantics). `Tag` = needs the Phase-2 memory-space-tag extension.
`Future` = async/compiler/hardware layer (out of v1 core). `Gap` = small genuine
core gap.

| Tilelang area | Verdict | Maps to / lands at |
|---|---|---|
| `T.Kernel` grid + `(bx,by,bz)` | Core | `Grid` / `GridIndex` / `Op.programId`; `threads` erased |
| `T.serial` / `T.unroll` / `T.Pipelined` (value) | Core | `Stmt.forRange` / `forLoop` |
| `T.Parallel` elementwise nest | Lower | raise to whole-tile elementwise `Op` + `Stmt.assign/store` |
| `T.if_then_else` / `if/else` | Core | `Op.where` / `Op.ite` / `Stmt.ifThenElse` |
| `T.gemm` (+`transpose_B`,`policy`) | Core | `Op.dot` (+`Op.transpose`); `policy` erased |
| `T.reduce_sum/max/min/abssum`, `T.cumsum/cummax` | Core | `Op.reduceSum/reduceMax` / `Op.scan` |
| `T.exp/exp2/log/rsqrt/sigmoid/abs/max/min` | Core | corresponding `Op.*` |
| `T.fill` / `T.clear` | Lower | `Op.full` via `Stmt.assign` |
| `T.copy` (any scope pair) | Lower | `Op.load` / `Stmt.assign` / `Stmt.store` |
| `T.alloc_shared/fragment/local/var` | Tag | scratch `RegionName` / register tile + memory-space tag |
| `T.atomic_add/max/min` | Core | `Stmt.atomicAdd` / `atomicRMW` |
| `T.reshape` / `T.view` | Core | `Op.reshape` / `Op.remap` |
| `T.ceildiv` | Core | cdiv lowering |
| `T.infinity` (+∞) | Gap | `WithBot ℝ` has −∞ only; +∞ needs `WithTop` (FA uses −inf only) |
| `T.async_copy` / `tma_copy` / `Pipelined` overlap / barriers / mbarrier / WGMMA / `sync_*` | Future | async/concurrency layer (#12/#409); same gap as Triton |
| `T.annotate_layout` / `use_swizzle` / `set_max_nreg` / warp policy / eviction hints | Future | erased performance hints (compiler/hardware) |
| `T.dp4a` / `gemm_sp` / `ieee_*` / fast-math `__exp` | Future | hardware-instruction fidelity → external compute-gap contract |

---

## 6. Design recommendations (for the frontend-implementation ticket that this fog graduates)

1. **Add Tilelang as a pure frontend** (`declare_syntax_cat` + `elab_rules`
   lowering to `ComputeKernel.mk`), parallel to `Triton/DSL/`. It owns **no**
   semantic node and **no** proof surface (union policy). Provenance ("this came
   from `T.copy`") is documentation, not types.
2. **Model `alloc_* + T.copy` as flat regions/register tiles**; attach the
   memory-space tag (Phase-2 foundation extension, #458) but keep semantic memory
   flat. Do **not** introduce a copy/alloc `Stmt` — decompose into
   `load`/`assign`/`store`.
3. **Implement the `T.Parallel`-nest → whole-tile-`Op` raise** as the frontend's
   central lowering pass; keep genuinely-sequential loops as `Stmt.forRange` with
   register-tile accumulators.
4. **Reuse the algorithm layer end-to-end** — `toAlgorithm?`,
   `ComputeCorrect`/`ComputeRefine`/`…R`, `StreamingAccumulator`, launch/grid.
   No property (correctness, fusion, FP-warning, async) needs a Tilelang-specific
   surface.
5. **Defer, don't invent:** async/pipeline overlap, layouts, tensor-core precision,
   and `+∞` are the async/compiler/hardware boundary (or a small optional carrier
   change) — none block a value-correct Tilelang frontend on the current core.

---

## 7. Where this sits on the map

- **Resolves** #463 (this study). **Opens** the "Tilelang frontend implementation"
  fog in #458 with a concrete lowering target (frontend + memory-space tags + the
  raise-to-tile pass), all landing on the neutral core Phase 1 (#472) creates.
- **Depends on** #465's neutral-core / union-policy / memory-space-tag decisions.
- **Feeds** the continuous gap analysis #458 asks for: the *top-IR-vs-memory-
  manipulation* gap is now characterized (Tier A in scope; Tier B/C future/external)
  and can be folded into the living proposal.
