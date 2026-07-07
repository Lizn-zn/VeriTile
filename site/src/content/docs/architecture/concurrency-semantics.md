---
title: "Concurrency Semantics Boundary"
---

This document records VeriTile's boundary for non-sequential GPU effects.
It is the design entry point for issue #12.

## Current Deterministic Boundary

The proof-facing algorithm semantics is still deterministic:

```lean
Kernel.exec : Kernel -> BlockState -> Option BlockState
```

`Kernel.exec` executes one symbolic program instance by stepping statements in
order. It has no scheduler, no trace of interleavings, no in-flight operations,
and no separate shared-memory or barrier state.

The deterministic memory stack is implemented through:

- #48 active-lane bounds safety;
- #60 predicate frame contracts;
- #49 disjoint whole-grid merge;
- #61 footprint extraction helpers;
- #62 unrelated-frame helpers.

That stack supports deterministic, disjoint, sequential memory reasoning. It
does not model overlapping writes, atomics, barriers, shared memory, async/TMA,
WGMMA dispatch/wait, warp specialization, or scheduling/interleavings.

## Two-Layer Architecture

VeriTile uses two kernel layers:

```text
ComputeKernel
  -- erase dtype / erase hardware payloads -->
AlgKernel (= Kernel)
```

`ComputeKernel` is the real-ish Triton layer. It owns effects that are not
directly part of the mathematical proof layer: dtype and bit payloads today,
and future atomics, async/TMA, barriers, shared memory, WGMMA, and scheduling
annotations if they are added.

`AlgKernel` is the proof layer. It is the existing `Kernel` type with
mathematical Real/Nat/Int semantics, possibly extended later with
proof-facing abstract effect markers such as an algebraic `atomic_add`.
`Kernel.Correct` / `Kernel.Refine` are proved here; the public compute-facing
surface exposes those proofs through `ComputeKernel.ComputeCorrect` /
`ComputeKernel.ComputeRefine`.

There is no separate `ConcurrentKernel` layer in the near-term architecture.
If a future feature has a concrete need for a third layer, that should be a new
architectural issue with a specific consumer.

## Projection Semantics

The formal bridge is:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except _ AlgKernel
```

Today this bridge covers compute-facing constructs that can be projected to
the algorithm layer. As #12 features arrive, the bridge should remain mostly a
representation-erasure step:

- erase dtype or bit payloads;
- fold supported constant bitcasts;
- project hardware-ish `atomic_add` into a proof-facing abstract AlgKernel
  atomic/reduction marker when an algorithm-level interpretation is available;
- project disciplined async/TMA into proof-facing sequentialization markers or
  ordinary algorithmic loads/stores when the discipline is explicit;
- reject unsupported or nondeterministic effects.

This bridge is intentionally allowed to fail. A failed projection means the
kernel has no `ProjectedCorrect` / `ProjectedRefine` statement in the current
proof layer, so the public `ComputeCorrect` / `ComputeRefine` surface cannot
be discharged for it.

The bridge does not need to eliminate every concurrent construct. In
particular, future atomic support may project:

```text
ComputeKernel.atomic_add fp32
  --> AlgKernel.atomic_add real
```

and leave the theorem:

```text
AlgKernel.atomic_add real == mathematical sum/fold
```

to the AlgKernel proof layer. This keeps associativity/commutativity arguments
where the mathematical laws actually hold.

## Correctness Tracks

| Feature class | Projected algorithm proof / ComputeCorrect | Runtime / differential testing |
| --- | --- | --- |
| Deterministic and algorithm-projectable | Yes, via `toAlgorithm?` and `Kernel.Correct` | Optional |
| Atomic or reduction with algebraic abstraction | Yes, if `toAlgorithm?` projects to an abstract AlgKernel reduction marker and a theorem discharges it | Recommended |
| Async/TMA with valid sequentialization discipline | Yes, if `toAlgorithm?` projects to a sequential AlgKernel form or an abstract sequentialization marker with a theorem | Recommended |
| Non-sequential with no algorithm projection | No | Testing/runtime-only until a stronger semantics exists |

`ComputeCorrect` / `ComputeRefine` are available only when
`ComputeKernel.toAlgorithm?` succeeds and the resulting `AlgKernel` theorem is
proved. Optional `GapPolicy` contracts (#58/#59) record externally checked
compute-to-algorithm gaps for effects that are represented syntactically but
not internally proved as bit-level compute semantics.

## Failure Modes

Future projection failures should be distinguishable in documentation and,
when useful, in the error type:

- a dtype or bit payload cannot be erased to mathematical semantics;
- an atomic operation has no accepted AlgKernel marker or algebraic theorem;
- async/TMA lacks a valid sequentialization marker or discipline theorem;
- barrier or shared-memory behavior requires a trace model that does not exist
  yet;
- overlapping writes are not covered by disjoint merge or by an atomic/reduce
  theorem.

This document does not require refactoring the current error type. It records
the categories that future #12 implementation slices should preserve.

## Follow-Up Order

The intended implementation order is:

1. Trace/interleaving vocabulary, only as much as atomics and barriers need.
2. Atomic-only semantics or algorithm abstraction. This is the direct
   prerequisite for atomic correctness work, including #43 FA-1 backward dQ.
3. Async/TMA sequentialization theorem family.
4. WGMMA, warp specialization, and full Hopper-shaped kernels, deferred until
   stronger concurrency infrastructure exists.

## Layer Roadmap

The concurrency roadmap is intentionally layered. Each layer has a different
semantic object, so later layers should extend earlier ones without forcing
their machinery into simpler proofs.

| Layer | Semantic object | Covers | Tracking |
| --- | --- | --- | --- |
| L1: single-cell linearized RMW | `MemCell -> RMWEvent -> Option (MemCell × RMWEvent)` plus a per-cell ordered event list | `atomic_add`, `atomic_xchg`, `atomic_cas`, single-cell max/min/and/or/xor | #66 for commutative add; #82 for order-sensitive xchg/cas |
| L2: multi-cell atomic transaction | state transformer over multiple cells | DCAS / MCAS / transactional-memory style primitives | no active issue; open when a real consumer appears |
| L3: cross-cell ordering + async | happens-before / visibility graph plus fences, barriers, and async completion | memory ordering, async copy, TMA/WGMMA visibility, producer-consumer warp specialization | #12 long-horizon |
| L4: lock-free data-structure invariants | L1/L3 plus a data-structure invariant relating abstract state to memory | queue / stack / set / lock-free protocols | no active issue; consumer-driven only |

#82 implements only L1. It must remain compatible with later L2/L3/L4 work,
but it must not depend on multi-cell transactions, a global scheduler, async
visibility, or data-structure invariants.

## Trace Vocabulary

The first vocabulary slice lives in:

```text
VeriTile/Concurrency/Trace.lean
```

It defines `ThreadId`, `RMWOp`, `MemoryEvent`, `TraceEvent`, and `Trace`.
`MemoryEvent.rmw` carries the shared `RMWEvent` payload:

```lean
structure RMWEvent where
  cell : MemCellAddr
  op : RMWOp
  input : MemCell
  extraInput : Option MemCell := none
  observed : Option MemCell := none
  result : Option MemCell := none

inductive MemoryEvent where
  | read (region : RegionName) (offset : Nat) (value : MemCell)
  | write (region : RegionName) (offset : Nat) (value : MemCell)
  | rmw (event : RMWEvent)
```

Commutative atomics use `input` as their contribution and leave the optional
fields empty. Order-sensitive atomics such as xchg/cas can fill `extraInput`,
`observed`, and `result` without introducing a second event type. The trace
module is not a scheduler and does not change `Kernel.exec`.

## #82 PR0 Audit Result

#82's first implementation step is an ownership/API audit before adding
`atomic_xchg` / `atomic_cas` semantics. The current result is:

```text
API ready; proceed to PR1.
```

Audit details:

- `PermissionModel` / `OwnershipMap` are abstract over a permission type.
  They are not hard-coded to `WarpId` ownership in the public discipline
  predicates.
- `RMWEvent` is already the single extensible RMW payload. The trace layer does
  not need a parallel CAS/XCHG event type.
- `MemoryEvent.rmw` already carries `RMWEvent`, so adding `.xchg` / `.cas`
  semantics does not require a constructor-shape refactor.
- `Stmt.atomicAdd` remains a separate public surface for #66's commutative
  theorem. #82 should add a new return-valued RMW constructor instead of
  overloading `Stmt.atomicAdd`.
- `Trace.LinearizesAt` already provides a per-cell trace hook. #82 may add a
  generalized event-list linearization predicate, but does not need a global
  scheduler or timestamp map.

## Grid Launchers

Two relational launchers turn per-program kernel correctness into whole-grid
correctness:

- **`Kernel.GridLaunchedOrdinary k g s sFinal`** — for ordinary
  (atomic-free) kernels. Says: starting from `s`, running `k` on every
  grid index in `g` produces `sFinal`, with per-program writes
  pairwise-disjoint. The composition of per-program frame writes is the
  load-bearing fact; `mergeFrames` does the merge.
- **`Kernel.GridLaunchedRMW k g s sFinal linearization`** — for kernels
  that include atomic RMW operations on a single cell. Combines ordinary
  frame writes with an explicit per-cell linearization witness. Used by
  the atomic slice below.

Lifting per-program correctness to a `ForAllProgramsSome k g s post`
statement gives "for every grid index `idx`, exec succeeds and the
postcondition holds with `s.withGridIndex idx`". A grid-level theorem
typically wraps a per-program ComputeCorrect via this `ForAllProgramsSome`
shape.

## Atomic Add Slice

The first concrete atomic slice is intentionally narrow:

- `tl.atomic_add` lowers to a proof-facing `Stmt.atomicAdd` marker;
- single-program `stepStmt` performs a sequential read-add-write update;
- `Stmt.atomicTraceEvents` records active lanes as `MemoryEvent.rmw ... .add value`;
- `Kernel.mergeFramesWithAtomic` combines #49 ordinary frame writes with
  selected grid-level atomic traces;
- `Kernel.mergeFramesWithAtomic_atomicAdd_eq_finsetSum` states the Real
  final-cell theorem as initial value plus a `Finset.sum` of trace payloads.

This is `Limited` atomic support, not a full concurrent executor.
`tl.atomic_xchg` and `tl.atomic_cas` now project to the return-valued
`Stmt.atomicRMW` algorithm marker, with the single-cell RMW semantics living in
`RMWOp.apply` / `RMWTrace.applyLinearized`. They also have executable
single-program statement semantics, stateful trace emission through
`Stmt.atomicTraceEvents` / `Kernel.AtomicTraceStateful`, and a single-cell
whole-grid launcher relation (`Kernel.GridLaunchedRMW`) that combines ordinary
frame writes with an explicit per-cell linearization witness. Other Triton
atomic surfaces (`tl.atomic_max`, `tl.atomic_min`, `tl.atomic_and`,
`tl.atomic_or`, and `tl.atomic_xor`) currently lower to
`ComputeStmt.effectMarker` and fail projection with `requiresEffectProjection`.

## Async / TMA Contract Slice

The async/TMA slice is currently a documented contract plus compute-facing
failure markers, not an implementation.

The design mirrors the `atomic_add` marker pattern. Future compute-facing
`tl.async_*` or TMA syntax should project only when a recognized discipline is
available:

```text
ComputeKernel async/TMA surface
  -> AlgKernel async/TMA sequentialization marker
  -> theorem eliminates marker into ordinary mathematical behavior
```

The contract names the future required discipline:

- every async issue has a matching wait;
- no read observes the destination before the matching wait;
- destination ownership is unambiguous within the program slice;
- overlapping destinations are explicitly ordered or rejected.

Projection failures use the named reason `requiresEffectProjection`.
`ComputeStmt.effectMarker` is the explicit AST hook for this path: it makes
async/TMA-shaped syntax representable in the compute layer while preserving
the fact that it has no `ProjectedCorrect` projection yet. The DSL surfaces
`tl.async_copy(dst, src)`,
`tl.async_wait()`, and `tl.debug_barrier()` currently lower to this failure
marker; they are not executable semantics and they do not imply shared-memory,
barrier, or TMA modeling.

Explicit shared-memory state, TMA destination state, WGMMA operand layout, and
scope-tagged footprints remain #65 triggers.

## Non-Goals

This boundary document does not implement:

- executable `tl.atomic_*` beyond the limited `tl.atomic_add` proof-facing slice;
- async copy or TMA;
- WGMMA or warp specialization;
- shared memory or barriers;
- Iris-style or separation-logic infrastructure;
- a scheduler or interleaving semantics;
- any change to `Kernel.exec`.

The purpose is to keep the current deterministic `ComputeCorrect` /
`ProjectedCorrect` story stable while defining where future non-sequential
effects may enter.
