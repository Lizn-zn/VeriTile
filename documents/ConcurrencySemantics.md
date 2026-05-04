# Concurrency Semantics Boundary

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
`AlgorithmCorrect` is proved here.

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
kernel has no `AlgorithmCorrect` statement in the current proof layer.

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

| Feature class | AlgorithmCorrect | Runtime / differential testing |
| --- | --- | --- |
| Deterministic and algorithm-projectable | Yes, via `toAlgorithm?` and `Kernel.Correct` | Optional |
| Atomic or reduction with algebraic abstraction | Yes, if `toAlgorithm?` projects to an abstract AlgKernel reduction marker and a theorem discharges it | Recommended |
| Async/TMA with valid sequentialization discipline | Yes, if `toAlgorithm?` projects to a sequential AlgKernel form or an abstract sequentialization marker with a theorem | Recommended |
| Non-sequential with no algorithm projection | No | Testing/runtime-only until a stronger semantics exists |

`AlgorithmCorrect` is available only when `ComputeKernel.toAlgorithm?`
succeeds and the resulting `AlgKernel` theorem is proved. Runtime or
differential testing (#58/#59) remains the validation path for compute effects
that are not fully represented by the algorithm layer.

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

## Trace Vocabulary

The first vocabulary slice lives in:

```text
VeriTile/Triton/Concurrency/Trace.lean
```

It defines `ThreadId`, `RMWOp`, `MemoryEvent`, `TraceEvent`, and `Trace`.
`MemoryEvent.rmw` records both the RMW operation and the algorithm-layer
payload value:

```lean
| rmw (region : RegionName) (offset : Nat) (op : RMWOp) (value : MemCell)
```

This is enough to state future atomic-add theorems as folds or sums over
`rmw .add` event values. The trace module is not a scheduler and does not
change `Kernel.exec`.

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

## Async / TMA Contract Slice

The async/TMA slice is currently a contract vocabulary plus a compute-facing
failure marker, not an implementation. The contract lives in:

```text
VeriTile/Triton/Concurrency/Async.lean
```

The design mirrors the `atomic_add` marker pattern. Future compute-facing
`tl.async_*` or TMA syntax should project only when a recognized discipline is
available:

```text
ComputeKernel async/TMA surface
  -> AlgKernel async/TMA sequentialization marker
  -> theorem eliminates marker into ordinary mathematical behavior
```

The current contract names the required discipline:

- every async issue has a matching wait;
- no read observes the destination before the matching wait;
- destination ownership is unambiguous within the program slice;
- overlapping destinations are explicitly ordered or rejected.

Projection failures should use a named reason such as
`requiresAsyncSequentialization`. `ComputeStmt.asyncMarker` is the first
explicit AST hook for this path: it makes async/TMA-shaped syntax representable
in the compute layer while preserving the fact that it has no
`AlgorithmCorrect` projection yet.

Explicit shared-memory state, TMA destination state, WGMMA operand layout, and
scope-tagged footprints remain #65 triggers.

## Non-Goals

This boundary document does not implement:

- `tl.atomic_*` beyond the limited `tl.atomic_add` proof-facing slice;
- async copy or TMA;
- WGMMA or warp specialization;
- shared memory or barriers;
- Iris-style or separation-logic infrastructure;
- a scheduler or interleaving semantics;
- any change to `Kernel.exec`.

The purpose is to keep the current deterministic `AlgorithmCorrect` story
stable while defining where future non-sequential effects may enter.
