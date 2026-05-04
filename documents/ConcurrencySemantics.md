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
  -- erase dtype / abstract effects / sequentialize if possible -->
AlgKernel (= Kernel)
```

`ComputeKernel` is the real-ish Triton layer. It owns effects that are not
directly part of the mathematical proof layer: dtype and bit payloads today,
and future atomics, async/TMA, barriers, shared memory, WGMMA, and scheduling
annotations if they are added.

`AlgKernel` is the proof layer. It is the existing `Kernel` type with
mathematical Real/Nat/Int semantics. `AlgorithmCorrect` is proved here.

There is no separate `ConcurrentKernel` layer in the near-term architecture.
If a future feature has a concrete need for a third layer, that should be a new
architectural issue with a specific consumer.

## Projection Semantics

The formal bridge is:

```lean
ComputeKernel.toAlgorithm? : ComputeKernel -> Except _ AlgKernel
```

Today this bridge covers compute-facing constructs that can be projected to
the algorithm layer. As #12 features arrive, the same bridge may also:

- erase dtype or bit payloads;
- fold supported constant bitcasts;
- abstract `atomic_add` into a mathematical monoid reduction when accepted
  algebraic laws exist;
- sequentialize disciplined async/TMA into ordinary algorithmic loads and
  stores;
- reject unsupported or nondeterministic effects.

This bridge is intentionally allowed to fail. A failed projection means the
kernel has no `AlgorithmCorrect` statement in the current proof layer.

## Correctness Tracks

| Feature class | AlgorithmCorrect | Runtime / differential testing |
| --- | --- | --- |
| Deterministic and algorithm-projectable | Yes, via `toAlgorithm?` and `Kernel.Correct` | Optional |
| Atomic or reduction with algebraic abstraction | Yes, if `toAlgorithm?` produces an algorithm reduction spec | Recommended |
| Async/TMA with valid sequentialization discipline | Yes, if `toAlgorithm?` succeeds under the discipline condition | Recommended |
| Non-sequential with no algorithm projection | No | Testing/runtime-only until a stronger semantics exists |

`AlgorithmCorrect` is available only when `ComputeKernel.toAlgorithm?`
succeeds and the resulting `AlgKernel` theorem is proved. Runtime or
differential testing (#58/#59) remains the validation path for compute effects
that are not fully represented by the algorithm layer.

## Failure Modes

Future projection failures should be distinguishable in documentation and,
when useful, in the error type:

- a dtype or bit payload cannot be erased to mathematical semantics;
- an atomic operation has no accepted algebraic abstraction;
- async/TMA lacks a valid sequentialization discipline;
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

## Non-Goals

This boundary document does not implement:

- `tl.atomic_*`;
- async copy or TMA;
- WGMMA or warp specialization;
- shared memory or barriers;
- Iris-style or separation-logic infrastructure;
- a scheduler or interleaving semantics;
- any change to `Kernel.exec`.

The purpose is to keep the current deterministic `AlgorithmCorrect` story
stable while defining where future non-sequential effects may enter.
