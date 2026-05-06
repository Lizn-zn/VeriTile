/-
VeriTile.Triton.Concurrency.Trace

Minimal trace/interleaving vocabulary for future #12 atomic/barrier semantics.
-/

import VeriTile.Triton.Memory.Frame

namespace VeriTile.Triton

/-!
This module is intentionally only a vocabulary layer. It does not change
`Kernel.exec`, does not implement atomics, and does not define a scheduler.
The goal is to give future atomic/barrier proofs a precise object for
ordering and linearization statements.
-/

/-- Program id in a future concurrent trace. Kept non-dependent for the first slice. -/
abbrev ProgramId := List Nat

/-- Warp identity for future ownership/permission disciplines. -/
abbrev WarpId := Nat

/-- Abstract permission interface for future ownership-transfer reasoning.

The first concrete instance is exclusive ownership; richer fractional /
invariant-carrying permissions for CAS/XCHG can instantiate the same interface
later without changing theorem statements. -/
class PermissionModel (Perm : Type) where
  valid : Perm → Prop
  disjoint : Perm → Perm → Prop
  combine : Perm → Perm → Option Perm
  combine_disjoint_valid :
    ∀ p q, disjoint p q → ∀ pq, combine p q = some pq →
      valid p ∧ valid q → valid pq

/-- Future ownership map over memory cells, parameterized by the permission model. -/
abbrev OwnershipMap (Perm : Type) [PermissionModel Perm] := MemCellAddr → Option Perm

/-- Current simple ownership instance: one warp owns a cell exclusively. -/
structure ExclusivePerm where
  owner : WarpId
  deriving DecidableEq, Repr

instance : PermissionModel ExclusivePerm where
  valid _ := True
  disjoint _ _ := False
  combine _ _ := none
  combine_disjoint_valid _ _ hdisjoint _ _ _ := False.elim hdisjoint

/-- Logical thread/lane identity for a trace event. -/
structure ThreadId where
  program : ProgramId
  lane : Nat
  deriving DecidableEq, Repr

/-- Extensible read-modify-write event payload.

Commutative atomics use `input` as their contribution and leave the optional
fields empty. CAS/XCHG-style operations can later fill `extraInput`,
`observed`, and `result` without changing the trace event constructor. -/
structure RMWEvent where
  cell : MemCellAddr
  op : RMWOp
  input : MemCell
  extraInput : Option MemCell := none
  observed : Option MemCell := none
  result : Option MemCell := none

namespace RMWEvent

def region (event : RMWEvent) : RegionName :=
  event.cell.1

def offset (event : RMWEvent) : Nat :=
  event.cell.2

def value (event : RMWEvent) : MemCell :=
  event.input

def withObservedResult (event : RMWEvent) (old : MemCell) : RMWEvent :=
  { event with observed := some old, result := some old }

end RMWEvent

namespace RMWOp

/--
Apply one linearized single-cell RMW event.

This is algorithm-layer semantics. In particular, CAS compares `MemCell`s by
mathematical equality, not by a hardware bit-pattern predicate. Bit-level
fidelity stays in the compute-gap/testing layer.

For this first order-sensitive slice, `.xchg` and `.cas` are modeled directly.
The existing `.add` proof surface remains the #66 sum theorem; `.max`, `.min`,
and bitwise atomics are left for their own commutative-family follow-up.
-/
noncomputable def apply : RMWOp → MemCell → RMWEvent → Option (MemCell × RMWEvent) := by
  classical
  intro op old event
  cases op with
  | xchg =>
      exact some (event.input, event.withObservedResult old)
  | cas =>
      match event.extraInput with
      | none => exact none
      | some replacement =>
          let event' := event.withObservedResult old
          if old = event.input then
            exact some (replacement, event')
          else
            exact some (old, event')
  | add => exact none
  | max => exact none
  | min => exact none

@[simp] theorem apply_xchg (old input : MemCell) (cell : MemCellAddr) :
    RMWOp.apply .xchg old
        { cell := cell, op := .xchg, input := input } =
      some (input,
        { cell := cell, op := .xchg, input := input,
          observed := some old, result := some old }) := rfl

@[simp] theorem apply_cas_missing_replacement
    (old cmp : MemCell) (cell : MemCellAddr) :
    RMWOp.apply .cas old
        { cell := cell, op := .cas, input := cmp } = none := rfl

theorem apply_cas_success
    (old replacement : MemCell) (cell : MemCellAddr) :
    RMWOp.apply .cas old
        { cell := cell, op := .cas, input := old,
          extraInput := some replacement } =
      some (replacement,
        { cell := cell, op := .cas, input := old,
          extraInput := some replacement,
          observed := some old, result := some old }) := by
  simp [apply, RMWEvent.withObservedResult]

theorem apply_cas_failure
    (old cmp replacement : MemCell) (cell : MemCellAddr)
    (h : old ≠ cmp) :
    RMWOp.apply .cas old
        { cell := cell, op := .cas, input := cmp,
          extraInput := some replacement } =
      some (old,
        { cell := cell, op := .cas, input := cmp,
          extraInput := some replacement,
          observed := some old, result := some old }) := by
  simp [apply, RMWEvent.withObservedResult, if_neg h]

end RMWOp

namespace RMWTrace

/--
Apply a per-cell linearization list.

The input list is the chosen linearization order. The returned list is in the
same order, with `observed` / `result` fields filled by `RMWOp.apply`.
-/
noncomputable def applyLinearized :
    MemCell → List RMWEvent → Option (MemCell × List RMWEvent)
  | initial, [] => some (initial, [])
  | initial, event :: events => do
      let (cell', event') ← RMWOp.apply event.op initial event
      let (final, events') ← applyLinearized cell' events
      some (final, event' :: events')

@[simp] theorem applyLinearized_nil (initial : MemCell) :
    applyLinearized initial [] = some (initial, []) := rfl

@[simp] theorem applyLinearized_cons
    (initial next final : MemCell)
    (event event' : RMWEvent) (events events' : List RMWEvent)
    (hStep : RMWOp.apply event.op initial event = some (next, event'))
    (hTail : applyLinearized next events = some (final, events')) :
    applyLinearized initial (event :: events) =
      some (final, event' :: events') := by
  simp [applyLinearized, hStep, hTail]

theorem applyLinearized_xchg_two
    (old first second : MemCell) (cell : MemCellAddr) :
    applyLinearized old
        [ { cell := cell, op := .xchg, input := first }
        , { cell := cell, op := .xchg, input := second } ] =
      some (second,
        [ { cell := cell, op := .xchg, input := first,
            observed := some old, result := some old }
        , { cell := cell, op := .xchg, input := second,
            observed := some first, result := some first } ]) := by
  simp [applyLinearized]

theorem applyLinearized_cas_success
    (old replacement : MemCell) (cell : MemCellAddr) :
    applyLinearized old
        [ { cell := cell, op := .cas, input := old,
            extraInput := some replacement } ] =
      some (replacement,
        [ { cell := cell, op := .cas, input := old,
            extraInput := some replacement,
            observed := some old, result := some old } ]) := by
  simp [applyLinearized, RMWOp.apply_cas_success]

theorem applyLinearized_cas_failure
    (old cmp replacement : MemCell) (cell : MemCellAddr)
    (h : old ≠ cmp) :
    applyLinearized old
        [ { cell := cell, op := .cas, input := cmp,
            extraInput := some replacement } ] =
      some (old,
        [ { cell := cell, op := .cas, input := cmp,
            extraInput := some replacement,
            observed := some old, result := some old } ]) := by
  simp [applyLinearized, RMWOp.apply_cas_failure, h]

end RMWTrace

/--
Memory event after projection to the algorithm layer.

The payload is a `MemCell`, not a hardware bit payload. In particular,
future `rmw .add` proofs can fold over the event values after dtype erasure.
-/
inductive MemoryEvent where
  | read (region : RegionName) (offset : Nat) (value : MemCell)
  | write (region : RegionName) (offset : Nat) (value : MemCell)
  | rmw (event : RMWEvent)

namespace MemoryEvent

def region : MemoryEvent → RegionName
  | .read region _ _ => region
  | .write region _ _ => region
  | .rmw event => event.region

def offset : MemoryEvent → Nat
  | .read _ offset _ => offset
  | .write _ offset _ => offset
  | .rmw event => event.offset

def cell (event : MemoryEvent) : MemCellAddr :=
  (event.region, event.offset)

def value : MemoryEvent → MemCell
  | .read _ _ value => value
  | .write _ _ value => value
  | .rmw event => event.value

def isRMWOp (op : RMWOp) : MemoryEvent → Prop
  | .rmw event => event.op = op
  | _ => False

def matchesCell (cell : MemCellAddr) (event : MemoryEvent) : Bool :=
  event.region == cell.1 && event.offset == cell.2

def rmwValue? (op : RMWOp) (cell : MemCellAddr) : MemoryEvent → Option MemCell
  | .rmw event =>
      if event.cell.1 == cell.1 && event.cell.2 == cell.2 && event.op == op then
        some event.value
      else
        none
  | _ => none

@[simp] theorem cell_read (region : RegionName) (offset : Nat) (value : MemCell) :
    MemoryEvent.cell (.read region offset value) = (region, offset) := rfl

@[simp] theorem cell_write (region : RegionName) (offset : Nat) (value : MemCell) :
    MemoryEvent.cell (.write region offset value) = (region, offset) := rfl

@[simp] theorem cell_rmw (event : RMWEvent) :
    MemoryEvent.cell (.rmw event) = event.cell := rfl

@[simp] theorem isRMWOp_rmw_self
    (event : RMWEvent) :
    MemoryEvent.isRMWOp event.op (.rmw event) := rfl

end MemoryEvent

/-- One event in a future interleaving trace. -/
structure TraceEvent where
  tid : ThreadId
  event : MemoryEvent

abbrev Trace := List TraceEvent

namespace TraceEvent

def cell (event : TraceEvent) : MemCellAddr :=
  event.event.cell

def value (event : TraceEvent) : MemCell :=
  event.event.value

def isRMWOp (op : RMWOp) (event : TraceEvent) : Prop :=
  event.event.isRMWOp op

def matchesCell (cell : MemCellAddr) (event : TraceEvent) : Bool :=
  event.event.matchesCell cell

def rmwValue? (op : RMWOp) (cell : MemCellAddr) (event : TraceEvent) : Option MemCell :=
  event.event.rmwValue? op cell

end TraceEvent

/-- Async event vocabulary for future producer-consumer pipelines. -/
inductive AsyncEvent where
  | copy (src dst : MemCellAddr)
  | wait
  deriving DecidableEq, Repr

/-- Barrier event vocabulary for future ordering/ownership-transfer proofs. -/
inductive BarrierEvent where
  | arrive (name : String)
  | wait (name : String)
  deriving DecidableEq, Repr

/-- General effect event for the #81 concurrency framework.

The existing `Trace` remains the memory/atomic trace used by #66. `EffectTrace`
is the broader producer-consumer vocabulary for async/barrier reasoning. -/
inductive EffectEvent where
  | memory (event : MemoryEvent)
  | async (event : AsyncEvent)
  | barrier (event : BarrierEvent)

/-- One event in a future effect trace. -/
structure EffectTraceEvent where
  tid : ThreadId
  event : EffectEvent

abbrev EffectTrace := List EffectTraceEvent

namespace EffectTrace

/-- Event `a` appears before event `b` in an effect trace. -/
def OccursBefore (trace : EffectTrace) (a b : EffectTraceEvent) : Prop :=
  ∃ (pref mid suff : EffectTrace),
    trace = pref ++ a :: mid ++ b :: suff

/-- First-slice happens-before relation: explicit trace order.

Future discipline layers can strengthen this with program-order, async-wait, and
barrier edges without changing the consumer theorem surface. -/
def HappensBefore (trace : EffectTrace) (a b : EffectTraceEvent) : Prop :=
  OccursBefore trace a b

theorem happensBefore_of_occursBefore {trace : EffectTrace} {a b : EffectTraceEvent}
    (h : OccursBefore trace a b) :
    HappensBefore trace a b := h

end EffectTrace

namespace Trace

/-- Events in a trace that touch a concrete memory cell. -/
def eventsAt (trace : Trace) (cell : MemCellAddr) : List TraceEvent :=
  trace.filter (TraceEvent.matchesCell cell)

/-- RMW events with operation `op` touching a concrete memory cell. -/
def rmwValuesAt (trace : Trace) (op : RMWOp) (cell : MemCellAddr) : List MemCell :=
  trace.filterMap (TraceEvent.rmwValue? op cell)

/-- Payloads contributed by future `atomic_add` events at `cell`, in trace order. -/
def atomicAddValues (trace : Trace) (cell : MemCellAddr) : List MemCell :=
  trace.rmwValuesAt .add cell

/--
`a` occurs before `b` in `trace`.

This is a predicate over list prefixes rather than an executable scheduler.
-/
def OccursBefore (trace : Trace) (a b : TraceEvent) : Prop :=
  ∃ (pref mid suff : Trace),
    trace = pref ++ a :: mid ++ b :: suff

/--
`values` is the linearized payload sequence for RMW events at `cell`.

The trace list order is the linearization order; this predicate gives future
atomic proofs a stable hook without introducing a scheduler in this slice.
-/
def LinearizesAt (trace : Trace) (op : RMWOp) (cell : MemCellAddr)
    (values : List MemCell) : Prop :=
  values = trace.rmwValuesAt op cell

theorem linearizesAt_rmwValuesAt (trace : Trace) (op : RMWOp) (cell : MemCellAddr) :
    LinearizesAt trace op cell (trace.rmwValuesAt op cell) := rfl

theorem atomicAddValues_eq_rmwValuesAt (trace : Trace) (cell : MemCellAddr) :
    trace.atomicAddValues cell = trace.rmwValuesAt .add cell := rfl

end Trace

end VeriTile.Triton
