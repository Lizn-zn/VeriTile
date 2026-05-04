/-
VeriTile.Triton.Concurrency.Trace

Minimal trace/interleaving vocabulary for future #12 atomic/barrier semantics.
-/

import VeriTile.Triton.MemoryFrame

namespace VeriTile.Triton

/-!
This module is intentionally only a vocabulary layer. It does not change
`Kernel.exec`, does not implement atomics, and does not define a scheduler.
The goal is to give future atomic/barrier proofs a precise object for
ordering and linearization statements.
-/

/-- Program id in a future concurrent trace. Kept non-dependent for the first slice. -/
abbrev ProgramId := List Nat

/-- Logical thread/lane identity for a trace event. -/
structure ThreadId where
  program : ProgramId
  lane : Nat
  deriving DecidableEq, Repr

/-- Read-modify-write operation tag for future atomic events. -/
inductive RMWOp where
  | add
  | max
  | min
  | xchg
  | cas
  deriving DecidableEq, BEq, Repr

/--
Memory event after projection to the algorithm layer.

The payload is a `MemCell`, not a hardware bit payload. In particular,
future `rmw .add` proofs can fold over the event values after dtype erasure.
-/
inductive MemoryEvent where
  | read (region : RegionName) (offset : Nat) (value : MemCell)
  | write (region : RegionName) (offset : Nat) (value : MemCell)
  | rmw (region : RegionName) (offset : Nat) (op : RMWOp) (value : MemCell)

namespace MemoryEvent

def region : MemoryEvent → RegionName
  | .read region _ _ => region
  | .write region _ _ => region
  | .rmw region _ _ _ => region

def offset : MemoryEvent → Nat
  | .read _ offset _ => offset
  | .write _ offset _ => offset
  | .rmw _ offset _ _ => offset

def cell (event : MemoryEvent) : MemCellAddr :=
  (event.region, event.offset)

def value : MemoryEvent → MemCell
  | .read _ _ value => value
  | .write _ _ value => value
  | .rmw _ _ _ value => value

def isRMWOp (op : RMWOp) : MemoryEvent → Prop
  | .rmw _ _ op' _ => op' = op
  | _ => False

def matchesCell (cell : MemCellAddr) (event : MemoryEvent) : Bool :=
  event.region == cell.1 && event.offset == cell.2

def rmwValue? (op : RMWOp) (cell : MemCellAddr) : MemoryEvent → Option MemCell
  | .rmw region offset op' value =>
      if region == cell.1 && offset == cell.2 && op' == op then
        some value
      else
        none
  | _ => none

@[simp] theorem cell_read (region : RegionName) (offset : Nat) (value : MemCell) :
    MemoryEvent.cell (.read region offset value) = (region, offset) := rfl

@[simp] theorem cell_write (region : RegionName) (offset : Nat) (value : MemCell) :
    MemoryEvent.cell (.write region offset value) = (region, offset) := rfl

@[simp] theorem cell_rmw
    (region : RegionName) (offset : Nat) (op : RMWOp) (value : MemCell) :
    MemoryEvent.cell (.rmw region offset op value) = (region, offset) := rfl

@[simp] theorem isRMWOp_rmw_self
    (region : RegionName) (offset : Nat) (op : RMWOp) (value : MemCell) :
    MemoryEvent.isRMWOp op (.rmw region offset op value) := rfl

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
