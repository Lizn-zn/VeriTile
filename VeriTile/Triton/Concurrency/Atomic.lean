/-
VeriTile.Triton.Concurrency.Atomic

Proof-facing atomic-add vocabulary and grid merge hooks for #12/#66.
-/

import VeriTile.Triton.Concurrency.Trace
import VeriTile.Triton.Launch.Composition

open scoped BigOperators

namespace VeriTile.Triton

namespace MemCell

/-- Algorithm-layer Real payload used by the first atomic-add theorem slice. -/
def realPayload (cell : MemCell) : ℝ :=
  match cell.readAs .real with
  | some (some value) => value
  | _ => 0

@[simp] theorem realPayload_real (value : ℝ) :
    (MemCell.real value).realPayload = value := rfl

end MemCell

namespace Trace

/-- Real sum of trace-ordered atomic-add payloads at one cell. -/
def atomicAddRealSum (trace : Trace) (cell : MemCellAddr) : ℝ :=
  (trace.atomicAddValues cell).foldl (fun acc value => acc + value.realPayload) 0

theorem trace_atomicAdd_real_correct (trace : Trace) (cell : MemCellAddr)
    (initial : ℝ) :
    initial + trace.atomicAddRealSum cell =
      initial +
        (trace.atomicAddValues cell).foldl
          (fun acc value => acc + value.realPayload) 0 := rfl

end Trace

namespace Stmt

/-- Evaluate a store/atomic mask into an active-lane predicate. -/
noncomputable def evalMask (s : BlockState) : MaskOpt dtype shape → Option (TileIndex shape → Bool)
  | .none => some fun _ => true
  | .mask mask => (evalOp mask s).map fun masks => fun i => masks.data i
  | .maskOther mask _ => (evalOp mask s).map fun masks => fun i => masks.data i

def atomicTraceEvent (tid : ThreadId) (region : RegionName) (offset : Nat)
    (dtype : TileDType) (value : TileCarrier dtype) : TraceEvent :=
  { tid := tid, event := .rmw region offset .add (MemCell.of dtype value) }

/--
Trace events emitted by one statement in one state.

This is proof vocabulary only: it does not replace `stepStmt` and it is not a
scheduler. Non-atomic statements emit no atomic events.
-/
noncomputable def atomicTraceEvents (tid : ThreadId) (s : BlockState) : Stmt → Option Trace
  | @Stmt.atomicAdd dtype _ shape mem value mask => do
      let values ← evalOp value s
      let active ← evalMask s mask
      match mem with
      | .region region off => do
          let offsets ← evalOp off s
          some <| (TileShape.allIndices shape).filterMap fun i =>
            if active i then
              some (atomicTraceEvent tid region (offsets.data i) dtype (values.data i))
            else
              none
      | .ptr ptr => do
          let ptrs ← evalOp ptr s
          some <| (TileShape.allIndices shape).filterMap fun i =>
            let p := ptrs.data i
            if active i then
              some (atomicTraceEvent tid p.1 p.2 dtype (values.data i))
            else
              none
      | .blockPtr ptr boundaryCheck => do
          let ptrs ← evalOp ptr s
          some <| (TileShape.allIndices shape).filterMap fun i =>
            let bp := ptrs.data i
            let idx := TileShape.indexToList shape i
            if active i && bp.inBounds idx boundaryCheck then
              some (atomicTraceEvent tid bp.region (bp.address idx) dtype (values.data i))
            else
              none
  | _ => some []

/-- Relational wrapper for statement-level atomic trace extraction. -/
noncomputable def AtomicTrace (st : Stmt) (tid : ThreadId) (s : BlockState) (trace : Trace) : Prop :=
  atomicTraceEvents tid s st = some trace

end Stmt

namespace Kernel

/--
Simple statement-list atomic trace extraction.

The first slice interprets each statement under the same input state; this is
enough to expose atomic event payloads for straight-line smoke tests. Stateful
trace extraction through `stepStmt` can refine this relation later.
-/
noncomputable def AtomicTrace (k : Kernel) (tid : ThreadId) (s : BlockState) (trace : Trace) : Prop :=
  (k.body.mapM (Stmt.atomicTraceEvents tid s)).map List.flatten = some trace

/-- Sum of Real atomic-add contributions from a selected set of programs. -/
noncomputable def atomicContributionRealSum {g : Grid}
    (contributors : Finset (GridIndex g))
    (atomicTrace : GridIndex g → Trace)
    (cell : MemCellAddr) : ℝ :=
  contributors.sum fun idx => (atomicTrace idx).atomicAddRealSum cell

/--
Merge disjoint ordinary frame writes with selected atomic-add contributions.

Ordinary cells written by #49 frames keep the #49 merge behavior. Atomic-only
cells outside the ordinary write footprint are updated by adding trace payloads
to the initial Real value.
-/
noncomputable def mergeFramesWithAtomic {k : Kernel} (g : Grid) (s : BlockState)
    (frames : Kernel.GridFrames k g s)
    (contributors : Finset (GridIndex g))
    (atomicTrace : GridIndex g → Trace) : BlockState := by
  classical
  exact { s with mem := fun region offset =>
    (by
      by_cases h : Kernel.GridWriteFootprint frames (region, offset)
      · exact (Kernel.mergeFrames g s frames).mem region offset
      · exact MemCell.real
          (s.readMem region offset +
            Kernel.atomicContributionRealSum contributors atomicTrace (region, offset))) }

theorem mergeFramesWithAtomic_atomicAdd_eq_finsetSum {k : Kernel} {g : Grid}
    {s : BlockState} (frames : Kernel.GridFrames k g s)
    (contributors : Finset (GridIndex g)) (atomicTrace : GridIndex g → Trace)
    (region : RegionName) (offset : Nat)
    (hNoOrdinaryWrite : ¬ Kernel.GridWriteFootprint frames (region, offset)) :
    (Kernel.mergeFramesWithAtomic g s frames contributors atomicTrace).readMem region offset =
      s.readMem region offset +
        contributors.sum
          (fun idx => (atomicTrace idx).atomicAddRealSum (region, offset)) := by
  classical
  unfold Kernel.mergeFramesWithAtomic Kernel.atomicContributionRealSum
  simp [hNoOrdinaryWrite, BlockState.readMem]

end Kernel

end VeriTile.Triton
