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

@[simp] theorem atomicAddRealSum_nil (cell : MemCellAddr) :
    Trace.atomicAddRealSum ([] : Trace) cell = 0 := rfl

end Trace

namespace Stmt

/-- Evaluate a store/atomic mask into an active-lane predicate. -/
noncomputable def evalMask (s : BlockState) : MaskOpt dtype shape → Option (TileIndex shape → Bool)
  | .none => some fun _ => true
  | .mask mask => (evalOp mask s).map fun masks => fun i => masks.data i
  | .maskOther mask _ => (evalOp mask s).map fun masks => fun i => masks.data i

def atomicTraceEvent (tid : ThreadId) (region : RegionName) (offset : Nat)
    (dtype : TileDType) (value : TileCarrier dtype) : TraceEvent :=
  { tid := tid,
    event := .rmw
      { cell := (region, offset)
        op := .add
        input := MemCell.of dtype value } }

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

theorem atomicTraceEvents_atomicAdd_region_none
    {dtype : TileDType} (hnum : NumericDType dtype) {shape : TileShape}
    (tid : ThreadId) (s : BlockState) (region : RegionName)
    (off : Op .nat shape) (value : Op dtype shape)
    (offsets : Tile .nat shape) (values : Tile dtype shape)
    (hOffsets : evalOp off s = some offsets)
    (hValues : evalOp value s = some values) :
    Stmt.atomicTraceEvents tid s
        (Stmt.atomicAdd hnum shape (MemAccess.region region off) value MaskOpt.none) =
      some ((TileShape.allIndices shape).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid region (offsets.data i) dtype (values.data i))) := by
  simp [atomicTraceEvents, evalMask, hOffsets, hValues]

end Stmt

namespace Trace

@[simp] theorem atomicAddRealSum_single_real
    (tid : ThreadId) (region : RegionName) (offset : Nat) (value : ℝ) :
    Trace.atomicAddRealSum ([Stmt.atomicTraceEvent tid region offset .real value] : Trace)
        (region, offset) = value := by
  unfold atomicAddRealSum atomicAddValues rmwValuesAt TraceEvent.rmwValue? MemoryEvent.rmwValue?
    Stmt.atomicTraceEvent
  simp only [List.filterMap_cons, List.filterMap_nil]
  have hadd : (RMWOp.add == RMWOp.add) = true := rfl
  simp [hadd, RMWEvent.value, MemCell.realPayload, MemCell.readAs_of_same]

end Trace

namespace Kernel

/--
Simple statement-list atomic trace extraction.

The first slice interprets each statement under the same input state; this is
enough to expose atomic event payloads for straight-line smoke tests. Stateful
trace extraction through `stepStmt` can refine this relation later.
-/
noncomputable def AtomicTrace (k : Kernel) (tid : ThreadId) (s : BlockState) (trace : Trace) : Prop :=
  (k.body.mapM (Stmt.atomicTraceEvents tid s)).map List.flatten = some trace

/--
Stateful statement-list atomic trace extraction.

Unlike `Kernel.AtomicTrace`, this relation follows the same state progression
as `stepStmts`: each statement's atomic events are extracted from the state in
which that statement actually executes.  This is the proof-facing trace surface
needed when an atomic payload is computed by earlier register assignments.
-/
noncomputable def AtomicTraceStatefulList
    (tid : ThreadId) : List Stmt → BlockState → Option (Trace × BlockState)
  | [], s => some ([], s)
  | st :: rest, s => do
      let traceHead ← Stmt.atomicTraceEvents tid s st
      let s' ← stepStmt st s
      let tail ← AtomicTraceStatefulList tid rest s'
      some (traceHead ++ tail.1, tail.2)

/-- Relational wrapper for stateful statement-list atomic trace extraction. -/
noncomputable def AtomicTraceStateful
    (k : Kernel) (tid : ThreadId) (s : BlockState) (trace : Trace)
    (final : BlockState) : Prop :=
  AtomicTraceStatefulList tid k.body s = some (trace, final)

@[simp] theorem AtomicTraceStatefulList_nil (tid : ThreadId) (s : BlockState) :
    AtomicTraceStatefulList tid [] s = some ([], s) := rfl

theorem AtomicTraceStatefulList_cons_some
    {tid : ThreadId} {st : Stmt} {rest : List Stmt} {s s' final : BlockState}
    {traceHead traceTail : Trace}
    (hTraceHead : Stmt.atomicTraceEvents tid s st = some traceHead)
    (hStep : stepStmt st s = some s')
    (hTail : AtomicTraceStatefulList tid rest s' = some (traceTail, final)) :
    AtomicTraceStatefulList tid (st :: rest) s =
      some (traceHead ++ traceTail, final) := by
  simp [AtomicTraceStatefulList, hTraceHead, hStep, hTail]

theorem AtomicTraceStatefulList_append_some
    {tid : ThreadId} {l₁ l₂ : List Stmt} {s s₁ s₂ : BlockState}
    {trace₁ trace₂ : Trace}
    (h₁ : AtomicTraceStatefulList tid l₁ s = some (trace₁, s₁))
    (h₂ : AtomicTraceStatefulList tid l₂ s₁ = some (trace₂, s₂)) :
    AtomicTraceStatefulList tid (l₁ ++ l₂) s = some (trace₁ ++ trace₂, s₂) := by
  induction l₁ generalizing s trace₁ s₁ with
  | nil =>
      simp [AtomicTraceStatefulList] at h₁
      cases h₁.1
      cases h₁.2
      simpa [AtomicTraceStatefulList] using h₂
  | cons st rest ih =>
      unfold AtomicTraceStatefulList at h₁
      cases hTraceHead : Stmt.atomicTraceEvents tid s st with
      | none =>
          simp [hTraceHead] at h₁
      | some traceHead =>
          cases hStep : stepStmt st s with
          | none =>
              simp [hTraceHead, hStep] at h₁
          | some s' =>
              cases hTail : AtomicTraceStatefulList tid rest s' with
              | none =>
                  simp [hTraceHead, hStep, hTail] at h₁
              | some tail =>
                  cases tail with
                  | mk traceTail sTail =>
                      simp [hTraceHead, hStep, hTail] at h₁
                      cases h₁.1
                      cases h₁.2
                      have hRest := ih hTail h₂
                      rw [List.cons_append]
                      simpa [List.append_assoc] using AtomicTraceStatefulList_cons_some
                        (hTraceHead := hTraceHead) (hStep := hStep) (hTail := hRest)

theorem AtomicTraceStatefulList_final_eq_stepStmts
    {tid : ThreadId} {body : List Stmt} {s final : BlockState} {trace : Trace}
    (hTrace : AtomicTraceStatefulList tid body s = some (trace, final)) :
    stepStmts body s = some final := by
  induction body generalizing s trace final with
  | nil =>
      simp [AtomicTraceStatefulList] at hTrace
      cases hTrace.2
      exact stepStmts.nil
  | cons st rest ih =>
      unfold AtomicTraceStatefulList at hTrace
      cases hTraceHead : Stmt.atomicTraceEvents tid s st with
      | none =>
          simp [hTraceHead] at hTrace
      | some traceHead =>
          cases hStep : stepStmt st s with
          | none =>
              simp [hTraceHead, hStep] at hTrace
          | some s' =>
              cases hTail : AtomicTraceStatefulList tid rest s' with
              | none =>
                  simp [hTraceHead, hStep, hTail] at hTrace
              | some tail =>
                  cases tail with
                  | mk traceTail finalTail =>
                      simp [hTraceHead, hStep, hTail] at hTrace
                      cases hTrace.2
                      rw [stepStmts.cons_some hStep]
                      exact ih hTail

theorem AtomicTraceStatefulList_empty_of_stepStmts
    {tid : ThreadId} {body : List Stmt} {s final : BlockState}
    (hNoAtomic :
      ∀ st ∈ body, ∀ s0, Stmt.atomicTraceEvents tid s0 st = some [])
    (hStep : stepStmts body s = some final) :
    AtomicTraceStatefulList tid body s = some ([], final) := by
  induction body generalizing s with
  | nil =>
      simp [AtomicTraceStatefulList] at hStep ⊢
      exact hStep
  | cons st rest ih =>
      conv at hStep => lhs; unfold stepStmts
      cases hStepHead : stepStmt st s with
      | none =>
          simp [hStepHead] at hStep
      | some s' =>
          simp [hStepHead] at hStep
          have hHeadTrace : Stmt.atomicTraceEvents tid s st = some [] :=
            hNoAtomic st (by simp) s
          have hTailNoAtomic :
              ∀ st' ∈ rest, ∀ s0, Stmt.atomicTraceEvents tid s0 st' = some [] := by
            intro st' hmem s0
            exact hNoAtomic st' (by simp [hmem]) s0
          have hTail := ih hTailNoAtomic hStep
          simp [AtomicTraceStatefulList, hHeadTrace, hStepHead, hTail]

theorem AtomicTraceStateful_final_eq_exec
    {k : Kernel} {tid : ThreadId} {s final : BlockState} {trace : Trace}
    (hTrace : Kernel.AtomicTraceStateful k tid s trace final) :
    exec k s = some final := by
  exact AtomicTraceStatefulList_final_eq_stepStmts hTrace

theorem AtomicTraceStatefulList_append_of_stepPrefix
    {tid : ThreadId} {l₁ l₂ : List Stmt} {s s₁ s₂ : BlockState}
    {trace₂ : Trace}
    (hNoAtomicPrefix :
      AtomicTraceStatefulList tid l₁ s = some ([], s₁))
    (hTail : AtomicTraceStatefulList tid l₂ s₁ = some (trace₂, s₂)) :
    AtomicTraceStatefulList tid (l₁ ++ l₂) s = some (trace₂, s₂) := by
  simpa using
    (AtomicTraceStatefulList_append_some
      (tid := tid) (l₁ := l₁) (l₂ := l₂) (s := s) (s₁ := s₁) (s₂ := s₂)
      (trace₁ := []) (trace₂ := trace₂) hNoAtomicPrefix hTail)

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
