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

def atomicRMWTraceEvent (tid : ThreadId) (event : RMWEvent) : TraceEvent :=
  { tid := tid, event := .rmw event }

private noncomputable def foldAtomicRMWTraceIndices
    {dtype : TileDType} {shape : TileShape}
    (tid : ThreadId) (op : RMWOp)
    (regionFn : TileIndex shape → RegionName)
    (offsetFn : TileIndex shape → Nat)
    (inputFn : TileIndex shape → TileCarrier dtype)
    (extraFn : Option (TileIndex shape → TileCarrier dtype))
    (active : TileIndex shape → Bool)
    (indices : List (TileIndex shape))
    (s : BlockState) : Option (BlockState × Trace) := by
  classical
  induction indices generalizing s with
  | nil =>
      exact some (s, [])
  | cons i rest ih =>
      if active i then
        let extra := extraFn.map (fun f => f i)
        match s.atomicRMWAt op dtype (regionFn i) (offsetFn i) (inputFn i) extra with
        | none => exact none
        | some (s', _, event) =>
            match ih s' with
            | none => exact none
            | some (final, tail) => exact some (final, atomicRMWTraceEvent tid event :: tail)
      else
        exact ih s

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
  | @Stmt.atomicRMW op _ shape mem input extraInput mask _ => do
      let inputs ← evalOp input s
      let extras ←
        match extraInput with
        | none => some none
        | some extra => (evalOp extra s).map some
      let active ← evalMask s mask
      let extraFn := extras.map fun extraTile => fun i : TileIndex shape => extraTile.data i
      let (_, trace) ←
        match mem with
        | .region region off => do
            let offsets ← evalOp off s
            foldAtomicRMWTraceIndices tid op (fun _ => region) (fun i => offsets.data i)
              (fun i => inputs.data i) extraFn active (TileShape.allIndices shape) s
        | .ptr ptr => do
            let ptrs ← evalOp ptr s
            foldAtomicRMWTraceIndices tid op (fun i => (ptrs.data i).1) (fun i => (ptrs.data i).2)
              (fun i => inputs.data i) extraFn active (TileShape.allIndices shape) s
        | .blockPtr ptr boundaryCheck => do
            let ptrs ← evalOp ptr s
            foldAtomicRMWTraceIndices tid op
              (fun i => (ptrs.data i).region)
              (fun i => (ptrs.data i).address (TileShape.indexToList shape i))
              (fun i => inputs.data i) extraFn
              (fun i =>
                active i && (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
              (TileShape.allIndices shape) s
      some trace
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

theorem atomicTraceEvents_atomicRMW_xchg_real_scalar_const
    (tid : ThreadId) (s : BlockState) (region : RegionName)
    (input : ℝ) :
    Stmt.atomicTraceEvents tid s
        (Stmt.atomicRMW .xchg .real []
          (MemAccess.region region (Op.constNat 0))
          (Op.const input) none MaskOpt.none none) =
      some
        [Stmt.atomicRMWTraceEvent tid
          { cell := (region, 0)
            op := .xchg
            input := MemCell.real input
            observed := some (s.mem region 0)
            result := some (s.mem region 0) }] := by
  rfl

/-- Trace emitted by an unmasked region-addressed `atomic_add` whose offset and
payload tiles are read from registers. -/
theorem atomicTraceEvents_atomicAdd_region_none_of_reg_refs
    {dtype : TileDType} (hnum : NumericDType dtype) {shape : TileShape}
    (tid : ThreadId) (s : BlockState) (region : RegionName)
    (ptrName valueName : RegName)
    (offsets : Tile .nat shape) (values : Tile dtype shape)
    (hOffsets : s.regs .nat shape ptrName = some offsets)
    (hValues : s.regs dtype shape valueName = some values) :
    Stmt.atomicTraceEvents tid s
        (Stmt.atomicAdd hnum shape
          (MemAccess.region region (Op.ref .nat shape ptrName))
          (Op.ref dtype shape valueName) MaskOpt.none) =
      some ((TileShape.allIndices shape).filterMap fun i =>
        some (Stmt.atomicTraceEvent tid region (offsets.data i) dtype (values.data i))) := by
  apply atomicTraceEvents_atomicAdd_region_none
    (hnum := hnum) (tid := tid) (s := s) (region := region)
    (off := Op.ref .nat shape ptrName) (value := Op.ref dtype shape valueName)
    (offsets := offsets) (values := values)
  · simp [evalOp, hOffsets]
  · simp [evalOp, hValues]

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

/-- General ND-grid thread identity used by launcher trace witnesses. -/
def threadIdOf {g : Grid} (idx : GridIndex g) : ThreadId :=
  { program := List.ofFn fun axis : Fin g.rank => (idx axis).val
    lane := 0 }

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

/-- One program instance's stateful execution trace and final state. -/
structure ProgramRun (k : Kernel) (g : Grid) (s : BlockState)
    (idx : GridIndex g) where
  trace : Trace
  final : BlockState
  h_trace :
    Kernel.AtomicTraceStateful k (Kernel.threadIdOf idx)
      (s.withGridIndex idx) trace final

namespace ProgramRun

theorem h_exec {k : Kernel} {g : Grid} {s : BlockState}
    {idx : GridIndex g} (run : Kernel.ProgramRun k g s idx) :
    exec k (s.withGridIndex idx) = some run.final :=
  Kernel.AtomicTraceStateful_final_eq_exec run.h_trace

/-- Package a uniform per-program stateful trace theorem into the `runs` field
expected by `GridLaunchedAtomic`. -/
def of_uniform {k : Kernel} {g : Grid} {s : BlockState}
    (trace : GridIndex g → Trace)
    (final : GridIndex g → BlockState)
    (hTrace :
      ∀ idx,
        Kernel.AtomicTraceStateful k (Kernel.threadIdOf idx)
          (s.withGridIndex idx) (trace idx) (final idx)) :
    (idx : GridIndex g) → Kernel.ProgramRun k g s idx :=
  fun idx =>
    { trace := trace idx
      final := final idx
      h_trace := hTrace idx }

end ProgramRun

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

/-- Recompose a kernel-level stateful atomic trace from a pure prefix and a
stateful tail trace.

This is the generic "drop a register-only prefix, replay the tail" pattern used
by compute-then-atomic kernels: the prefix may update registers and ordinary
state, but every statement in it must emit an empty atomic trace. -/
theorem AtomicTraceStateful_of_dropPrefix
    {k : Kernel} {tid : ThreadId} {s sPrefix final : BlockState}
    {trace : Trace} (n : Nat)
    (hPrefixEmpty :
      ∀ st ∈ k.body.take n, ∀ s0, Stmt.atomicTraceEvents tid s0 st = some [])
    (hPrefixStep : stepStmts (k.body.take n) s = some sPrefix)
    (hTail :
      AtomicTraceStatefulList tid (k.body.drop n) sPrefix = some (trace, final)) :
    AtomicTraceStateful k tid s trace final := by
  rw [AtomicTraceStateful]
  rw [show k.body = k.body.take n ++ k.body.drop n by
    exact (List.take_append_drop n k.body).symm]
  apply AtomicTraceStatefulList_append_of_stepPrefix
  · exact AtomicTraceStatefulList_empty_of_stepStmts hPrefixEmpty hPrefixStep
  · exact hTail

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

noncomputable def gridRMWEvents {k : Kernel} {g : Grid} {s : BlockState}
    (runs : (idx : GridIndex g) → Kernel.ProgramRun k g s idx)
    (contributors : Finset (GridIndex g)) (cell : MemCellAddr) :
    List RMWEvent :=
  contributors.toList.flatMap fun idx => (runs idx).trace.rmwEventsAt cell

/--
Merge ordinary disjoint frame writes with one order-sensitive RMW cell.

The RMW cell is supplied by a chosen per-cell linearization. Other ordinary
frame writes still use the #49 merge behavior, and unrelated cells keep the
initial memory.
-/
noncomputable def mergeFramesWithRMW {k : Kernel} (g : Grid) (s : BlockState)
    (frames : Kernel.GridFrames k g s)
    (cell : MemCellAddr) (finalCell : MemCell) : BlockState := by
  classical
  exact { s with mem := fun region offset =>
    (by
      if hCell : (region, offset) = cell then
        exact finalCell
      else if h : Kernel.GridWriteFootprint frames (region, offset) then
        exact (Kernel.mergeFrames g s frames).mem region offset
      else
        exact s.mem region offset) }

theorem mergeFramesWithRMW_mem_cell {k : Kernel} {g : Grid} {s : BlockState}
    (frames : Kernel.GridFrames k g s) (cell : MemCellAddr)
    (finalCell : MemCell) :
    (Kernel.mergeFramesWithRMW g s frames cell finalCell).mem cell.1 cell.2 =
      finalCell := by
  classical
  unfold Kernel.mergeFramesWithRMW
  simp

theorem mergeFramesWithRMW_mem_written_of_ne {k : Kernel} {g : Grid}
    {s : BlockState} {frames : Kernel.GridFrames k g s}
    (hDisjoint : Kernel.GridWritesDisjoint frames)
    (cell : MemCellAddr) (finalCell : MemCell)
    (idx : GridIndex g) (region : RegionName) (offset : Nat)
    (hNe : (region, offset) ≠ cell)
    (hWrite : (frames idx).writes (region, offset)) :
    (Kernel.mergeFramesWithRMW g s frames cell finalCell).mem region offset =
      (frames idx).final.mem region offset := by
  classical
  unfold Kernel.mergeFramesWithRMW
  dsimp
  have hWritten : Kernel.GridWriteFootprint frames (region, offset) := ⟨idx, hWrite⟩
  simp [hNe, hWritten, Kernel.mergeFrames_mem_written hDisjoint idx region offset hWrite]

theorem mergeFramesWithRMW_mem_unwritten_of_ne {k : Kernel} {g : Grid}
    {s : BlockState} {frames : Kernel.GridFrames k g s}
    (cell : MemCellAddr) (finalCell : MemCell)
    (region : RegionName) (offset : Nat)
    (hNe : (region, offset) ≠ cell)
    (hNotWritten : ¬ Kernel.GridWriteFootprint frames (region, offset)) :
    (Kernel.mergeFramesWithRMW g s frames cell finalCell).mem region offset =
      s.mem region offset := by
  classical
  unfold Kernel.mergeFramesWithRMW
  simp [hNe, hNotWritten]

/-- Relational whole-grid launch surface for kernels with atomic-add
contributions.

The relation packages per-program stateful traces, ordinary write frames, the
selected atomic contributors, and the final-state merge.  It deliberately stays
relational: no arbitrary program scheduling order is chosen. -/
structure GridLaunchedAtomic (k : Kernel) (g : Grid)
    (s sFinal : BlockState) where
  runs : (idx : GridIndex g) → Kernel.ProgramRun k g s idx
  frames : Kernel.GridFrames k g s
  h_frame_final : ∀ idx, (frames idx).final = (runs idx).final
  h_disjoint : Kernel.GridWritesDisjoint frames
  contributors : Finset (GridIndex g)
  h_final :
    sFinal = Kernel.mergeFramesWithAtomic g s frames contributors
      (fun idx => (runs idx).trace)

namespace GridLaunchedAtomic

theorem observeAtomicCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedAtomic k g s sFinal)
    (region : RegionName) (offset : Nat)
    (hNoOrdinaryWrite : ¬ Kernel.GridWriteFootprint h.frames (region, offset)) :
    sFinal.readMem region offset =
      s.readMem region offset +
        h.contributors.sum
          (fun idx => (h.runs idx).trace.atomicAddRealSum (region, offset)) := by
  rcases h with ⟨runs, frames, _hFrameFinal, _hDisjoint, contributors, hFinal⟩
  subst sFinal
  exact Kernel.mergeFramesWithAtomic_atomicAdd_eq_finsetSum
    (frames := frames) (contributors := contributors)
    (atomicTrace := fun idx => (runs idx).trace)
    (region := region) (offset := offset)
    (hNoOrdinaryWrite := hNoOrdinaryWrite)

theorem observeOrdinaryCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedAtomic k g s sFinal)
    (idx : GridIndex g) (region : RegionName) (offset : Nat)
    (hWrite : (h.frames idx).writes (region, offset)) :
    sFinal.mem region offset = (h.frames idx).final.mem region offset := by
  classical
  rcases h with ⟨_runs, frames, _hFrameFinal, hDisjoint, _contributors, hFinal⟩
  subst sFinal
  unfold Kernel.mergeFramesWithAtomic
  dsimp
  have hWritten : Kernel.GridWriteFootprint frames (region, offset) := ⟨idx, hWrite⟩
  simp [hWritten]
  exact Kernel.mergeFrames_mem_written hDisjoint idx region offset hWrite

theorem observeAtomicCell_of_sum_eq {k : Kernel} {g : Grid}
    {s sFinal : BlockState}
    (h : Kernel.GridLaunchedAtomic k g s sFinal)
    (region : RegionName) (offset : Nat)
    (hNoOrdinaryWrite : ¬ Kernel.GridWriteFootprint h.frames (region, offset))
    (rhs : ℝ)
    (hSum :
      h.contributors.sum
          (fun idx => (h.runs idx).trace.atomicAddRealSum (region, offset)) =
        rhs) :
    sFinal.readMem region offset = s.readMem region offset + rhs := by
  rw [h.observeAtomicCell region offset hNoOrdinaryWrite, hSum]

end GridLaunchedAtomic

/-- Relational whole-grid launch surface for one order-sensitive RMW cell.

This is the #82 launcher hook for `atomic_xchg` / `atomic_cas`: callers provide
per-program stateful traces, an explicit contributor set, and a per-cell
linearization list.  The final cell is produced by `RMWTrace.applyLinearized`.
No global scheduler or multi-cell transaction semantics are chosen here. -/
structure GridLaunchedRMW (k : Kernel) (g : Grid)
    (s sFinal : BlockState) where
  runs : (idx : GridIndex g) → Kernel.ProgramRun k g s idx
  frames : Kernel.GridFrames k g s
  h_frame_final : ∀ idx, (frames idx).final = (runs idx).final
  h_disjoint : Kernel.GridWritesDisjoint frames
  contributors : Finset (GridIndex g)
  cell : MemCellAddr
  linearization : List RMWEvent
  finalCell : MemCell
  linearizedEvents : List RMWEvent
  h_no_ordinary : ¬ Kernel.GridWriteFootprint frames cell
  h_events :
    linearization.Perm (Kernel.gridRMWEvents runs contributors cell)
  h_apply :
    RMWTrace.applyLinearized (s.mem cell.1 cell.2) linearization =
      some (finalCell, linearizedEvents)
  h_final :
    sFinal = Kernel.mergeFramesWithRMW g s frames cell finalCell

namespace GridLaunchedRMW

theorem observeRMWCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedRMW k g s sFinal) :
    sFinal.mem h.cell.1 h.cell.2 = h.finalCell := by
  rcases h with
    ⟨runs, frames, hFrameFinal, hDisjoint, contributors, cell, linearization,
      finalCell, linearizedEvents, hNoOrdinary, hEvents, hApply, hFinal⟩
  subst sFinal
  exact Kernel.mergeFramesWithRMW_mem_cell frames cell finalCell

theorem observeOrdinaryCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedRMW k g s sFinal)
    (idx : GridIndex g) (region : RegionName) (offset : Nat)
    (hNe : (region, offset) ≠ h.cell)
    (hWrite : (h.frames idx).writes (region, offset)) :
    sFinal.mem region offset = (h.frames idx).final.mem region offset := by
  rcases h with
    ⟨runs, frames, hFrameFinal, hDisjoint, contributors, cell, linearization,
      finalCell, linearizedEvents, hNoOrdinary, hEvents, hApply, hFinal⟩
  subst sFinal
  exact Kernel.mergeFramesWithRMW_mem_written_of_ne hDisjoint cell finalCell
    idx region offset hNe hWrite

theorem observeUnwrittenCell {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedRMW k g s sFinal)
    (region : RegionName) (offset : Nat)
    (hNe : (region, offset) ≠ h.cell)
    (hNotWritten : ¬ Kernel.GridWriteFootprint h.frames (region, offset)) :
    sFinal.mem region offset = s.mem region offset := by
  rcases h with
    ⟨runs, frames, hFrameFinal, hDisjoint, contributors, cell, linearization,
      finalCell, linearizedEvents, hNoOrdinary, hEvents, hApply, hFinal⟩
  subst sFinal
  exact Kernel.mergeFramesWithRMW_mem_unwritten_of_ne cell finalCell
    region offset hNe hNotWritten

theorem applyLinearized {k : Kernel} {g : Grid} {s sFinal : BlockState}
    (h : Kernel.GridLaunchedRMW k g s sFinal) :
    RMWTrace.applyLinearized (s.mem h.cell.1 h.cell.2) h.linearization =
      some (h.finalCell, h.linearizedEvents) :=
  h.h_apply

end GridLaunchedRMW

end Kernel

end VeriTile.Triton
