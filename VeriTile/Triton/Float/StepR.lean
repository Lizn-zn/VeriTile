/-
VeriTile.Triton.Float.StepR

Rounding-model-parametric statement stepping and kernel execution (#447).

`stepStmtR R` is `stepStmt` with two differences:
* expressions evaluate through `evalOpR R` (float casts round at the
  destination dtype — rounding-event site 1);
* stores and `atomicAdd` writes go through `BlockState.writeMemTypedR R`,
  which quantizes the written value at the buffer's dtype (rounding-event
  site 2).

Everything else is a verbatim copy that threads `R` through the recursion.
The degeneration theorems (`stepStmtR_triv`, `execR_triv`) state that the
trivial model recovers today's semantics exactly.

Scope notes (Phase A, deliberate):
* `.real` stores never round (the mathematical channel), so they delegate to
  the existing `writeMemAs`.
* `Stmt.atomicRMW` (the xchg/cas/max/min read-modify-write family) delegates
  to the existing `stepStmt`: its writes are exact `MemCell` replacements via
  `writeCell` even today, and its input expressions evaluate under the
  trivial model. Making RMW inputs rounding-aware is follow-up work; no
  current bench value proof depends on float casts inside RMW inputs.

This file does NOT modify `stepStmt` or anything it depends on.
-/

import VeriTile.Triton.Semantics.Step
import VeriTile.Triton.Float.EvalOpR

namespace VeriTile.Triton

namespace BlockState

/-- `writeMemAs` with the stored value quantized by the rounding model at the
buffer's dtype (rounding-event site 2 of 2; site 1 is `Op.castFloat` in
`evalOpR`). The `⊥ ↦ 0` finite fallback happens before rounding, matching
`FloatDType.storeValue`. -/
def writeMemAsR (R : RoundingModel) (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) : BlockState :=
  { s with mem := fun r o =>
      if r = region ∧ o = offset then
        MemCell.of dtype.toTileDType (dtype.ofReal (R.storeValue dtype v))
      else
        s.mem r o }

@[simp] theorem writeMemAsR_triv (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) :
    s.writeMemAsR RoundingModel.triv dtype region offset v =
      s.writeMemAs dtype region offset v := by
  simp [writeMemAsR, writeMemAs]

/-- `writeMemTyped` with rounding-aware narrowing float stores. The `.real`
channel and all non-float channels delegate to the existing exact writes:
only `.fp32` / `.fp16` / `.bf16` stores are rounding events. -/
def writeMemTypedR (R : RoundingModel) (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) : BlockState :=
  match dtype with
  | .fp32 => s.writeMemAsR R .fp32 region offset v
  | .fp16 => s.writeMemAsR R .fp16 region offset v
  | .bf16 => s.writeMemAsR R .bf16 region offset v
  | dtype => s.writeMemTyped dtype region offset v

@[simp] theorem writeMemTypedR_triv (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) :
    s.writeMemTypedR RoundingModel.triv dtype region offset v =
      s.writeMemTyped dtype region offset v := by
  cases dtype <;> simp [writeMemTypedR, writeMemTyped]

end BlockState

mutual

/-- `stepStmt` parametrized by a `RoundingModel`. See the module docstring
for the two rounding-event sites and the `atomicRMW` scope note. -/
noncomputable def stepStmtR (R : RoundingModel) : Stmt → BlockState → Option BlockState
  | .assign dtype shape name e, s => do
      let v ← evalOpR R e s
      some (s.setReg name dtype shape v)
  | .store dtype shape mem val mask, s => do
      let values ← evalOpR R val s
      let mkActive : Option (TileIndex shape → Bool) :=
        match mask with
        | .none => some (fun _ : TileIndex shape => true)
        | .mask mask =>
            (evalOpR R mask s).map fun masks => fun i : TileIndex shape => masks.data i
        | .maskOther mask _ =>
            (evalOpR R mask s).map fun masks => fun i : TileIndex shape => masks.data i
      let active ← mkActive
      let writeValue : TileIndex shape → TileCarrier dtype := fun i => values.data i
      match mem with
      | .region region off => do
          let offsets ← evalOpR R off s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              if active i then acc.writeMemTypedR R dtype (Region.cast region) (offsets.data i) (writeValue i)
              else acc) s)
      | .ptr ptr => do
          let ptrs ← evalOpR R ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let p := ptrs.data i
              if active i then acc.writeMemTypedR R dtype p.1 p.2 (writeValue i)
              else acc) s)
      | .blockPtr ptr boundaryCheck => do
          let ptrTile ← evalOpR R ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let bp := ptrTile.data i
              let idx := TileShape.indexToList shape i
              if active i && bp.inBounds idx boundaryCheck then
                acc.writeMemTypedR R dtype bp.region (bp.address idx) (writeValue i)
              else acc) s)
  | @Stmt.atomicAdd dtype h shape mem val mask, s => do
      let values ← evalOpR R val s
      let mkActive : Option (TileIndex shape → Bool) :=
        match mask with
        | .none => some (fun _ : TileIndex shape => true)
        | .mask mask =>
            (evalOpR R mask s).map fun masks => fun i : TileIndex shape => masks.data i
        | .maskOther mask _ =>
            (evalOpR R mask s).map fun masks => fun i : TileIndex shape => masks.data i
      let active ← mkActive
      let atomicValue : BlockState → RegionName → Nat → TileIndex shape → TileCarrier dtype :=
        fun acc region offset i =>
          h.add (acc.readMemValue dtype region offset) (values.data i)
      match mem with
      | .region region off => do
          let offsets ← evalOpR R off s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              if active i then
                acc.writeMemTypedR R dtype (Region.cast region) (offsets.data i)
                  (atomicValue acc (Region.cast region) (offsets.data i) i)
              else acc) s)
      | .ptr ptr => do
          let ptrs ← evalOpR R ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let p := ptrs.data i
              if active i then
                acc.writeMemTypedR R dtype p.1 p.2 (atomicValue acc p.1 p.2 i)
              else acc) s)
      | .blockPtr ptr boundaryCheck => do
          let ptrTile ← evalOpR R ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let bp := ptrTile.data i
              let idx := TileShape.indexToList shape i
              if active i && bp.inBounds idx boundaryCheck then
                acc.writeMemTypedR R dtype bp.region (bp.address idx)
                  (atomicValue acc bp.region (bp.address idx) i)
              else acc) s)
  | .atomicRMW op dtype shape mem input extraInput mask dest, s =>
      -- Deliberate delegation: RMW writes are exact cell replacements even in
      -- the base semantics (`writeCell`), and RMW inputs evaluate under the
      -- trivial model in Phase A. See the module docstring.
      stepStmt (.atomicRMW op dtype shape mem input extraInput mask dest) s
  | .forLoop idx n body, s =>
      stepForLoopAuxR R idx 0 n body s
  | .forRange idx start stop step body, s =>
      stepForRangeAuxR R idx start stop step body s
  | .forRangeDyn idx start stop step body, s => do
      let start' ← evalOpR R start s
      let stop' ← evalOpR R stop s
      let step' ← evalOpR R step s
      stepForRangeAuxR R idx (start'.data PUnit.unit) (stop'.data PUnit.unit)
        (step'.data PUnit.unit) body s
  | .ifThen cond body, s => do
      let c ← evalOpR R cond s
      if c.data PUnit.unit then stepStmtsR R body s else some s
  | .ifThenElse cond thenBody elseBody, s => do
      let c ← evalOpR R cond s
      if c.data PUnit.unit then stepStmtsR R thenBody s else stepStmtsR R elseBody s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

noncomputable def stepStmtsR (R : RoundingModel) : List Stmt → BlockState → Option BlockState
  | [], s => some s
  | st :: rest, s =>
      match stepStmtR R st s with
      | some s' => stepStmtsR R rest s'
      | none => none
termination_by l _ => (sizeOf l, 0)
decreasing_by all_goals (simp_wf; omega)

noncomputable def stepForLoopAuxR (R : RoundingModel)
    (idx : RegName) (start n : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if start < n then
        match stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar start)) with
        | some s' => stepForLoopAuxR R idx (start + 1) n body s'
        | none => none
      else some s
termination_by _ => (sizeOf body + 1, n - start)
decreasing_by all_goals omega

noncomputable def stepForRangeAuxR (R : RoundingModel)
    (idx : RegName) (cur stop step : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if step = 0 then some s
      else if cur < stop then
        match stepStmtsR R body (s.setReg idx .nat [] (Tile.scalar cur)) with
        | some s' => stepForRangeAuxR R idx (cur + step) stop step body s'
        | none => none
      else some s
termination_by _ => (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega

end

/-- Kernel execution under a rounding model. -/
noncomputable def execR (R : RoundingModel) (k : Kernel) (s : BlockState) :
    Option BlockState :=
  stepStmtsR R k.body s

/-! ## Degeneration to the trivial model

At `RoundingModel.triv` both rounding-event sites are the identity
(`evalOpR_triv` for casts, `BlockState.writeMemTypedR_triv` for narrowing
stores), so the parametric stepper collapses onto `stepStmt` arm by arm.
The four mutual theorems mirror the termination measures of the four mutual
definitions above. -/

set_option maxHeartbeats 1600000 in
mutual

/-- The rounding-model-parametric stepper degenerates to the existing stepper
at the trivial model: `stepStmtR .triv` *is* `stepStmt`. -/
theorem stepStmtR_triv : ∀ (st : Stmt) (s : BlockState),
    stepStmtR RoundingModel.triv st s = stepStmt st s
  | .assign dtype shape name e, s => by
      simp only [stepStmtR, stepStmt, evalOpR_triv e s]
  | .store dtype shape mem val mask, s => by
      cases mem <;> cases mask <;>
        simp only [stepStmtR, stepStmt, evalOpR_triv, BlockState.writeMemTypedR_triv]
  | @Stmt.atomicAdd dtype h shape mem val mask, s => by
      cases mem <;> cases mask <;>
        simp only [stepStmtR, stepStmt, evalOpR_triv, BlockState.writeMemTypedR_triv]
  | .atomicRMW op dtype shape mem input extraInput mask dest, s => by
      simp only [stepStmtR]
  | .forLoop idx n body, s => by
      simp only [stepStmtR, stepStmt, stepForLoopAuxR_triv idx 0 n body s]
  | .forRange idx start stop step body, s => by
      simp only [stepStmtR, stepStmt, stepForRangeAuxR_triv idx start stop step body s]
  | .forRangeDyn idx start stop step body, s => by
      simp only [stepStmtR, stepStmt, evalOpR_triv start s, evalOpR_triv stop s,
        evalOpR_triv step s]
      cases evalOp start s with
      | none => rfl
      | some start' =>
        cases evalOp stop s with
        | none => rfl
        | some stop' =>
          cases evalOp step s with
          | none => rfl
          | some step' =>
            exact stepForRangeAuxR_triv idx (start'.data PUnit.unit) (stop'.data PUnit.unit)
              (step'.data PUnit.unit) body s
  | .ifThen cond body, s => by
      simp only [stepStmtR, stepStmt, evalOpR_triv cond s, stepStmtsR_triv body s]
  | .ifThenElse cond thenBody elseBody, s => by
      simp only [stepStmtR, stepStmt, evalOpR_triv cond s, stepStmtsR_triv thenBody s,
        stepStmtsR_triv elseBody s]
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

/-- Statement-list degeneration: `stepStmtsR .triv` *is* `stepStmts`. -/
theorem stepStmtsR_triv : ∀ (l : List Stmt) (s : BlockState),
    stepStmtsR RoundingModel.triv l s = stepStmts l s
  | [], s => by simp only [stepStmtsR, stepStmts]
  | st :: rest, s => by
      simp only [stepStmtsR, stepStmts, stepStmtR_triv st s]
      cases stepStmt st s with
      | none => rfl
      | some s' => exact stepStmtsR_triv rest s'
termination_by l _ => (sizeOf l, 0)
decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary degeneration. -/
theorem stepForLoopAuxR_triv : ∀ (idx : RegName) (start n : Nat) (body : List Stmt)
    (s : BlockState),
    stepForLoopAuxR RoundingModel.triv idx start n body s = stepForLoopAux idx start n body s
  | idx, start, n, body, s => by
      rw [stepForLoopAuxR, stepForLoopAux]
      simp only [stepStmtsR_triv body (s.setReg idx .nat [] (Tile.scalar start))]
      split
      · cases stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rfl
        | some s' => exact stepForLoopAuxR_triv idx (start + 1) n body s'
      · rfl
termination_by _ start n body _ => (sizeOf body + 1, n - start)
decreasing_by all_goals omega

/-- `forRange` auxiliary degeneration. -/
theorem stepForRangeAuxR_triv : ∀ (idx : RegName) (cur stop step : Nat) (body : List Stmt)
    (s : BlockState),
    stepForRangeAuxR RoundingModel.triv idx cur stop step body s =
      stepForRangeAux idx cur stop step body s
  | idx, cur, stop, step, body, s => by
      rw [stepForRangeAuxR, stepForRangeAux]
      simp only [stepStmtsR_triv body (s.setReg idx .nat [] (Tile.scalar cur))]
      split
      · rfl
      · split
        · cases stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rfl
          | some s' => exact stepForRangeAuxR_triv idx (cur + step) stop step body s'
        · rfl
termination_by _ cur stop _ body _ => (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega

end

/-- Kernel-execution degeneration: `execR .triv` *is* `exec`. -/
theorem execR_triv (k : Kernel) (s : BlockState) :
    execR RoundingModel.triv k s = exec k s := by
  simp only [execR, exec, stepStmtsR_triv k.body s]

end VeriTile.Triton
