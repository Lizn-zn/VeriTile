/-
VeriTile.Semantics.Step

Statement stepping, kernel execution, and execution-state preservation lemmas.
-/

import VeriTile.Semantics.EvalOp
import VeriTile.Semantics.AtomicReduction

namespace VeriTile

private noncomputable def foldAtomicRMWIndices
    {dtype : TileDType} {shape : TileShape}
    (op : RMWOp)
    (regionFn : TileIndex shape → RegionName)
    (offsetFn : TileIndex shape → Nat)
    (inputFn : TileIndex shape → TileCarrier dtype)
    (extraFn : Option (TileIndex shape → TileCarrier dtype))
    (active : TileIndex shape → Bool)
    (indices : List (TileIndex shape))
    (s : BlockState)
    (retFn : TileIndex shape → TileCarrier dtype) :
    Option (BlockState × Tile dtype shape) := by
  classical
  induction indices generalizing s retFn with
  | nil =>
      exact some (s, ⟨retFn⟩)
  | cons i rest ih =>
      if active i then
        let extra := extraFn.map (fun f => f i)
        match s.atomicRMWAt op dtype (regionFn i) (offsetFn i) (inputFn i) extra with
        | none => exact none
        | some (s', ret, _) =>
            exact ih s' (fun j => if j = i then ret else retFn j)
      else
        exact ih s retFn

private noncomputable def stepAtomicRMWRaw
    (op : RMWOp) (dtype : TileDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) (s : BlockState) : Option BlockState := do
  let inputs ← evalOp input s
  let extras ←
    match extraInput with
    | none => some none
    | some extra => (evalOp extra s).map some
  let mkActive : Option (TileIndex shape → Bool) :=
    match mask with
    | .none => some (fun _ : TileIndex shape => true)
    | .mask mask =>
        (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
    | .maskOther mask _ =>
        (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
  let active ← mkActive
  let extraFn := extras.map fun extraTile => fun i : TileIndex shape => extraTile.data i
  let (s', returns) ←
    match mem with
    | .region region off => do
        let offsets ← evalOp off s
        foldAtomicRMWIndices op (fun _ => Region.cast region) (fun i => offsets.data i)
          (fun i => inputs.data i) extraFn active (TileShape.allIndices shape) s
          (fun _ => BlockState.defaultCarrier dtype)
    | .ptr ptr => do
        let ptrs ← evalOp ptr s
        foldAtomicRMWIndices op (fun i => (ptrs.data i).1) (fun i => (ptrs.data i).2)
          (fun i => inputs.data i) extraFn active (TileShape.allIndices shape) s
          (fun _ => BlockState.defaultCarrier dtype)
    | .blockPtr ptr boundaryCheck => do
        let ptrs ← evalOp ptr s
        foldAtomicRMWIndices op
          (fun i => (ptrs.data i).region)
          (fun i => (ptrs.data i).address (TileShape.indexToList shape i))
          (fun i => inputs.data i) extraFn
          (fun i =>
            active i && (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
          (TileShape.allIndices shape) s
          (fun _ => BlockState.defaultCarrier dtype)
  match dest with
  | none => some s'
  | some name => some (s'.setReg name dtype shape returns)

private noncomputable def stepAtomicRMW
    (op : RMWOp) (dtype : TileDType) (shape : TileShape)
    (mem : MemAccess dtype shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) (s : BlockState) : Option BlockState :=
  (stepAtomicRMWRaw op dtype shape mem input extraInput mask dest s).map
    (fun s' => { s' with pids := s.pids })

private theorem stepAtomicRMW_pid
    {op : RMWOp} {dtype : TileDType} {shape : TileShape}
    {mem : MemAccess dtype shape} {input : Op dtype shape}
    {extraInput : Option (Op dtype shape)} {mask : MaskOpt dtype shape}
    {dest : Option RegName} {s s' : BlockState}
    (h : stepAtomicRMW op dtype shape mem input extraInput mask dest s = some s') :
    s'.pid = s.pid := by
  unfold stepAtomicRMW at h
  cases hRaw : stepAtomicRMWRaw op dtype shape mem input extraInput mask dest s <;>
    simp [hRaw] at h
  cases h
  rfl

mutual

noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign dtype shape name e, s => do
      let v ← evalOp e s
      some (s.setReg name dtype shape v)
  | .store dtype shape mem val mask, s => do
      let values ← evalOp val s
      let mkActive : Option (TileIndex shape → Bool) :=
        match mask with
        | .none => some (fun _ : TileIndex shape => true)
        | .mask mask =>
            (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
        | .maskOther mask _ =>
            (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
      let active ← mkActive
      let writeValue : TileIndex shape → TileCarrier dtype := fun i => values.data i
      match mem with
      | .region region off => do
          let offsets ← evalOp off s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              if active i then acc.writeMemTyped dtype (Region.cast region) (offsets.data i) (writeValue i)
              else acc) s)
      | .ptr ptr => do
          let ptrs ← evalOp ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let p := ptrs.data i
              if active i then acc.writeMemTyped dtype p.1 p.2 (writeValue i)
              else acc) s)
      | .blockPtr ptr boundaryCheck => do
          let ptrTile ← evalOp ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let bp := ptrTile.data i
              let idx := TileShape.indexToList shape i
              if active i && bp.inBounds idx boundaryCheck then
                acc.writeMemTyped dtype bp.region (bp.address idx) (writeValue i)
              else acc) s)
  | @Stmt.atomicAdd dtype h shape mem val mask, s => do
      let values ← evalOp val s
      let mkActive : Option (TileIndex shape → Bool) :=
        match mask with
        | .none => some (fun _ : TileIndex shape => true)
        | .mask mask =>
            (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
        | .maskOther mask _ =>
            (evalOp mask s).map fun masks => fun i : TileIndex shape => masks.data i
      let active ← mkActive
      let atomicValue : BlockState → RegionName → Nat → TileIndex shape → TileCarrier dtype :=
        fun acc region offset i =>
          h.add (acc.readMemValue dtype region offset) (values.data i)
      match mem with
      | .region region off => do
          let offsets ← evalOp off s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              if active i then
                acc.writeMemTyped dtype (Region.cast region) (offsets.data i)
                  (atomicValue acc (Region.cast region) (offsets.data i) i)
              else acc) s)
      | .ptr ptr => do
          let ptrs ← evalOp ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let p := ptrs.data i
              if active i then
                acc.writeMemTyped dtype p.1 p.2 (atomicValue acc p.1 p.2 i)
              else acc) s)
      | .blockPtr ptr boundaryCheck => do
          let ptrTile ← evalOp ptr s
          some ((TileShape.allIndices shape).foldl
            (fun acc i =>
              let bp := ptrTile.data i
              let idx := TileShape.indexToList shape i
              if active i && bp.inBounds idx boundaryCheck then
                acc.writeMemTyped dtype bp.region (bp.address idx)
                  (atomicValue acc bp.region (bp.address idx) i)
              else acc) s)
  | .atomicRMW op dtype shape mem input extraInput mask dest, s =>
      stepAtomicRMW op dtype shape mem input extraInput mask dest s
  | .forLoop idx n body, s =>
      stepForLoopAux idx 0 n body s
  | .forRange idx start stop step body, s =>
      stepForRangeAux idx start stop step body s
  | .forRangeDyn idx start stop step body, s => do
      let start' ← evalOp start s
      let stop' ← evalOp stop s
      let step' ← evalOp step s
      stepForRangeAux idx (start'.data PUnit.unit) (stop'.data PUnit.unit)
        (step'.data PUnit.unit) body s
  | .ifThen cond body, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts body s else some s
  | .ifThenElse cond thenBody elseBody, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts thenBody s else stepStmts elseBody s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals simp_wf
  all_goals (try omega)
  all_goals (have : 0 < sizeOf idx := by cases idx; simp)
  all_goals omega

noncomputable def stepStmts : List Stmt → BlockState → Option BlockState
  | [], s => some s
  | st :: rest, s =>
      match stepStmt st s with
      | some s' => stepStmts rest s'
      | none => none
termination_by l _ => (sizeOf l, 0)
decreasing_by all_goals (simp_wf; omega)

noncomputable def stepForLoopAux
    (idx : RegName) (start n : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if start < n then
        match stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | some s' => stepForLoopAux idx (start + 1) n body s'
        | none => none
      else some s
termination_by _ => (sizeOf body + 1, n - start)
decreasing_by all_goals omega

noncomputable def stepForRangeAux
    (idx : RegName) (cur stop step : Nat) (body : List Stmt) :
    BlockState → Option BlockState
  | s =>
      if step = 0 then some s
      else if cur < stop then
        match stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
        | some s' => stepForRangeAux idx (cur + step) stop step body s'
        | none => none
      else some s
termination_by _ => (sizeOf body + 1, stop - cur)
decreasing_by all_goals omega

end

namespace stepStmts

@[simp] theorem nil {s : BlockState} :
    stepStmts [] s = some s := by
  unfold stepStmts
  rfl

theorem cons_some {st : Stmt} {rest : List Stmt} {s s' : BlockState}
    (h : stepStmt st s = some s') :
    stepStmts (st :: rest) s = stepStmts rest s' := by
  conv_lhs => unfold stepStmts
  rw [h]

theorem append_some {l1 l2 : List Stmt} {s s' : BlockState}
    (h : stepStmts l1 s = some s') :
    stepStmts (l1 ++ l2) s = stepStmts l2 s' := by
  induction l1 generalizing s with
  | nil =>
      unfold stepStmts at h
      injection h with hs
      subst hs
      rfl
  | cons st rest ih =>
      conv_lhs at h => unfold stepStmts
      cases hst : stepStmt st s with
      | none =>
          simp [hst] at h
      | some smid =>
          simp [hst] at h
          rw [List.cons_append, cons_some hst]
          exact ih h

theorem append_some_iff {l1 l2 : List Stmt} {s s' : BlockState} :
    stepStmts (l1 ++ l2) s = some s' ↔
      ∃ smid, stepStmts l1 s = some smid ∧ stepStmts l2 smid = some s' := by
  induction l1 generalizing s with
  | nil =>
      constructor
      · intro h
        refine ⟨s, ?_, h⟩
        simp
      · intro h
        rcases h with ⟨smid, hLeft, hRight⟩
        simp at hLeft
        subst smid
        exact hRight
  | cons st rest ih =>
      constructor
      · intro h
        unfold stepStmts at h
        cases hst : stepStmt st s with
        | none =>
            simp [hst] at h
        | some s1 =>
            simp [hst] at h
            rcases (ih.mp h) with ⟨smid, hRest, hRight⟩
            refine ⟨smid, ?_, hRight⟩
            rw [cons_some hst]
            exact hRest
      · intro h
        rcases h with ⟨smid, hLeft, hRight⟩
        unfold stepStmts at hLeft
        cases hst : stepStmt st s with
        | none =>
            simp [hst] at hLeft
        | some s1 =>
            simp [hst] at hLeft
            rw [List.cons_append, cons_some hst]
            exact ih.mpr ⟨smid, hLeft, hRight⟩

/-- Sequencing law in `bind` form: running the concatenation of two statement
lists is running the first, then (on success) the second. This is the packaged
version of `append_some` / `append_some_iff` used to reason about kernel body
concatenation (`seqCompose`). -/
theorem append (l1 l2 : List Stmt) (s : BlockState) :
    stepStmts (l1 ++ l2) s = (stepStmts l1 s).bind (fun s' => stepStmts l2 s') := by
  induction l1 generalizing s with
  | nil => simp
  | cons st rest ih =>
      simp [stepStmts]
      cases stepStmt st s <;> simp [ih]

/-- Running a `flatMap`-concatenated body is the monadic left-fold of running
each piece in sequence. The engine behind `exec_seqCompose`: a pipeline of `n`
kernels executes as `foldlM` over the individual bodies. -/
theorem flatMap {α : Type _} (f : α → List Stmt) (ks : List α) (s : BlockState) :
    stepStmts (ks.flatMap f) s = ks.foldlM (fun s k => stepStmts (f k) s) s := by
  induction ks generalizing s with
  | nil => simp
  | cons k rest ih =>
      simp only [List.flatMap_cons, List.foldlM_cons, append]
      cases stepStmts (f k) s <;> simp [ih]

end stepStmts

namespace stepForLoopAux

@[simp] theorem step_ge {idx} {start n} {body} {s} (h : n ≤ start) :
    stepForLoopAux idx start n body s = some s := by
  unfold stepForLoopAux
  simp [Nat.not_lt.mpr h]

@[simp] theorem step_eq_self {idx} {n} {body} (s) :
    stepForLoopAux idx n n body s = some s :=
  step_ge (le_refl n)

@[simp] theorem step_lt {idx} {start n} {body} {s} (h : start < n) :
    stepForLoopAux idx start n body s
      = (stepStmts body (s.setReg idx .nat [] (Tile.scalar start))).bind
          (stepForLoopAux idx (start + 1) n body) := by
  conv_lhs => unfold stepForLoopAux
  simp [h]
  cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) <;> rfl

@[simp] theorem forLoop_unfold {idx} {n} {body} {s} :
    stepStmt (.forLoop idx n body) s
      = stepForLoopAux idx 0 n body s := by
  unfold stepStmt
  rfl

end stepForLoopAux

namespace stepForRangeAux

@[simp] theorem step_zero_step {idx} {cur stop} {body} {s} :
    stepForRangeAux idx cur stop 0 body s = some s := by
  unfold stepForRangeAux
  simp

@[simp] theorem step_ge {idx} {cur stop step} {body} {s}
    (hstep : step ≠ 0) (h : stop ≤ cur) :
    stepForRangeAux idx cur stop step body s = some s := by
  unfold stepForRangeAux
  simp [hstep, Nat.not_lt.mpr h]

@[simp] theorem step_lt {idx} {cur stop step} {body} {s}
    (hstep : step ≠ 0) (h : cur < stop) :
    stepForRangeAux idx cur stop step body s
      = (stepStmts body (s.setReg idx .nat [] (Tile.scalar cur))).bind
          (stepForRangeAux idx (cur + step) stop step body) := by
  conv_lhs => unfold stepForRangeAux
  simp [hstep, h]
  cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) <;> rfl

@[simp] theorem forRange_unfold {idx} {start stop step} {body} {s} :
    stepStmt (.forRange idx start stop step body) s
      = stepForRangeAux idx start stop step body s := by
  unfold stepStmt
  rfl

/-- One-step-and-terminate combinator: when `cur < stop ≤ cur + step`, the
loop executes one body iteration then immediately terminates. Produces a
clean `Option.bind`-free RHS suitable for further simp. Avoids the
`stepForRangeAux _ (cur+step) stop step _` residual that arises from
chaining `step_lt` with `step_ge` separately. -/
theorem step_one_iter {idx} {cur stop step} {body} {s}
    (hstep : step ≠ 0) (hlt : cur < stop) (hle : stop ≤ cur + step) :
    stepForRangeAux idx cur stop step body s =
      stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) := by
  rw [step_lt hstep hlt]
  cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur))
  · simp [Option.bind]
  · simp [Option.bind, step_ge hstep hle]

@[simp] theorem forRangeDyn_unfold {idx} {start stop step} {body} {s} :
    stepStmt (.forRangeDyn idx start stop step body) s
      = (evalOp start s).bind (fun start' =>
          (evalOp stop s).bind (fun stop' =>
            (evalOp step s).bind (fun step' =>
              stepForRangeAux idx (start'.data PUnit.unit) (stop'.data PUnit.unit)
                (step'.data PUnit.unit) body s))) := by
  unfold stepStmt
  rfl

end stepForRangeAux

noncomputable def exec (k : Kernel) (s : BlockState) : Option BlockState :=
  stepStmts k.body s

theorem stepStmt_atomicAdd_regs {dtype : TileDType} (hnum : NumericDType dtype)
    {shape : TileShape} {mem : MemAccess dtype shape} {val : Op dtype shape}
    {mask : MaskOpt dtype shape} {s s' : BlockState}
    (h : stepStmt (Stmt.atomicAdd hnum shape mem val mask) s = some s')
    (dtype' : TileDType) (shape' : TileShape) (name : RegName) :
    s'.regs dtype' shape' name = s.regs dtype' shape' name := by
  cases hval : evalOp val s with
  | none =>
      unfold stepStmt at h
      simp [hval] at h
  | some values =>
      unfold stepStmt at h
      simp [hval] at h
      cases mask <;> simp at h
      · cases mem with
        | region region off =>
          cases hoff : evalOp off s <;> simp [hoff] at h
          rename_i offsets
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun _ : TileIndex shape => Region.cast region)
              (fun i : TileIndex shape => offsets.data i)
              (fun _ i => values.data i)
              (fun _ : TileIndex shape => true)
              (TileShape.allIndices shape) s dtype' shape' name
        | ptr ptr =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrs
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrs.data i).1)
              (fun i : TileIndex shape => (ptrs.data i).2)
              (fun _ i => values.data i)
              (fun _ : TileIndex shape => true)
              (TileShape.allIndices shape) s dtype' shape' name
        | blockPtr ptr boundaryCheck =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrTile
          cases h
          simpa [Bool.and_eq_true] using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrTile.data i).region)
              (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
              (fun _ i => values.data i)
              (fun i : TileIndex shape =>
                (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
              (TileShape.allIndices shape) s dtype' shape' name
      · rename_i mask
        cases hmask : evalOp mask s <;> simp [hmask] at h
        rename_i masks
        cases mem with
        | region region off =>
          cases hoff : evalOp off s <;> simp [hoff] at h
          rename_i offsets
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun _ : TileIndex shape => Region.cast region)
              (fun i : TileIndex shape => offsets.data i)
              (fun _ i => values.data i)
              (fun i : TileIndex shape => masks.data i)
              (TileShape.allIndices shape) s dtype' shape' name
        | ptr ptr =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrs
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrs.data i).1)
              (fun i : TileIndex shape => (ptrs.data i).2)
              (fun _ i => values.data i)
              (fun i : TileIndex shape => masks.data i)
              (TileShape.allIndices shape) s dtype' shape' name
        | blockPtr ptr boundaryCheck =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrTile
          cases h
          simpa [Bool.and_eq_true] using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrTile.data i).region)
              (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
              (fun _ i => values.data i)
              (fun i : TileIndex shape =>
                masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
              (TileShape.allIndices shape) s dtype' shape' name
      · rename_i mask other
        cases hmask : evalOp mask s <;> simp [hmask] at h
        rename_i masks
        cases mem with
        | region region off =>
          cases hoff : evalOp off s <;> simp [hoff] at h
          rename_i offsets
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun _ : TileIndex shape => Region.cast region)
              (fun i : TileIndex shape => offsets.data i)
              (fun _ i => values.data i)
              (fun i : TileIndex shape => masks.data i)
              (TileShape.allIndices shape) s dtype' shape' name
        | ptr ptr =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrs
          cases h
          simpa using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrs.data i).1)
              (fun i : TileIndex shape => (ptrs.data i).2)
              (fun _ i => values.data i)
              (fun i : TileIndex shape => masks.data i)
              (TileShape.allIndices shape) s dtype' shape' name
        | blockPtr ptr boundaryCheck =>
          cases hptr : evalOp ptr s <;> simp [hptr] at h
          rename_i ptrTile
          cases h
          simpa [Bool.and_eq_true] using
            foldl_atomicAddAt_regs dtype hnum
              (fun i : TileIndex shape => (ptrTile.data i).region)
              (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
              (fun _ i => values.data i)
              (fun i : TileIndex shape =>
                masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
              (TileShape.allIndices shape) s dtype' shape' name

/-- Convenience wrapper around `stepStmt_atomicAdd_regs`: lift a register
hypothesis from before to after an atomic-add step. The atomic step only
writes memory, so any register `(dtype', shape', name)`'s value carries
through unchanged.

Used by FA-style backward kernels (`fa1BackwardAtomicDQ*`,
`fa2BackwardAtomicDQ*`) where 4 register hypotheses (`q_ptrs`,
`k_block_ptrs`, `dK_block`, `dV_block`, etc.) need to be lifted past the
leading atomic dQ store before the tail dK/dV stores can use them. -/
theorem stepStmt_atomicAdd_lift_regs {dtype : TileDType} (hnum : NumericDType dtype)
    {shape : TileShape} {mem : MemAccess dtype shape} {val : Op dtype shape}
    {mask : MaskOpt dtype shape} {s s' : BlockState}
    (h : stepStmt (Stmt.atomicAdd hnum shape mem val mask) s = some s')
    {dtype' : TileDType} {shape' : TileShape} {name : RegName}
    {t : Tile dtype' shape'} (hPre : s.regs dtype' shape' name = some t) :
    s'.regs dtype' shape' name = some t := by
  rw [stepStmt_atomicAdd_regs hnum h]; exact hPre

theorem stepStmt_assign_eq_some
    {dtype : TileDType} {shape : TileShape} {name : RegName}
    {e : Op dtype shape} {s : BlockState} {v : Tile dtype shape}
    (h : evalOp e s = some v) :
    stepStmt (.assign dtype shape name e) s = some (s.setReg name dtype shape v) := by
  simp [stepStmt, h]

mutual

theorem stepStmt_pid {st : Stmt} {s s' : BlockState}
    (h : stepStmt st s = some s') :
    s'.pid = s.pid := by
  cases st
  case assign dtype shape name e =>
    cases hv : evalOp e s with
    | none =>
        simp [stepStmt, hv] at h
    | some v =>
        simp [stepStmt, hv] at h
        cases h
        rfl
  case store dtype shape mem val mask =>
    cases hval : evalOp val s with
    | none =>
        unfold stepStmt at h
        simp [hval] at h
    | some values =>
        unfold stepStmt at h
        simp [hval] at h
        cases mask <;> simp at h
        · cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTyped_pid dtype (Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun i : TileIndex shape => values.data i)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTypedAt_pid dtype
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun i : TileIndex shape => values.data i)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
              BlockState.foldl_writeMemTypedAt_masked_pid dtype
                (fun i : TileIndex shape => (ptrTile.data i).region)
                (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape =>
                  (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                (TileShape.allIndices shape) s
        · rename_i mask
          cases hmask : evalOp mask s <;> simp [hmask] at h
          rename_i masks
          cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTyped_masked_pid dtype (Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTypedAt_masked_pid dtype
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
              BlockState.foldl_writeMemTypedAt_masked_pid dtype
                (fun i : TileIndex shape => (ptrTile.data i).region)
                (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape =>
                  masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                (TileShape.allIndices shape) s
        · rename_i mask other
          cases hmask : evalOp mask s <;> simp [hmask] at h
          rename_i masks
          cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTyped_masked_pid dtype (Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              BlockState.foldl_writeMemTypedAt_masked_pid dtype
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun i : TileIndex shape => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
                BlockState.foldl_writeMemTypedAt_masked_pid dtype
                  (fun i : TileIndex shape => (ptrTile.data i).region)
                  (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                  (fun i : TileIndex shape => values.data i)
                  (fun i : TileIndex shape =>
                    masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                  (TileShape.allIndices shape) s
  case atomicAdd dtype hnum shape mem val mask =>
    cases hval : evalOp val s with
    | none =>
        unfold stepStmt at h
        simp [hval] at h
    | some values =>
        unfold stepStmt at h
        simp [hval] at h
        cases mask <;> simp at h
        · cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun _ : TileIndex shape => Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun _ i => values.data i)
                (fun _ : TileIndex shape => true)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun _ i => values.data i)
                (fun _ : TileIndex shape => true)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrTile.data i).region)
                (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                (fun _ i => values.data i)
                (fun i : TileIndex shape =>
                  (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                (TileShape.allIndices shape) s
        · rename_i mask
          cases hmask : evalOp mask s <;> simp [hmask] at h
          rename_i masks
          cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun _ : TileIndex shape => Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun _ i => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun _ i => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrTile.data i).region)
                (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                (fun _ i => values.data i)
                (fun i : TileIndex shape =>
                  masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                (TileShape.allIndices shape) s
        · rename_i mask other
          cases hmask : evalOp mask s <;> simp [hmask] at h
          rename_i masks
          cases mem with
          | region region off =>
            cases hoff : evalOp off s <;> simp [hoff] at h
            rename_i offsets
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun _ : TileIndex shape => Region.cast region)
                (fun i : TileIndex shape => offsets.data i)
                (fun _ i => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | ptr ptr =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrs
            cases h
            simpa [BlockState.pid] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrs.data i).1)
                (fun i : TileIndex shape => (ptrs.data i).2)
                (fun _ i => values.data i)
                (fun i : TileIndex shape => masks.data i)
                (TileShape.allIndices shape) s
          | blockPtr ptr boundaryCheck =>
            cases hptr : evalOp ptr s <;> simp [hptr] at h
            rename_i ptrTile
            cases h
            simpa [BlockState.pid, Bool.and_eq_true] using
              foldl_atomicAddAt_pid dtype hnum
                (fun i : TileIndex shape => (ptrTile.data i).region)
                (fun i : TileIndex shape => (ptrTile.data i).address (TileShape.indexToList shape i))
                (fun _ i => values.data i)
                (fun i : TileIndex shape =>
                  masks.data i && (ptrTile.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                (TileShape.allIndices shape) s
  case atomicRMW op dtype shape mem input extraInput mask dest =>
    have h' : stepAtomicRMW op dtype shape mem input extraInput mask dest s = some s' := by
      simpa [stepStmt] using h
    exact stepAtomicRMW_pid h'
  case forLoop idx n body =>
    simp at h
    exact stepForLoopAux_pid h
  case forRange idx start stop step body =>
    simp at h
    exact stepForRangeAux_pid h
  case forRangeDyn idx start stop step body =>
    cases hstart : evalOp start s with
    | none => simp [hstart] at h
    | some startTile =>
      cases hstop : evalOp stop s with
      | none => simp [hstart, hstop] at h
      | some stopTile =>
        cases hstep : evalOp step s with
        | none => simp [hstart, hstop, hstep] at h
        | some stepTile =>
          simp [hstart, hstop, hstep] at h
          exact stepForRangeAux_pid h
  case ifThen cond body =>
    cases hcond : evalOp cond s with
    | none => simp [stepStmt, hcond] at h
    | some c =>
        simp [stepStmt, hcond] at h
        by_cases hc : c.data PUnit.unit
        · simp [hc] at h
          exact stepStmts_pid h
        · simp [hc] at h
          rw [← h]
  case ifThenElse cond thenBody elseBody =>
    cases hcond : evalOp cond s with
    | none => simp [stepStmt, hcond] at h
    | some c =>
        simp [stepStmt, hcond] at h
        by_cases hc : c.data PUnit.unit
        · simp [hc] at h
          exact stepStmts_pid h
        · simp [hc] at h
          exact stepStmts_pid h

theorem stepStmts_pid {body : List Stmt} {s s' : BlockState}
    (h : stepStmts body s = some s') :
    s'.pid = s.pid := by
  cases body with
  | nil =>
      simp at h
      simp_all
  | cons st rest =>
      unfold stepStmts at h
      cases hst : stepStmt st s <;> simp [hst] at h
      rename_i mid
      exact (stepStmts_pid h).trans (stepStmt_pid hst)

theorem stepForLoopAux_pid {idx : RegName} {start n : Nat}
    {body : List Stmt} {s s' : BlockState}
    (h : stepForLoopAux idx start n body s = some s') :
    s'.pid = s.pid := by
  by_cases hlt : start < n
  · rw [stepForLoopAux.step_lt hlt] at h
    cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) <;>
      simp [hbody] at h
    rename_i mid
    exact (stepForLoopAux_pid h).trans
      ((stepStmts_pid hbody).trans (BlockState.setReg_pid s idx .nat [] (Tile.scalar start)))
  · have hge : n ≤ start := Nat.le_of_not_gt hlt
    rw [stepForLoopAux.step_ge hge] at h
    simp_all

theorem stepForRangeAux_pid {idx : RegName} {cur stop step : Nat}
    {body : List Stmt} {s s' : BlockState}
    (h : stepForRangeAux idx cur stop step body s = some s') :
    s'.pid = s.pid := by
  by_cases hstep : step = 0
  · subst hstep
    rw [stepForRangeAux.step_zero_step] at h
    simp_all
  · by_cases hlt : cur < stop
    · rw [stepForRangeAux.step_lt hstep hlt] at h
      cases hbody : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) <;>
        simp [hbody] at h
      rename_i mid
      exact (stepForRangeAux_pid h).trans
        ((stepStmts_pid hbody).trans (BlockState.setReg_pid s idx .nat [] (Tile.scalar cur)))
    · have hge : stop ≤ cur := Nat.le_of_not_gt hlt
      rw [stepForRangeAux.step_ge hstep hge] at h
      simp_all

end

theorem exec_pid {k : Kernel} {s s' : BlockState}
    (h : exec k s = some s') :
    s'.pid = s.pid := by
  exact stepStmts_pid h

example : evalOp (.const 5) default = some (Tile.scalar (some 5)) := by
  unfold evalOp
  rfl

example : evalOp (.constNat 7) default = some (Tile.scalar 7) := by
  unfold evalOp
  rfl

example : evalOp (.programId 0) default = some (Tile.scalar 0) := by
  unfold evalOp
  rfl

example : evalOp (.add .nat .nil (.constNat 2) (.constNat 3)) default
    = some (Tile.scalar 5) := by
  simp [evalOp, Tile.bop, NumericDType.add]


/-! ## Store-free statements and the memory frame lemma

A statement is *store-free* when it performs no memory write (`store` /
`atomicAdd` / `atomicRMW`), recursively through `if`/`else` bodies. Such a
statement — and any list of them — preserves every memory cell when executed.
This is the generic frame used to show that an already-committed output store
survives a subsequent block of pure register computation (e.g. a branchy
loss/scale tail run after an `lse` store). -/

/-- A statement that performs no memory writes (no `store`/`atomicAdd`/
`atomicRMW`), recursively through `if`/`else`/loop bodies. Loops are treated
conservatively as possibly-writing. -/
def storeFree : Stmt → Bool
  | .assign _ _ _ _ => Bool.true
  | .store _ _ _ _ _ => Bool.false
  | .atomicAdd _ _ _ _ _ => Bool.false
  | .atomicRMW _ _ _ _ _ _ _ _ => Bool.false
  | .forLoop _ _ _ => Bool.false
  | .forRange _ _ _ _ _ => Bool.false
  | .forRangeDyn _ _ _ _ _ => Bool.false
  | .ifThen _ body => body.attach.all (fun ⟨st, _⟩ => storeFree st)
  | .ifThenElse _ t e =>
      t.attach.all (fun ⟨st, _⟩ => storeFree st) &&
      e.attach.all (fun ⟨st, _⟩ => storeFree st)

mutual
/-- A store-free statement preserves all of memory. -/
theorem storeFree_stepStmt_mem (st : Stmt) (s s' : BlockState)
    (hsf : storeFree st = Bool.true) (h : stepStmt st s = some s') :
    s'.mem = s.mem := by
  match st with
  | .assign dtype shape name e =>
      cases hv : evalOp e s with
      | none => simp [stepStmt, hv] at h
      | some v => simp [stepStmt, hv] at h; subst h; rfl
  | .store _ _ _ _ _ => simp [storeFree] at hsf
  | .atomicAdd _ _ _ _ _ => simp [storeFree] at hsf
  | .atomicRMW _ _ _ _ _ _ _ _ => simp [storeFree] at hsf
  | .ifThen cond body =>
      cases hc : evalOp cond s with
      | none => simp [stepStmt, hc] at h
      | some c =>
          by_cases hb : c.data PUnit.unit
          · simp [stepStmt, hc, hb] at h
            exact storeFree_stepStmts_mem body s s' (by simpa [storeFree] using hsf) h
          · simp [stepStmt, hc, hb] at h; subst h; rfl
  | .ifThenElse cond t e =>
      cases hc : evalOp cond s with
      | none => simp [stepStmt, hc] at h
      | some c =>
          simp only [storeFree, Bool.and_eq_true] at hsf
          by_cases hb : c.data PUnit.unit
          · simp [stepStmt, hc, hb] at h
            exact storeFree_stepStmts_mem t s s' (by simpa using hsf.1) h
          · simp [stepStmt, hc, hb] at h
            exact storeFree_stepStmts_mem e s s' (by simpa using hsf.2) h
  | .forLoop _ _ _ => simp [storeFree] at hsf
  | .forRange _ _ _ _ _ => simp [storeFree] at hsf
  | .forRangeDyn _ _ _ _ _ => simp [storeFree] at hsf

/-- A list of store-free statements preserves all of memory. -/
theorem storeFree_stepStmts_mem (stmts : List Stmt) (s s' : BlockState)
    (hsf : stmts.all (fun st => storeFree st) = Bool.true)
    (h : stepStmts stmts s = some s') :
    s'.mem = s.mem := by
  match stmts with
  | [] => rw [stepStmts.nil] at h; injection h with h; subst h; rfl
  | st :: rest =>
      simp only [List.all_cons, Bool.and_eq_true] at hsf
      cases hstep : stepStmt st s with
      | none =>
          rw [show stepStmts (st :: rest) s = none from by
            conv_lhs => unfold stepStmts
            rw [hstep]] at h
          exact absurd h (by simp)
      | some smid =>
          rw [stepStmts.cons_some hstep] at h
          have h1 := storeFree_stepStmt_mem st s smid hsf.1 hstep
          have h2 := storeFree_stepStmts_mem rest smid s' hsf.2 h
          rw [h2, h1]
end

end VeriTile
