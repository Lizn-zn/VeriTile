/-
VeriTile.Triton.Semantics.Step

Statement stepping, kernel execution, and execution-state preservation lemmas.
-/

import VeriTile.Triton.Semantics.EvalOp

namespace VeriTile.Triton

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
    (mem : MemAccess shape) (input : Op dtype shape)
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
        foldAtomicRMWIndices op (fun _ => region) (fun i => offsets.data i)
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
    (mem : MemAccess shape) (input : Op dtype shape)
    (extraInput : Option (Op dtype shape)) (mask : MaskOpt dtype shape)
    (dest : Option RegName) (s : BlockState) : Option BlockState :=
  (stepAtomicRMWRaw op dtype shape mem input extraInput mask dest s).map
    (fun s' => { s' with pids := s.pids })

private theorem stepAtomicRMW_pid
    {op : RMWOp} {dtype : TileDType} {shape : TileShape}
    {mem : MemAccess shape} {input : Op dtype shape}
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
              if active i then acc.writeMemTyped dtype region (offsets.data i) (writeValue i)
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
                acc.writeMemTyped dtype region (offsets.data i)
                  (atomicValue acc region (offsets.data i) i)
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
  | .ifThen cond body, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts body s else some s
  | .ifThenElse cond thenBody elseBody, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts thenBody s else stepStmts elseBody s
termination_by st _ => (sizeOf st, 0)
decreasing_by
  all_goals (try (simp_wf; omega))
  simp_wf
  have h : 0 < sizeOf idx := by
    cases idx; simp
  omega

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

noncomputable def exec (k : Kernel) (s : BlockState) : Option BlockState :=
  stepStmts k.body s

private theorem foldl_atomicAddAt_pid {α : Type}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype) (active : α → Bool)
    (l : List α) (s : BlockState) :
    (l.foldl
      (fun acc i =>
        if active i then
          acc.writeMemTyped dtype (regionFn i) (offsetFn i)
            (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
        else acc) s).pid = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hactive : active hd = true
      · simp only [hactive, if_true]
        rw [ih]
        simp
      · have hfalse : active hd = false := by
          cases h' : active hd
          · rfl
          · exact False.elim (hactive h')
        simp only [hfalse]
        exact ih s

private theorem foldl_atomicAddAt_regs {α : Type}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype) (active : α → Bool)
    (l : List α) (s : BlockState)
    (dtype' : TileDType) (shape' : TileShape) (name : RegName) :
    (l.foldl
      (fun acc i =>
        if active i then
          acc.writeMemTyped dtype (regionFn i) (offsetFn i)
            (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
        else acc) s).regs dtype' shape' name = s.regs dtype' shape' name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hactive : active hd = true
      · simp only [hactive, if_true]
        rw [ih]
        simp
      · have hfalse : active hd = false := by
          cases h' : active hd
          · rfl
          · exact False.elim (hactive h')
        simp only [hfalse]
        exact ih s

theorem stepStmt_atomicAdd_regs {dtype : TileDType} (hnum : NumericDType dtype)
    {shape : TileShape} {mem : MemAccess shape} {val : Op dtype shape}
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
              (fun _ : TileIndex shape => region)
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
              (fun _ : TileIndex shape => region)
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
              (fun _ : TileIndex shape => region)
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
              BlockState.foldl_writeMemTyped_pid dtype region
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
              BlockState.foldl_writeMemTyped_masked_pid dtype region
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
              BlockState.foldl_writeMemTyped_masked_pid dtype region
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
                (fun _ : TileIndex shape => region)
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
                (fun _ : TileIndex shape => region)
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
                (fun _ : TileIndex shape => region)
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
  unfold evalOp Tile.bop NumericDType.add
  rfl



end VeriTile.Triton
