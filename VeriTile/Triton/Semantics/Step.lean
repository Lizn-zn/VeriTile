/-
VeriTile.Triton.Semantics.Step

Statement stepping, kernel execution, and execution-state preservation lemmas.
-/

import VeriTile.Triton.Semantics.EvalOp

namespace VeriTile.Triton

mutual

noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign dtype shape name e, s => do
      let v ← evalOp e s
      some (s.setReg name dtype shape v)
  | .store region shape off val, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      -- `mem : RegionName → Nat → ℝ`; demote `WithBot ℝ → ℝ` via `unbotD 0`.
      -- Well-formed kernels never store `⊥`, so the default value is unobservable.
      some ((TileShape.allIndices shape).foldl
        (fun acc i => acc.writeMem region (offsets.data i)
                        ((values.data i).unbotD 0)) s)
  | .storeMask region shape off val mask, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          if masks.data i then acc.writeMem region (offsets.data i)
                                ((values.data i).unbotD 0)
          else acc) s)
  | .storePtr shape ptr val, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          acc.writeMem p.1 p.2 ((values.data i).unbotD 0)) s)
  | .storePtrMask shape ptr val mask, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          if masks.data i then acc.writeMem p.1 p.2 ((values.data i).unbotD 0)
          else acc) s)
  | .storeFloat h region shape off val, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i => acc.writeMem region (offsets.data i)
                        (h.storeValue (values.data i))) s)
  | .storeFloatMask h region shape off val mask, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          if masks.data i then acc.writeMem region (offsets.data i)
                                (h.storeValue (values.data i))
          else acc) s)
  | .storePtrFloat h shape ptr val, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          acc.writeMem p.1 p.2 (h.storeValue (values.data i))) s)
  | .storePtrFloatMask h shape ptr val mask, s => do
      let ptrs ← evalOp ptr s
      let values ← evalOp val s
      let masks ← evalOp mask s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let p := ptrs.data i
          if masks.data i then acc.writeMem p.1 p.2 (h.storeValue (values.data i))
          else acc) s)
  | .storeBlockPtr shape ptr val boundaryCheck, s => do
      let ptrTile ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let bp := ptrTile.data i
          let idx := TileShape.indexToList shape i
          if bp.inBounds idx boundaryCheck then
            acc.writeMem bp.region (bp.address idx) ((values.data i).unbotD 0)
          else acc) s)
  | .storeBlockPtrFloat h shape ptr val boundaryCheck, s => do
      let ptrTile ← evalOp ptr s
      let values ← evalOp val s
      some ((TileShape.allIndices shape).foldl
        (fun acc i =>
          let bp := ptrTile.data i
          let idx := TileShape.indexToList shape i
          if bp.inBounds idx boundaryCheck then
            acc.writeMem bp.region (bp.address idx) (h.storeValue (values.data i))
          else acc) s)
  | .forLoop idx n body, s =>
      stepForLoopAux idx 0 n body s
  | .ifThen cond body, s => do
      let c ← evalOp cond s
      if c.data PUnit.unit then stepStmts body s else some s
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
  case store region shape off val =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMem_pid]
  case storeMask region shape off val mask =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storePtr shape ptr val =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_pid]
  case storePtrMask shape ptr val mask =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storeFloat hfloat region shape off val =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMem_pid]
  case storeFloatMask hfloat region shape off val mask =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storePtrFloat hfloat shape ptr val =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_pid]
  case storePtrFloatMask hfloat shape ptr val mask =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrs
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storeBlockPtr shape ptr val boundaryCheck =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrTile
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
  case storeBlockPtrFloat hfloat shape ptr val boundaryCheck =>
    cases hptr : evalOp ptr s <;> simp [stepStmt, hptr] at h
    rename_i ptrTile
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases h
    simp [BlockState.foldl_writeMemAt_masked_pid]
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
