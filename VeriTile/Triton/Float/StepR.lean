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

/-! ## Cast-free degeneration

A loop whose body steps identically under `execR R` and `exec` (e.g. a body
with no `castFloat` / rounded store) folds identically too — the ∀-`R`
analogue of `stepForLoopAuxR_triv`. Supply the body-level degeneration
`hbody` by `simp only [body, stepStmtsR, stepStmts, stepStmtR, stepStmt,
evalOpR.eq_def, evalOp]; rfl` on the concrete cast-free body. -/

/-- `forLoop` auxiliary cast-free degeneration: if the body steps identically
under `stepStmtsR R` and `stepStmts`, so does the whole loop fold. -/
theorem stepForLoopAuxR_castFree (R : RoundingModel) (body : List Stmt)
    (hbody : ∀ t : BlockState, stepStmtsR R body t = stepStmts body t) (idx : RegName) :
    ∀ (start n : Nat) (s : BlockState),
      stepForLoopAuxR R idx start n body s = stepForLoopAux idx start n body s
  | start, n, s => by
      rw [stepForLoopAuxR, stepForLoopAux]
      simp only [hbody (s.setReg idx .nat [] (Tile.scalar start))]
      split
      · cases stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rfl
        | some s' => exact stepForLoopAuxR_castFree R body hbody idx (start + 1) n s'
      · rfl
  termination_by start n _ => n - start
  decreasing_by omega

/-- `forRange` auxiliary cast-free degeneration: the `forRange` mirror of
`stepForLoopAuxR_castFree`. If the body steps identically under
`stepStmtsR R` and `stepStmts`, the whole strided fold does too. -/
theorem stepForRangeAuxR_castFree (R : RoundingModel) (body : List Stmt)
    (hbody : ∀ t : BlockState, stepStmtsR R body t = stepStmts body t) (idx : RegName) :
    ∀ (cur stop step : Nat) (s : BlockState),
      stepForRangeAuxR R idx cur stop step body s = stepForRangeAux idx cur stop step body s
  | cur, stop, step, s => by
      rw [stepForRangeAuxR, stepForRangeAux]
      simp only [hbody (s.setReg idx .nat [] (Tile.scalar cur))]
      split
      · rfl
      · split
        · cases stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rfl
          | some s' =>
              exact stepForRangeAuxR_castFree R body hbody idx (cur + step) stop step s'
        · rfl
  termination_by cur stop _ _ => stop - cur
  decreasing_by omega

/-! ## R-scatter readback (#447 Phase C)

The `writeMemAsR` mirror of the fp16 `MemCell` scatter-readback family in
`VeriTile.Triton.KernelLemmas.ScatterStore` (`scatter_memcell_fp16_prop_masked_nd`).
Narrow float cells are **not** decodable by `BlockState.readMem` (it reads the
`.real` channel only, and `MemCell.readAs` is tag-exact), so — exactly like
the existing fp16-store bench proofs — the readback is stated at the `MemCell`
layer: the cell an active lane leaves behind is
`MemCell.of dtype.toTileDType (dtype.ofReal (R.storeValue dtype v))`. -/

namespace BlockState

/-- `writeMemAsR` only rewrites `mem`, so register reads pass through it —
lets a later store's value register be read over an earlier store. -/
@[simp] theorem writeMemAsR_regs (R : RoundingModel) (s : BlockState)
    (d : FloatDType) (reg : RegionName) (o : Nat) (v : TileCarrier d.toTileDType)
    (dt : TileDType) (sh : TileShape) (nm : RegName) :
    (s.writeMemAsR R d reg o v).regs dt sh nm = s.regs dt sh nm := rfl

/-- `writeMemTypedR` at a concrete narrow-float tag reduces to `writeMemAsR`
(store-side rounding-event site). -/
@[simp] theorem writeMemTypedR_fp32 (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier .fp32) :
    s.writeMemTypedR R .fp32 region offset v =
      s.writeMemAsR R .fp32 region offset v := rfl

@[simp] theorem writeMemTypedR_fp16 (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier .fp16) :
    s.writeMemTypedR R .fp16 region offset v =
      s.writeMemAsR R .fp16 region offset v := rfl

@[simp] theorem writeMemTypedR_bf16 (R : RoundingModel) (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier .bf16) :
    s.writeMemTypedR R .bf16 region offset v =
      s.writeMemAsR R .bf16 region offset v := rfl

/-- Single-cell readback of the R-write, at the `MemCell` layer. -/
theorem writeMemAsR_mem (R : RoundingModel) (dtype : FloatDType)
    (s : BlockState) (region : RegionName) (offset : Nat)
    (v : TileCarrier dtype.toTileDType) (r : RegionName) (o : Nat) :
    (s.writeMemAsR R dtype region offset v).mem r o =
      if r = region ∧ o = offset then
        MemCell.of dtype.toTileDType (dtype.ofReal (R.storeValue dtype v))
      else s.mem r o := rfl

/-- A masked `writeMemAsR` scatter `foldl` preserves any cell not hit by an
active lane. -/
theorem foldl_writeMemAsR_preserve_masked {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype.toTileDType)
    (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ s : BlockState, (∀ k ∈ l, mask k = true → offsetFn k ≠ o) →
      ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
        s).mem region o) = s.mem region o := by
  induction l with
  | nil => intro s _h; rfl
  | cons hd tl ih =>
      intro s h
      rw [List.foldl_cons]
      have htl : ∀ k ∈ tl, mask k = true → offsetFn k ≠ o :=
        fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
      by_cases hmaskhd : mask hd = true
      · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
        simp only [hmaskhd, if_true]
        rw [ih _ htl, writeMemAsR_mem]
        rw [if_neg (by intro hsame; exact hhd hsame.2.symm)]
      · have hmaskhd' : mask hd = false := by
          cases hm : mask hd
          · rfl
          · exact absurd hm hmaskhd
        simp only [hmaskhd', if_false, Bool.false_eq_true]
        exact ih _ htl

/-- An unmasked `writeMemAsR` scatter `foldl` leaves every memory cell it
does not hit unchanged — the cell-level frame for unmasked rounded stores
(the `execR` sibling of the exact-store scatter frame, and the unmasked
sibling of `foldl_writeMemAsR_preserve_masked`). -/
theorem foldl_writeMemAsR_preserve_cell {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype.toTileDType)
    (r : RegionName) (o : Nat) (l : List α)
    (hnot : ∀ k ∈ l, ¬(region = r ∧ offsetFn k = o)) :
    ∀ s : BlockState,
      (l.foldl (fun acc k =>
          acc.writeMemAsR R dtype region (offsetFn k) (valueFn k)) s).mem r o
        = s.mem r o := by
  induction l with
  | nil => intro _; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons,
        ih (fun k hk => hnot k (List.mem_cons_of_mem hd hk)) _,
        BlockState.writeMemAsR_mem]
      exact if_neg (fun hc => hnot hd List.mem_cons_self ⟨hc.1.symm, hc.2.symm⟩)

/-- Readback of a `P`-masked `writeMemAsR` scatter store at lane `i`'s offset:
the `R`-rounded stored cell if `P i`, else the prior contents, given injective
offsets. `R`-mirror of `scatter_memcell_fp16_prop_masked_nd`. -/
theorem scatter_memcell_R_prop_masked_nd (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → TileCarrier dtype.toTileDType)
    (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if P k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
       s).mem region (offsetFn i)
    = if P i then
        MemCell.of dtype.toTileDType (dtype.ofReal (R.storeValue dtype (valueFn i)))
      else
        s.mem region (offsetFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (TileShape.mem_allIndices shape i)
  have h_nodup := TileShape.allIndices_nodup shape
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [hl, List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, decide (P k) = true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  have hstep :
      (fun (acc : BlockState) k =>
        if P k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k)
        else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  rw [hstep]
  rw [foldl_writeMemAsR_preserve_masked R dtype offsetFn valueFn (fun k => decide (P k))
    (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hPi : P i
  · simp only [hPi, if_true]
    rw [writeMemAsR_mem, if_pos ⟨rfl, rfl⟩]
  · simp only [hPi, if_false]
    rw [foldl_writeMemAsR_preserve_masked R dtype offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₁ _ h_l1_not_in]

/-- The R-write leaves the program ids untouched. -/
@[simp] theorem writeMemAsR_pids (R : RoundingModel) (dtype : FloatDType)
    (s : BlockState) (region : RegionName) (offset : Nat)
    (v : TileCarrier dtype.toTileDType) :
    (s.writeMemAsR R dtype region offset v).pids = s.pids := rfl

/-- A `P`-masked `writeMemAsR` scatter `foldl` leaves the program ids
untouched. -/
theorem foldl_writeMemAsR_masked_pids {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype.toTileDType)
    (P : α → Prop) [DecidablePred P] (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
        if P k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
        s).pids) = s.pids := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons]
      by_cases h : P hd
      · simp only [h, if_true]
        rw [ih, writeMemAsR_pids]
      · simp only [h, if_false]
        exact ih s

/-- A `P`-masked `writeMemAsR` scatter `foldl` into `region` leaves every
other region's cells untouched (the frame the pipeline composition needs). -/
theorem foldl_writeMemAsR_preserve_other_region {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype.toTileDType)
    (P : α → Prop) [DecidablePred P]
    (r : RegionName) (hr : r ≠ region) (o : Nat) (l : List α) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
        if P k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
        s).mem r o) = s.mem r o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons]
      by_cases h : P hd
      · simp only [h, if_true]
        rw [ih, writeMemAsR_mem, if_neg (fun hc => hr hc.1)]
      · simp only [h, if_false]
        exact ih s

/-- Tag-exact readback of a bf16 cell through the operational `tl.load`
channel: `readMemValue .bf16` on a `MemCell.of .bf16 (some z)` cell returns
`some z` (the finite-fallback round trip is the identity on `some`). -/
theorem readMemValue_bf16_of_cell {s : BlockState} {region : RegionName}
    {offset : Nat} {z : ℝ}
    (h : s.mem region offset = MemCell.of .bf16 (some z : WithBot ℝ)) :
    s.readMemValue .bf16 region offset = (some z : WithBot ℝ) := by
  simp [readMemValue, readMemAs, h, FloatDType.storeValue, FloatDType.ofReal]

/-- A `P`-masked `writeMemAsR` scatter preserves every same-region offset not
hit by an active lane (the `Prop`-mask twin of
`foldl_writeMemAsR_preserve_masked`, for the writes-equality frame proofs). -/
theorem foldl_writeMemAsR_preserve_masked_prop {α : Type} (R : RoundingModel)
    (dtype : FloatDType) {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → TileCarrier dtype.toTileDType)
    (P : α → Prop) [DecidablePred P] (o : Nat) (l : List α)
    (ho : ∀ k ∈ l, P k → offsetFn k ≠ o) :
    ∀ s : BlockState,
      ((l.foldl (fun acc k =>
        if P k then acc.writeMemAsR R dtype region (offsetFn k) (valueFn k) else acc)
        s).mem region o) = s.mem region o := by
  induction l with
  | nil => intro s; rfl
  | cons hd tl ih =>
      intro s
      rw [List.foldl_cons]
      by_cases h : P hd
      · simp only [h, if_true]
        rw [ih (fun k hk hPk => ho k (List.mem_cons_of_mem hd hk) hPk),
          writeMemAsR_mem,
          if_neg (fun hc => ho hd (List.mem_cons_self) h hc.2.symm)]
      · simp only [h, if_false]
        exact ih (fun k hk hPk => ho k (List.mem_cons_of_mem hd hk) hPk) s

end BlockState

end VeriTile.Triton

/-! ## Sequential decomposition -/

namespace VeriTile.Triton

/-- `stepStmtsR` splits over list concatenation: running `xs ++ ys` is
running `xs`, then (if it succeeds) running `ys` from the resulting state.
The n-ary pipeline split (`execR_toAlgKernel_seq`) is built on this. -/
theorem stepStmtsR_append (R : RoundingModel) (xs ys : List Stmt) (s : BlockState) :
    stepStmtsR R (xs ++ ys) s = (stepStmtsR R xs s).bind (fun s' => stepStmtsR R ys s') := by
  induction xs generalizing s with
  | nil => simp [stepStmtsR]
  | cons st rest ih =>
      simp [stepStmtsR]
      cases stepStmtR R st s <;> simp [ih]

end VeriTile.Triton
