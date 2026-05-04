/-
VeriTile.Triton.MemoryFrame

Layer-2a predicate-level memory footprint and frame primitives.
-/

import VeriTile.Triton.MemoryBounds
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

/-!
This module implements #60's Layer-2a frame interface. The footprint
representation is intentionally predicate-level: structured footprints can
refine into this API later, while #49 can already consume a stable semantic
contract.
-/

/-- Address of one logical HBM cell. Offsets are cell offsets, not byte addresses. -/
abbrev MemCellAddr := RegionName × Nat

/-- Predicate-level write footprint. -/
abbrev WriteFootprint := MemCellAddr → Prop

namespace WriteFootprint

/-- Empty write footprint. -/
def empty : WriteFootprint := fun _ => False

/-- Singleton write footprint. -/
def singleton (region : RegionName) (offset : Nat) : WriteFootprint :=
  fun addr => addr = (region, offset)

/-- Union of two write footprints. -/
def union (P Q : WriteFootprint) : WriteFootprint :=
  fun addr => P addr ∨ Q addr

/-- Predicate-level disjointness for write footprints. -/
def disjoint (P Q : WriteFootprint) : Prop :=
  ∀ addr, P addr → Q addr → False

@[simp] theorem not_mem_empty (addr : MemCellAddr) :
    ¬ empty addr := by
  simp [empty]

@[simp] theorem mem_singleton {region : RegionName} {offset : Nat} :
    singleton region offset (region, offset) := by
  simp [singleton]

@[simp] theorem mem_union {P Q : WriteFootprint} {addr : MemCellAddr} :
    union P Q addr ↔ P addr ∨ Q addr := Iff.rfl

theorem disjoint_comm {P Q : WriteFootprint} :
    disjoint P Q → disjoint Q P := by
  intro h addr hq hp
  exact h addr hp hq

end WriteFootprint

namespace BlockState

/-- Two states agree on all cells inside `P`. -/
def AgreesOn (P : WriteFootprint) (s₀ s₁ : BlockState) : Prop :=
  ∀ region offset, P (region, offset) → s₀.mem region offset = s₁.mem region offset

/-- Two states agree on all cells outside `P`. -/
def AgreesOutside (P : WriteFootprint) (s₀ s₁ : BlockState) : Prop :=
  ∀ region offset, ¬ P (region, offset) → s₀.mem region offset = s₁.mem region offset

/-- The transition from `s₀` to `s₁` only modified cells inside `P`.

This is definitionally equal to `AgreesOutside`; the separate name is the
theorem-facing frame contract used by #49. -/
def WriteWithin (P : WriteFootprint) (s₀ s₁ : BlockState) : Prop :=
  AgreesOutside P s₀ s₁

@[simp] theorem agreesOutside_empty {s₀ s₁ : BlockState} :
    AgreesOutside WriteFootprint.empty s₀ s₁ ↔ s₀.mem = s₁.mem := by
  constructor
  · intro h
    funext region offset
    exact h region offset (by simp [WriteFootprint.empty])
  · intro h region offset _
    exact congrFun (congrFun h region) offset

@[simp] theorem agreesOutside_union {P Q : WriteFootprint} {s₀ s₁ : BlockState} :
    AgreesOutside (WriteFootprint.union P Q) s₀ s₁ ↔
      ∀ region offset,
        ¬ P (region, offset) → ¬ Q (region, offset) →
          s₀.mem region offset = s₁.mem region offset := by
  constructor
  · intro h region offset hp hq
    exact h region offset (by
      intro hpq
      cases hpq with
      | inl hp' => exact hp hp'
      | inr hq' => exact hq hq')
  · intro h region offset hpq
    exact h region offset
      (fun hp => hpq (Or.inl hp))
      (fun hq => hpq (Or.inr hq))

@[simp] theorem writeWithin_empty {s₀ s₁ : BlockState} :
    WriteWithin WriteFootprint.empty s₀ s₁ ↔ s₀.mem = s₁.mem := by
  exact agreesOutside_empty

theorem writeWithin_refl (P : WriteFootprint) (s : BlockState) :
    WriteWithin P s s := by
  intro _ _ _
  rfl

theorem writeWithin_trans {P : WriteFootprint} {s₀ s₁ s₂ : BlockState}
    (h₀₁ : WriteWithin P s₀ s₁) (h₁₂ : WriteWithin P s₁ s₂) :
    WriteWithin P s₀ s₂ := by
  intro region offset hnot
  exact (h₀₁ region offset hnot).trans (h₁₂ region offset hnot)

theorem writeMemTyped_writeWithin {P : WriteFootprint} {s : BlockState}
    (dtype : TileDType) (region : RegionName) (offset : Nat)
    (value : TileCarrier dtype) (hmem : P (region, offset)) :
    WriteWithin P s (s.writeMemTyped dtype region offset value) := by
  intro r o hnot
  by_cases h : r = region ∧ o = offset
  · rcases h with ⟨rfl, rfl⟩
    exact False.elim (hnot hmem)
  · cases dtype <;> simp [writeMemTyped, writeMemAs, h]

theorem foldl_writeMemTypedAt_masked_writeWithin {α : Type} {P : WriteFootprint}
    (dtype : TileDType) (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype) (mask : α → Bool) (l : List α)
    (s : BlockState)
    (hmem : ∀ i, i ∈ l → mask i = true → P (regionFn i, offsetFn i)) :
    WriteWithin P s
      (l.foldl
        (fun acc i =>
          if mask i then acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
          else acc) s) := by
  induction l generalizing s with
  | nil =>
      exact writeWithin_refl P s
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hmask : mask hd = true
      · simp only [hmask, if_true]
        have hfirst :
            WriteWithin P s
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd)) :=
          writeMemTyped_writeWithin dtype (regionFn hd) (offsetFn hd)
            (valueFn hd) (hmem hd (List.mem_cons_self) hmask)
        have htail :
            WriteWithin P
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd))
              (tl.foldl
                (fun acc i =>
                  if mask i then acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
                  else acc)
                (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd))) :=
          ih _ (fun i hi hmi => hmem i (List.mem_cons_of_mem hd hi) hmi)
        exact writeWithin_trans hfirst htail
      · have hmaskFalse : mask hd = false := by
          cases h : mask hd
          · rfl
          · exact False.elim (hmask h)
        simp only [hmaskFalse]
        exact ih s (fun i hi hmi => hmem i (List.mem_cons_of_mem hd hi) hmi)

theorem foldl_writeMemTypedAt_prop_writeWithin {α : Type} {P : WriteFootprint}
    (dtype : TileDType) (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype) (active : α → Prop) [DecidablePred active]
    (l : List α) (s : BlockState)
    (hmem : ∀ i, i ∈ l → active i → P (regionFn i, offsetFn i)) :
    WriteWithin P s
      (l.foldl
        (fun acc i =>
          if active i then acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
          else acc) s) := by
  induction l generalizing s with
  | nil =>
      exact writeWithin_refl P s
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hactive : active hd
      · simp only [hactive, if_true]
        have hfirst :
            WriteWithin P s
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd)) :=
          writeMemTyped_writeWithin dtype (regionFn hd) (offsetFn hd)
            (valueFn hd) (hmem hd (List.mem_cons_self) hactive)
        have htail :
            WriteWithin P
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd))
              (tl.foldl
                (fun acc i =>
                  if active i then acc.writeMemTyped dtype (regionFn i) (offsetFn i) (valueFn i)
                  else acc)
                (s.writeMemTyped dtype (regionFn hd) (offsetFn hd) (valueFn hd))) :=
          ih _ (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai)
        exact writeWithin_trans hfirst htail
      · simp only [hactive, if_false]
        exact ih s (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai)

end BlockState

namespace Stmt

/-- Active store addresses are included in the supplied footprint.

This predicate mirrors `stepStmt`'s store cases: masked-off lanes and checked
block-pointer OOB lanes do not write and therefore need no footprint membership. -/
def StoreAddressesWithin (P : WriteFootprint) (s : BlockState)
    (mem : MemAccess shape) (mask : MaskOpt dtype shape) : Prop :=
  match mask with
  | .none =>
      match mem with
      | .region region off =>
          ∀ offsets, evalOp off s = some offsets →
            ∀ i : TileIndex shape, P (region, offsets.data i)
      | .ptr ptr =>
          ∀ ptrs, evalOp ptr s = some ptrs →
            ∀ i : TileIndex shape, P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ ptrs, evalOp ptr s = some ptrs →
            ∀ i : TileIndex shape,
              (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                P ((ptrs.data i).region,
                  (ptrs.data i).address (TileShape.indexToList shape i))
  | .mask maskOp =>
      match mem with
      | .region region off =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ offsets, evalOp off s = some offsets →
              ∀ i : TileIndex shape, masks.data i = true → P (region, offsets.data i)
      | .ptr ptr =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape, masks.data i = true → P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                masks.data i = true →
                (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                  P ((ptrs.data i).region,
                    (ptrs.data i).address (TileShape.indexToList shape i))
  | .maskOther maskOp _ =>
      match mem with
      | .region region off =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ offsets, evalOp off s = some offsets →
              ∀ i : TileIndex shape, masks.data i = true → P (region, offsets.data i)
      | .ptr ptr =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape, masks.data i = true → P (ptrs.data i)
      | .blockPtr ptr boundaryCheck =>
          ∀ masks, evalOp maskOp s = some masks →
            ∀ ptrs, evalOp ptr s = some ptrs →
              ∀ i : TileIndex shape,
                masks.data i = true →
                (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true →
                  P ((ptrs.data i).region,
                    (ptrs.data i).address (TileShape.indexToList shape i))

end Stmt

theorem stepStmt_store_writeWithin {P : WriteFootprint} {s s' : BlockState}
    {dtype : TileDType} {shape : TileShape}
    {mem : MemAccess shape} {value : Op dtype shape} {mask : MaskOpt dtype shape}
    (hStep : stepStmt (Stmt.store dtype shape mem value mask) s = some s')
    (hAddr : Stmt.StoreAddressesWithin P s mem mask) :
    BlockState.WriteWithin P s s' := by
  cases hvalue : evalOp value s with
  | none =>
      unfold stepStmt at hStep
      simp [hvalue] at hStep
  | some values =>
      cases mask with
      | none =>
          cases mem with
          | region region off =>
              cases hoff : evalOp off s with
              | none => simp [stepStmt, hvalue, hoff] at hStep
              | some offsets =>
                  simp [stepStmt, hvalue, hoff] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun _ : TileIndex shape => region)
                    (fun i : TileIndex shape => offsets.data i)
                    (fun i : TileIndex shape => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr offsets hoff i)
          | ptr ptr =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun i : TileIndex shape => (ptrs.data i).1)
                    (fun i : TileIndex shape => (ptrs.data i).2)
                    (fun i : TileIndex shape => values.data i)
                    (fun _ : TileIndex shape => true)
                    (TileShape.allIndices shape) s
                    (fun i hi _ => hAddr ptrs hptr i)
          | blockPtr ptr boundaryCheck =>
              cases hptr : evalOp ptr s with
              | none => simp [stepStmt, hvalue, hptr] at hStep
              | some ptrs =>
                  simp [stepStmt, hvalue, hptr] at hStep
                  cases hStep
                  exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                    (fun i : TileIndex shape => (ptrs.data i).region)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).address (TileShape.indexToList shape i))
                    (fun i : TileIndex shape => values.data i)
                    (fun i : TileIndex shape =>
                      (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck)
                    (TileShape.allIndices shape) s
                    (fun i hi hactive => hAddr ptrs hptr i hactive)
      | mask maskOp =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_prop_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)
      | maskOther maskOp other =>
          cases hmasks : evalOp maskOp s with
          | none => simp [stepStmt, hvalue, hmasks] at hStep
          | some masks =>
              cases mem with
              | region region off =>
                  cases hoff : evalOp off s with
                  | none => simp [stepStmt, hvalue, hmasks, hoff] at hStep
                  | some offsets =>
                      simp [stepStmt, hvalue, hmasks, hoff] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun _ : TileIndex shape => region)
                        (fun i : TileIndex shape => offsets.data i)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks offsets hoff i hactive)
              | ptr ptr =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_masked_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).1)
                        (fun i : TileIndex shape => (ptrs.data i).2)
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape => masks.data i)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => hAddr masks hmasks ptrs hptr i hactive)
              | blockPtr ptr boundaryCheck =>
                  cases hptr : evalOp ptr s with
                  | none => simp [stepStmt, hvalue, hmasks, hptr] at hStep
                  | some ptrs =>
                      simp [stepStmt, hvalue, hmasks, hptr] at hStep
                      cases hStep
                      exact BlockState.foldl_writeMemTypedAt_prop_writeWithin dtype
                        (fun i : TileIndex shape => (ptrs.data i).region)
                        (fun i : TileIndex shape =>
                          (ptrs.data i).address (TileShape.indexToList shape i))
                        (fun i : TileIndex shape => values.data i)
                        (fun i : TileIndex shape =>
                          masks.data i = true ∧
                            (ptrs.data i).inBounds (TileShape.indexToList shape i) boundaryCheck = true)
                        (TileShape.allIndices shape) s
                        (fun i hi hactive => by
                          have hm : masks.data i = true := hactive.1
                          have hb : (ptrs.data i).inBounds
                              (TileShape.indexToList shape i) boundaryCheck = true := hactive.2
                          exact hAddr masks hmasks ptrs hptr i hm hb)

namespace Kernel

/-- Successful single-program execution with an explicit write footprint. -/
structure ExecFrame (k : Kernel) (s : BlockState) where
  final : BlockState
  writes : WriteFootprint
  h_exec : exec k s = some final
  h_writeWithin : BlockState.WriteWithin writes s final

/-- Reusable single-kernel frame predicate. If execution fails, this is vacuous;
use `ExecFrame` when a concrete successful final state is needed. -/
def ExecWritesWithin (k : Kernel) (s : BlockState) (P : WriteFootprint) : Prop :=
  ∀ s', exec k s = some s' → BlockState.WriteWithin P s s'

end Kernel

namespace ComputeKernel

/-- Compute-kernel frame predicate through the algorithm projection. -/
def ExecWritesWithin (ck : ComputeKernel) (s : BlockState) (P : WriteFootprint) : Prop :=
  match ck.toAlgorithm? with
  | Except.ok k => k.ExecWritesWithin s P
  | Except.error _ => False

@[simp] theorem execWritesWithin_fromAlg (k : Kernel) (s : BlockState)
    (P : WriteFootprint) :
    (ComputeKernel.fromAlg k).ExecWritesWithin s P = k.ExecWritesWithin s P := rfl

end ComputeKernel

end VeriTile.Triton
