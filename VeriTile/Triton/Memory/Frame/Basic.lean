/-
VeriTile.Triton.Memory.Frame.Basic

Layer-2a predicate-level memory footprint and frame primitives.
-/

import VeriTile.Triton.Memory.Bounds
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

namespace WriteWithin

theorem mem_eq_of_not_written {P : WriteFootprint} {s₀ s₁ : BlockState}
    (h : BlockState.WriteWithin P s₀ s₁)
    {region : RegionName} {offset : Nat}
    (hnot : ¬ P (region, offset)) :
    s₁.mem region offset = s₀.mem region offset :=
  (h region offset hnot).symm

theorem mem_eq_of_region_not_written {P : WriteFootprint} {s₀ s₁ : BlockState}
    (h : BlockState.WriteWithin P s₀ s₁)
    {region : RegionName}
    (hnot : ∀ offset, ¬ P (region, offset)) :
    s₁.mem region = s₀.mem region := by
  funext offset
  exact mem_eq_of_not_written h (hnot offset)

end WriteWithin

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

/-- Single real `writeMem` stays within a footprint that contains its cell. -/
theorem writeMem_writeWithin {P : WriteFootprint} {s : BlockState}
    (region : RegionName) (offset : Nat) (v : ℝ) (hmem : P (region, offset)) :
    WriteWithin P s (s.writeMem region offset v) := by
  intro r o hnot
  by_cases h : r = region ∧ o = offset
  · rcases h with ⟨rfl, rfl⟩; exact absurd hmem hnot
  · simp [writeMem, h]

/-- A masked scatter of real `writeMem`s over a list stays within a footprint
that contains every active cell. The `writeMem` (real) analogue of
`foldl_writeMemTypedAt_prop_writeWithin`. -/
theorem foldl_writeMem_prop_writeWithin {α : Type} {P : WriteFootprint}
    (region : RegionName) (offFn : α → Nat) (vFn : α → ℝ)
    (active : α → Prop) [DecidablePred active] (l : List α) (s : BlockState)
    (hmem : ∀ i ∈ l, active i → P (region, offFn i)) :
    WriteWithin P s
      (l.foldl
        (fun acc i =>
          if active i then acc.writeMem region (offFn i) (vFn i) else acc) s) := by
  induction l generalizing s with
  | nil => exact writeWithin_refl P s
  | cons hd tl ih =>
      rw [List.foldl_cons]
      by_cases hact : active hd
      · simp only [hact, if_true]
        exact writeWithin_trans
          (writeMem_writeWithin region (offFn hd) (vFn hd)
            (hmem hd List.mem_cons_self hact))
          (ih _ (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai))
      · simp only [hact, if_false]
        exact ih s (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai)

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

theorem foldl_atomicAddAt_masked_writeWithin {α : Type} {P : WriteFootprint}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype) (mask : α → Bool)
    (l : List α) (s : BlockState)
    (hmem : ∀ i, i ∈ l → mask i = true → P (regionFn i, offsetFn i)) :
    WriteWithin P s
      (l.foldl
        (fun acc i =>
          if mask i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i)
              (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
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
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd))) :=
          writeMemTyped_writeWithin dtype (regionFn hd) (offsetFn hd)
            _ (hmem hd (List.mem_cons_self) hmask)
        have htail :
            WriteWithin P
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd)))
              (tl.foldl
                (fun acc i =>
                  if mask i then
                    acc.writeMemTyped dtype (regionFn i) (offsetFn i)
                      (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
                  else acc)
                (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                  (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd)))) :=
          ih _ (fun i hi hmi => hmem i (List.mem_cons_of_mem hd hi) hmi)
        exact writeWithin_trans hfirst htail
      · have hmaskFalse : mask hd = false := by
          cases h' : mask hd
          · rfl
          · exact False.elim (hmask h')
        simp only [hmaskFalse]
        exact ih s (fun i hi hmi => hmem i (List.mem_cons_of_mem hd hi) hmi)

theorem foldl_atomicAddAt_prop_writeWithin {α : Type} {P : WriteFootprint}
    (dtype : TileDType) (h : NumericDType dtype)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → TileCarrier dtype)
    (active : α → Prop) [DecidablePred active]
    (l : List α) (s : BlockState)
    (hmem : ∀ i, i ∈ l → active i → P (regionFn i, offsetFn i)) :
    WriteWithin P s
      (l.foldl
        (fun acc i =>
          if active i then
            acc.writeMemTyped dtype (regionFn i) (offsetFn i)
              (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
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
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd))) :=
          writeMemTyped_writeWithin dtype (regionFn hd) (offsetFn hd)
            _ (hmem hd (List.mem_cons_self) hactive)
        have htail :
            WriteWithin P
              (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd)))
              (tl.foldl
                (fun acc i =>
                  if active i then
                    acc.writeMemTyped dtype (regionFn i) (offsetFn i)
                      (h.add (acc.readMemValue dtype (regionFn i) (offsetFn i)) (valueFn acc i))
                  else acc)
                (s.writeMemTyped dtype (regionFn hd) (offsetFn hd)
                  (h.add (s.readMemValue dtype (regionFn hd) (offsetFn hd)) (valueFn s hd)))) :=
          ih _ (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai)
        exact writeWithin_trans hfirst htail
      · simp only [hactive, if_false]
        exact ih s (fun i hi hai => hmem i (List.mem_cons_of_mem hd hi) hai)

end BlockState

end VeriTile.Triton
