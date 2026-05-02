/-
VeriTile.Triton.Semantics

Typed operational semantics for the Triton subset.

The runtime value layer is intentionally absent: expression types carry their
dtype and shape, and evaluation returns `Tile dtype shape` directly.
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Exp
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Sigmoid
import Mathlib.Analysis.Complex.Trigonometric
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core

namespace VeriTile.Triton

namespace Tile

@[simp] theorem scalar_data {dtype : TileDType} (x : TileCarrier dtype) :
    (Tile.scalar x).data PUnit.unit = x := rfl

@[simp] theorem scalar_data_index {dtype : TileDType}
    (x : TileCarrier dtype) (i : TileIndex []) :
    (Tile.scalar x).data i = x := by
  cases i
  rfl

@[simp] theorem scalar_eta {dtype : TileDType} (x : TileCarrier dtype) :
    ({ data := fun _ : TileIndex [] => x } : Tile dtype []) =
      Tile.scalar x := rfl

@[simp] theorem vec_data {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) (i : TileIndex [n]) :
    (Tile.vec f).data i = f i.1 := rfl

@[simp] theorem vec_data_fin {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) (i : Fin n) :
    (Tile.vec f).data (i, PUnit.unit) = f i := rfl

end Tile

/-- `⊥`-propagating arithmetic on `WithBot ℝ`.

`Option.map₂ f` returns `none` if either argument is `none`, and `some (f a b)`
on two `some` values. This gives uniform ⊥-propagation for sub/mul/div without
having to chase Mathlib's `WithBot` typeclass instances (which don't always
align with our intended IEEE-style propagation — e.g. Mathlib's `WithBot.Mul`
defines `0 * ⊥ = 0`, but we want `0 * ⊥ = ⊥`). -/
@[simp] def WithBot.realAdd : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· + ·)
@[simp] def WithBot.realSub : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· - ·)
@[simp] def WithBot.realMul : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· * ·)
@[simp] noncomputable def WithBot.realDiv : WithBot ℝ → WithBot ℝ → WithBot ℝ := Option.map₂ (· / ·)

namespace NumericDType

def add : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realAdd x y
  | .nat, x, y => x + y

def sub : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realSub x y
  | .nat, x, y => x - y

def mul : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realMul x y
  | .nat, x, y => x * y

noncomputable def div : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => WithBot.realDiv x y
  | .nat, x, y => x / y

end NumericDType

namespace ComparableDType

noncomputable def lt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x < y)
  | .nat, x, y => decide (x < y)

noncomputable def le :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≤ y)
  | .nat, x, y => decide (x ≤ y)

noncomputable def eq :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x = y)
  | .nat, x, y => decide (x = y)

noncomputable def gt :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x > y)
  | .nat, x, y => decide (x > y)

noncomputable def ge :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≥ y)
  | .nat, x, y => decide (x ≥ y)

noncomputable def ne :
    ComparableDType dtype → TileCarrier dtype → TileCarrier dtype → Bool
  | .real, x, y => decide (x ≠ y)
  | .nat, x, y => decide (x ≠ y)

end ComparableDType

/-- Cast a typed tile across definitional dtype/shape equalities. -/
def castTile {dtype shape dtype' shape'}
    (hd : dtype = dtype') (hs : shape = shape')
    (v : Tile dtype' shape') : Tile dtype shape := by
  subst dtype'
  subst shape'
  exact v

@[simp] theorem castTile_self {dtype shape}
    (hd : dtype = dtype) (hs : shape = shape) (v : Tile dtype shape) :
    castTile hd hs v = v := by
  cases hd
  cases hs
  rfl

/-- A typed register file. The same register name can only be observed at the
    dtype/shape requested by the typed AST node that references it. -/
abbrev RegFile :=
  (dtype : TileDType) → (shape : TileShape) → RegName → Option (Tile dtype shape)

/--
A block-level execution state.

`regs` is typed directly; there is no dynamically tagged runtime value.
`undef` models Triton's masked load with `other=None`.
-/
structure BlockState where
  mem   : RegionName → Nat → ℝ
  regs  : RegFile
  /-- Per-axis program IDs. `pids axis` is the value of
  `tl.program_id(axis)`. Total over `Nat`; out-of-range axes default to `0`
  (or whatever the launch model picks). FA-1's batch/head launch grid uses
  axes 0/1/2 for `(q_block, head, batch)`. -/
  pids  : Nat → Nat
  undef : RegionName → Nat → ℝ := fun _ _ => 0

instance : Inhabited BlockState :=
  ⟨{ mem := fun _ _ => 0
   , regs := fun _ _ _ => none
   , pids := fun _ => 0
   , undef := fun _ _ => 0 }⟩

namespace BlockState

/-- Backward-compatible single-axis program id: equals `pids 0`. -/
@[reducible] def pid (s : BlockState) : Nat := s.pids 0

@[simp] theorem pid_eq (s : BlockState) : s.pid = s.pids 0 := rfl

@[ext] theorem ext {s t : BlockState}
    (hmem : ∀ region offset, s.mem region offset = t.mem region offset)
    (hregs : ∀ dtype shape name, s.regs dtype shape name = t.regs dtype shape name)
    (hpids : ∀ axis, s.pids axis = t.pids axis)
    (hundef : ∀ region offset, s.undef region offset = t.undef region offset) :
    s = t := by
  cases s
  cases t
  simp only at hmem hregs hpids hundef
  congr
  · exact funext fun region => funext fun offset => hmem region offset
  · exact funext fun dtype => funext fun shape => funext fun name => hregs dtype shape name
  · exact funext hpids
  · exact funext fun region => funext fun offset => hundef region offset

/-- Update a single typed register. -/
def setReg (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) : BlockState :=
  { s with regs := fun dtype' shape' name' =>
      if hd : dtype' = dtype then
        if hs : shape' = shape then
          if name' = name then some (castTile hd hs v) else s.regs dtype' shape' name'
        else s.regs dtype' shape' name'
      else s.regs dtype' shape' name' }

@[simp] theorem setReg_same (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    (s.setReg name dtype shape v).regs dtype shape name = some v := by
  simp [setReg]

@[simp] theorem setReg_ne_name (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (h : name' ≠ name) :
    (s.setReg name dtype shape v).regs dtype' shape' name' =
      s.regs dtype' shape' name' := by
  simp [setReg, h]

@[simp] theorem setReg_ne_dtype (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (h : dtype' ≠ dtype) :
    (s.setReg name dtype shape v).regs dtype' shape' name' =
      s.regs dtype' shape' name' := by
  simp [setReg, h]

@[simp] theorem setReg_ne_shape (s : BlockState) (name name' : RegName)
    (dtype dtype' : TileDType) (shape shape' : TileShape) (v : Tile dtype shape)
    (h : shape' ≠ shape) :
    (s.setReg name dtype shape v).regs dtype' shape' name' =
      s.regs dtype' shape' name' := by
  simp [setReg, h]

@[simp] theorem setReg_mem (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape)
    (region : RegionName) (offset : Nat) :
    (s.setReg name dtype shape v).mem region offset = s.mem region offset := rfl

@[simp] theorem setReg_pids (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    (s.setReg name dtype shape v).pids = s.pids := rfl

@[simp] theorem setReg_pid (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape) :
    (s.setReg name dtype shape v).pid = s.pid := rfl

@[simp] theorem setReg_undef (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape)
    (region : RegionName) (offset : Nat) :
    (s.setReg name dtype shape v).undef region offset = s.undef region offset := rfl

/-- Write a single scalar to memory. -/
def writeMem (s : BlockState) (region : RegionName) (offset : Nat) (v : ℝ) : BlockState :=
  { s with mem := fun r o =>
      if r = region ∧ o = offset then v else s.mem r o }

@[simp] theorem writeMem_regs (s : BlockState) (region : RegionName)
    (offset : Nat) (v : ℝ) (dtype : TileDType) (shape : TileShape)
    (name : RegName) :
    (s.writeMem region offset v).regs dtype shape name = s.regs dtype shape name := rfl

@[simp] theorem writeMem_pids (s : BlockState) (region : RegionName)
    (offset : Nat) (v : ℝ) :
    (s.writeMem region offset v).pids = s.pids := rfl

@[simp] theorem writeMem_pid (s : BlockState) (region : RegionName)
    (offset : Nat) (v : ℝ) :
    (s.writeMem region offset v).pid = s.pid := rfl

@[simp] theorem writeMem_undef (s : BlockState) (region : RegionName)
    (offset : Nat) (v : ℝ) (r : RegionName) (o : Nat) :
    (s.writeMem region offset v).undef r o = s.undef r o := rfl

/-- Read a single scalar from memory. -/
@[inline] def readMem (s : BlockState) (region : RegionName) (offset : Nat) : ℝ :=
  s.mem region offset

private theorem foldl_writeMem_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, offsetFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region o)
      = s.mem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
    have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
      fun k hk => h k (List.mem_cons_of_mem hd hk)
    rw [List.foldl_cons, ih _ htl]
    show (if region = region ∧ o = offsetFn hd then valueFn hd else s.mem region o)
        = s.mem region o
    rw [if_neg]
    rintro ⟨_, h_eq⟩
    exact hhd h_eq.symm

theorem scatter_readback_list {α : Type} {region : RegionName}
    (l : List α) (s : BlockState) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (i : α) (h_nodup : l.Nodup) (h_mem : i ∈ l)
    (h_inj : Function.Injective offsetFn) :
    (l.foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region (offsetFn i)
    = valueFn i := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem h_mem
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  rw [hl, List.foldl_append, List.foldl_cons]
  have h_not_in : ∀ k ∈ l₂, offsetFn k ≠ offsetFn i := fun k hk heq => by
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMem_preserves offsetFn valueFn (offsetFn i) l₂ _ h_not_in]
  show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _)
      = valueFn i
  exact if_pos ⟨rfl, rfl⟩

theorem scatter_readback {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region (offsetFn i)
      = valueFn i := by
  exact scatter_readback_list (List.finRange n) s offsetFn valueFn
    i (List.nodup_finRange n) (List.mem_finRange i) h_inj

theorem scatter_readback_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region (offsetFn i)
    = valueFn i :=
  scatter_readback_list (TileShape.allIndices shape) s offsetFn valueFn
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) h_inj

private theorem foldl_writeMem_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, mask k = true → offsetFn k ≠ o) →
      ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region o)
      = s.mem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    rw [List.foldl_cons]
    have htl : ∀ k ∈ tl, mask k = true → offsetFn k ≠ o :=
      fun k hk hmk => h k (List.mem_cons_of_mem hd hk) hmk
    by_cases hmaskhd : mask hd = true
    · have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self) hmaskhd
      simp only [hmaskhd, if_true]
      rw [ih _ htl]
      show (if region = region ∧ o = offsetFn hd then valueFn hd else s.mem region o)
          = s.mem region o
      rw [if_neg]
      rintro ⟨_, h_eq⟩
      exact hhd h_eq.symm
    · have hmaskhd' : mask hd = false := by
        rcases hmaskFalse : mask hd
        · rfl
        · exact absurd hmaskFalse hmaskhd
      simp only [hmaskhd', if_false, Bool.false_eq_true]
      exact ih _ htl

theorem scatter_readback_masked_list {α : Type} {region : RegionName}
    (l : List α) (s : BlockState)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool)
    (i : α) (h_nodup : l.Nodup) (h_mem : i ∈ l)
    (h_inj : Function.Injective offsetFn) :
    (l.foldl
       (fun acc k =>
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if mask i then valueFn i else s.mem region (offsetFn i) := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem h_mem
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, hl1_disj⟩ := h_nodup
  rw [hl, List.foldl_append, List.foldl_cons]
  have h_l1_not_in : ∀ k ∈ l₁, mask k = true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    rw [hki] at hk
    exact (hl1_disj i hk i (List.mem_cons_self)) rfl
  have h_l2_not_in : ∀ k ∈ l₂, mask k = true → offsetFn k ≠ offsetFn i := by
    intro k hk _hmk heq
    have hki : k = i := h_inj heq
    subst hki
    exact hi_notin_l2 hk
  rw [foldl_writeMem_masked_preserves offsetFn valueFn mask (offsetFn i) l₂ _ h_l2_not_in]
  by_cases hmi : mask i = true
  · simp only [hmi, if_true]
    show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _)
        = valueFn i
    exact if_pos ⟨rfl, rfl⟩
  · have hmi' : mask i = false := by
      rcases hmiFalse : mask i
      · rfl
      · exact absurd hmiFalse hmi
    simp only [hmi', if_false, Bool.false_eq_true]
    rw [foldl_writeMem_masked_preserves offsetFn valueFn mask (offsetFn i) l₁ _ h_l1_not_in]

theorem scatter_readback_masked {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (mask : Fin n → Bool)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k =>
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if mask i then valueFn i else s.mem region (offsetFn i) :=
  scatter_readback_masked_list (List.finRange n) s offsetFn valueFn mask
    i (List.nodup_finRange n) (List.mem_finRange i) h_inj

theorem scatter_readback_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (mask : TileIndex shape → Bool)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if mask i then valueFn i else s.mem region (offsetFn i) :=
  scatter_readback_masked_list (TileShape.allIndices shape) s offsetFn valueFn mask
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) h_inj

theorem scatter_readback_prop_masked {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (P : Fin n → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if P i then valueFn i else s.mem region (offsetFn i) := by
  have h := scatter_readback_masked (region := region) s offsetFn valueFn
              (fun k => decide (P k)) h_inj i
  simp only [decide_eq_true_eq] at h
  exact h

theorem scatter_readback_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if P i then valueFn i else s.mem region (offsetFn i) := by
  have h := scatter_readback_masked_nd (region := region) s offsetFn valueFn
              (fun k => decide (P k)) h_inj i
  simp only [decide_eq_true_eq] at h
  exact h

theorem scatter_readback_prop_masked_list_of_true {α : Type} {region : RegionName}
    (l : List α) (s : BlockState)
    (offsetFn : α → Nat) (valueFn : α → ℝ) (P : α → Prop) [DecidablePred P]
    (i : α) (h_nodup : l.Nodup) (h_mem : i ∈ l) (hPi : P i)
    (h_no_collision :
      ∀ k, k ∈ l → P k → offsetFn k = offsetFn i → k = i) :
    (l.foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = valueFn i := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem h_mem
  rw [hl] at h_nodup
  rw [List.nodup_append, List.nodup_cons] at h_nodup
  obtain ⟨_, ⟨hi_notin_l2, _⟩, _⟩ := h_nodup
  rw [hl, List.foldl_append, List.foldl_cons]
  have h_l2_not_in : ∀ k ∈ l₂, decide (P k) = true → offsetFn k ≠ offsetFn i := by
    intro k hk hPkDec heq
    have hPk : P k := by simpa only [decide_eq_true_eq] using hPkDec
    have hki : k = i :=
      h_no_collision k (by
        rw [hl]
        exact List.mem_append_right l₁ (List.mem_cons_of_mem i hk)) hPk heq
    subst hki
    exact hi_notin_l2 hk
  simp only [hPi, if_true]
  let st : BlockState :=
    (List.foldl
      (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      s l₁).writeMem region (offsetFn i) (valueFn i)
  have hpres :=
    foldl_writeMem_masked_preserves (region := region) offsetFn valueFn (fun k => decide (P k))
      (offsetFn i) l₂ st h_l2_not_in
  have hstep :
      (fun (acc : BlockState) k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
        =
      (fun (acc : BlockState) k =>
        if decide (P k) then acc.writeMem region (offsetFn k) (valueFn k) else acc) := by
    funext acc k
    by_cases hk : P k <;> simp [hk]
  change (List.foldl
      (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st l₂).mem region (offsetFn i) = valueFn i
  rw [show List.foldl
      (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st l₂ =
    List.foldl
      (fun acc k => if decide (P k) then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st l₂ by
        rw [hstep]]
  rw [hpres]
  simp [BlockState.writeMem, st]

theorem scatter_readback_prop_masked_nd_of_true {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (P : TileIndex shape → Prop) [DecidablePred P]
    (i : TileIndex shape) (hPi : P i)
    (h_no_collision :
      ∀ k, P k → offsetFn k = offsetFn i → k = i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = valueFn i :=
  scatter_readback_prop_masked_list_of_true (TileShape.allIndices shape) s offsetFn valueFn P
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) hPi
    (fun k _hk hPk heq => h_no_collision k hPk heq)

theorem scatter_preserves_other_region {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (R : RegionName) (h_ne : R ≠ region) (off : Nat) :
    ∀ (l : List α) (s : BlockState),
      ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        R off)
      = s.mem R off := by
  intro l
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s
    rw [List.foldl_cons, ih]
    show (s.writeMem region (offsetFn hd) (valueFn hd)).mem R off = s.mem R off
    unfold writeMem
    show (if R = region ∧ off = offsetFn hd then valueFn hd else s.mem R off)
        = s.mem R off
    rw [if_neg]
    rintro ⟨h_R, _⟩
    exact h_ne h_R

theorem foldl_writeMem_pid {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMem_masked_pid {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemAt_pid {α : Type}
    (regionFn : α → RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMem (regionFn k) (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMemAt_masked_pid {α : Type}
    (regionFn : α → RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMem (regionFn k) (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

end BlockState

def Tile.bop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → TileCarrier dtype)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile dtype out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

def Tile.cop {dtype a b out}
    (op : TileCarrier dtype → TileCarrier dtype → Bool)
    (bc : Broadcast a b out) (x : Tile dtype a) (y : Tile dtype b) :
    Tile .bool out :=
  ⟨fun i => op (x.data (bc.leftIndex i)) (y.data (bc.rightIndex i))⟩

def Tile.ptrAdd {a b out}
    (bc : Broadcast a b out) (ptrs : Tile .ptr a) (offs : Tile .nat b) :
    Tile .ptr out :=
  ⟨fun i =>
    let p := ptrs.data (bc.leftIndex i)
    let o := offs.data (bc.rightIndex i)
    (p.1, p.2 + o)⟩

def Tile.uop {shape} (op : WithBot ℝ → WithBot ℝ) (x : Tile .real shape) : Tile .real shape :=
  ⟨fun i => op (x.data i)⟩

/-! ### Lifted unary functions on `WithBot ℝ`

`exp(-∞) = 0` and `sigmoid(-∞) = 0` are the IEEE-faithful values; this is what
makes the OnlineSoftmax `m`-seed flow correctly: with `m_0 = ⊥`, the first
iteration's `exp(m_0 - m_1) * l_0 = exp(⊥) * 0 = 0 * 0 = 0`, recovering the
correct base-case `l_1 = exp(x_0 - x_0) = 1`. -/

noncomputable def WithBot.realExp : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- exp(-∞) = 0
  | some r => some (Real.exp r)

noncomputable def WithBot.realLog : WithBot ℝ → WithBot ℝ
  | none   => none
  | some r => some (Real.log r)

noncomputable def WithBot.realSigmoid : WithBot ℝ → WithBot ℝ
  | none   => some 0           -- sigmoid(-∞) = 0
  | some r => some (Real.sigmoid r)

noncomputable def WithBot.realSqrt : WithBot ℝ → WithBot ℝ
  | none   => none             -- sqrt(-∞) undefined
  | some r => some (Real.sqrt r)

noncomputable def WithBot.realTanh : WithBot ℝ → WithBot ℝ
  | none   => some (-1)        -- tanh(-∞) = -1
  | some r => some (Real.tanh r)

@[simp] theorem WithBot.realExp_some (r : ℝ) :
    WithBot.realExp (some r) = some (Real.exp r) := rfl
@[simp] theorem WithBot.realLog_some (r : ℝ) :
    WithBot.realLog (some r) = some (Real.log r) := rfl
@[simp] theorem WithBot.realSigmoid_some (r : ℝ) :
    WithBot.realSigmoid (some r) = some (Real.sigmoid r) := rfl
@[simp] theorem WithBot.realSqrt_some (r : ℝ) :
    WithBot.realSqrt (some r) = some (Real.sqrt r) := rfl
@[simp] theorem WithBot.realTanh_some (r : ℝ) :
    WithBot.realTanh (some r) = some (Real.tanh r) := rfl

@[simp] theorem WithBot.realExp_bot :
    WithBot.realExp (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_bot :
    WithBot.realSigmoid (⊥ : WithBot ℝ) = ((0 : ℝ) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_bot :
    WithBot.realTanh (⊥ : WithBot ℝ) = ((-1 : ℝ) : WithBot ℝ) := rfl

@[simp] theorem WithBot.realExp_coe (r : ℝ) :
    WithBot.realExp ((r : ℝ) : WithBot ℝ) = (((Real.exp r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realLog_coe (r : ℝ) :
    WithBot.realLog ((r : ℝ) : WithBot ℝ) = (((Real.log r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSigmoid_coe (r : ℝ) :
    WithBot.realSigmoid ((r : ℝ) : WithBot ℝ) = (((Real.sigmoid r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSqrt_coe (r : ℝ) :
    WithBot.realSqrt ((r : ℝ) : WithBot ℝ) = (((Real.sqrt r : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realTanh_coe (r : ℝ) :
    WithBot.realTanh ((r : ℝ) : WithBot ℝ) = (((Real.tanh r : ℝ)) : WithBot ℝ) := rfl

/-! ### Algebraic simp lemmas on `WithBot ℝ` arithmetic helpers -/

@[simp] theorem WithBot.realAdd_coe_coe (a b : ℝ) :
    WithBot.realAdd ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a + b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realSub_coe_coe (a b : ℝ) :
    WithBot.realSub ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a - b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realMul_coe_coe (a b : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a * b : ℝ)) : WithBot ℝ) := rfl
@[simp] theorem WithBot.realDiv_coe_coe (a b : ℝ) :
    WithBot.realDiv ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (((a / b : ℝ)) : WithBot ℝ) := rfl

/-- `realSub ⊥ x = ⊥` (and symmetrically). The corresponding `realAdd` and
`realMul` propagate `⊥` for the same reason — but only `Sub` is needed for the
OnlineSoftmax L proof at iter 0 (`exp(M_0 - M_1) = exp(⊥ - ↑x₀) = exp(⊥) = 0`). -/
@[simp] theorem WithBot.realSub_bot_left (x : WithBot ℝ) :
    WithBot.realSub (⊥ : WithBot ℝ) x = (⊥ : WithBot ℝ) := by
  cases x <;> rfl
@[simp] theorem WithBot.realSub_bot_right (x : WithBot ℝ) :
    WithBot.realSub x (⊥ : WithBot ℝ) = (⊥ : WithBot ℝ) := by
  cases x <;> rfl

@[simp] theorem WithBot.realMul_coe_zero (a : ℝ) :
    WithBot.realMul ((a : ℝ) : WithBot ℝ) (((0 : ℝ)) : WithBot ℝ)
      = (((0 : ℝ)) : WithBot ℝ) := by
  show ((a * 0 : ℝ) : WithBot ℝ) = _
  simp

/-- `↑a + ↑b = ↑(a + b)` for `WithBot ℝ`. Reverse of Mathlib's `WithBot.coe_add`.
NOT marked `@[simp]` to avoid loop with the forward direction; use as `←` rewrite
where consolidating coe outward is desired. -/
theorem WithBot.coe_add_coe_eq (a b : ℝ) :
    ((a : ℝ) : WithBot ℝ) + ((b : ℝ) : WithBot ℝ) = (((a + b : ℝ)) : WithBot ℝ) :=
  (WithBot.coe_add a b).symm

/-- `Option.map f ↑x = ↑(f x)` for `WithBot ℝ`. -/
@[simp] theorem WithBot.coe_map_coe_eq {α} (f : ℝ → α) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = some (f a) := rfl

/-- Reduce-sum with `keep_dims = false` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.eraseAxis shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum with `keep_dims = true` along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSumKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile .real (TileShape.setAxisOne shape axis) :=
  ⟨fun outIdx =>
    @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
      (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩

/-- Reduce-sum along an arbitrary Triton axis. -/
noncomputable def Tile.reduceSum {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Tile .real (TileShape.reduceShape shape axis keepDims) := by
  cases keepDims
  · exact Tile.reduceSumDrop axis x
  · exact Tile.reduceSumKeep axis x

/-! ### `simp` lemmas for the structural cases of reduceSum -/

@[simp] theorem Tile.reduceSum_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis false x = Tile.reduceSumDrop axis x := rfl

@[simp] theorem Tile.reduceSum_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceSum axis true x = Tile.reduceSumKeep axis x := rfl

/-- Body of `reduceSumDrop` at an output index. Lets `simp` push past the
opaque `Tile` constructor so kernel proofs can reach the `Finset.sum` form. -/
@[simp] theorem Tile.reduceSumDrop_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.eraseAxis shape axis)) :
    (Tile.reduceSumDrop axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k)) := rfl

@[simp] theorem Tile.reduceSumKeep_data {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape)
    (outIdx : TileIndex (TileShape.setAxisOne shape axis)) :
    (Tile.reduceSumKeep axis x).data outIdx =
      @Finset.sum (Fin (TileShape.axisDim shape axis)) (WithBot ℝ) _ Finset.univ
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k)) := rfl

/-! ### `simp` helpers for the `WithBot ℝ` boundary

A typical kernel proof reduces `tl.sum(x)` to `∑ i, (some (xs i) : WithBot ℝ)`,
then `tl.store` demotes via `Option.getD … 0`. Mathlib's `WithBot.coe_sum`
gives us `↑(∑ f i) = ∑ ↑(f i)` (for `WithBot.coe = some`); read backwards, it
collapses the sum-of-`some`s to a `some` of the sum, which then evaluates the
`getD`. -/

/-- `WithBot.unbotD` on a `some _` projects out the value. Same as
`WithBot.unbotD_coe` but stated in the `some`-form `evalOp` produces. -/
@[simp] theorem WithBot.unbotD_some {α} (d a : α) :
    WithBot.unbotD d (some a : WithBot α) = a := rfl

/-- `Option.map₂` on two `some`-lifted values produces a `some`-lifted result.
Required because `Tile.bop` for `.real` arithmetic uses `Option.map₂`. -/
@[simp] theorem Option.map₂_some_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map_some_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f (some a : WithBot ℝ) = (some (f a) : WithBot ℝ) := rfl

/-- `↑`-form: when proofs land on the coe view of `WithBot ℝ`, the same
reductions apply. Simp matches syntactically, so we need both forms. -/
@[simp] theorem Option.map₂_coe_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = ((f a b : ℝ) : WithBot ℝ) := rfl

@[simp] theorem Option.map_coe_real (f : ℝ → ℝ) (a : ℝ) :
    Option.map f ((a : ℝ) : WithBot ℝ) = ((f a : ℝ) : WithBot ℝ) := rfl

/-- Mixed-form: `some` on left, `↑` on right (and vice versa). Both forms appear
in goals after partial simp normalization. -/
@[simp] theorem Option.map₂_some_coe (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f (some a : WithBot ℝ) ((b : ℝ) : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl

@[simp] theorem Option.map₂_coe_some (f : ℝ → ℝ → ℝ) (a b : ℝ) :
    Option.map₂ f ((a : ℝ) : WithBot ℝ) (some b : WithBot ℝ)
      = (some (f a b) : WithBot ℝ) := rfl


@[simp] theorem WithBot.sum_some_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = (((∑ i ∈ s, f i) : ℝ) : WithBot ℝ) :=
  (WithBot.coe_sum s f).symm

/-- `some`-form companion to `WithBot.sum_some_eq_some`. The explicit
`AddCommMonoid (WithBot ℝ)` ascription via `@Finset.sum` is what makes the
typeclass resolution fire — bare `some _ : WithBot ℝ` ascription elaborates
to `Option ℝ` for typeclass purposes. -/
@[simp] theorem WithBot.sum_someTerm_eq_some {ι} (s : Finset ι) (f : ι → ℝ) :
    @Finset.sum ι (WithBot ℝ) _ s (fun i => (some (f i) : WithBot ℝ))
      = (some (∑ i ∈ s, f i) : WithBot ℝ) := by
  show @Finset.sum ι (WithBot ℝ) _ s (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sum_some_eq_some]
  rfl


/-- After a `tl.sum` reduce on a `some`-lifted tile, demoting via `unbotD 0`
(used by `tl.store`) recovers the underlying ℝ-valued sum.

Stated in the `some`-form (rather than `↑`) because that's what `evalOp`
produces — `tl.load` lifts via `some (s.readMem ...)` and `tl.sum` on a tile
of `some`-values gives `∑ some (xs i)` literally. -/
@[simp] theorem WithBot.unbotD_sum_some {ι} (s : Finset ι) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (∑ i ∈ s, (some (f i) : WithBot ℝ)) = ∑ i ∈ s, f i := by
  show WithBot.unbotD (0 : ℝ) (∑ i ∈ s, ((f i : ℝ) : WithBot ℝ)) = ∑ i ∈ s, f i
  rw [WithBot.sum_some_eq_some]
  rfl

/-- `sup'` over `↑`-lifted values equals `↑` of the `sup'`. -/
theorem WithBot.sup'_coe {ι} [LinearOrder ι] (s : Finset ι) (h : s.Nonempty)
    (f : ι → ℝ) :
    s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ)) = ((s.sup' h f : ℝ) : WithBot ℝ) := by
  refine h.cons_induction ?_ ?_
  · intro a; rfl
  · intro a s' ha hne ih
    rw [Finset.sup'_cons hne, Finset.sup'_cons hne, ih]
    rfl

/-- `some`-form of `WithBot.sup'_coe`. Required because `evalOp` produces
literal `some _` calls, not `↑`. The `@Finset.sup' (WithBot ℝ) ι` ascription
forces typeclass resolution to use `SemilatticeSup (WithBot ℝ)` rather than
the elaborated `Option ℝ`. -/
@[simp] theorem WithBot.sup'_someTerm_eq_some {ι} [LinearOrder ι]
    (s : Finset ι) (h : s.Nonempty) (f : ι → ℝ) :
    @Finset.sup' (WithBot ℝ) ι _ s h (fun i => (some (f i) : WithBot ℝ))
      = (some (s.sup' h f) : WithBot ℝ) := by
  show @Finset.sup' (WithBot ℝ) ι _ s h (fun i => ((f i : ℝ) : WithBot ℝ)) = _
  rw [WithBot.sup'_coe]; rfl

/-- Companion to `WithBot.sup'_coe`: store-side demote on a max reduce. -/
@[simp] theorem WithBot.unbotD_sup'_some {ι} [LinearOrder ι] (s : Finset ι)
    (h : s.Nonempty) (f : ι → ℝ) :
    WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => (some (f i) : WithBot ℝ)))
      = s.sup' h f := by
  show WithBot.unbotD (0 : ℝ) (s.sup' h (fun i => ((f i : ℝ) : WithBot ℝ))) = _
  rw [WithBot.sup'_coe]; rfl

/-! ### Reduce-max along an arbitrary axis

Returns `none` only when the reduced dimension is `0` (empty `sup'`
undefined). -/

noncomputable def Tile.reduceMaxDrop {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.eraseAxis shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.insertAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMaxKeep {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Option (Tile .real (TileShape.setAxisOne shape axis)) :=
  if h : 0 < TileShape.axisDim shape axis then
    some ⟨fun outIdx =>
      (Finset.univ : Finset (Fin (TileShape.axisDim shape axis))).sup'
        (by exact ⟨⟨0, h⟩, Finset.mem_univ _⟩)
        (fun k => x.data (TileShape.replaceAxisIndex shape axis outIdx k))⟩
  else
    none

noncomputable def Tile.reduceMax {shape : TileShape}
    (axis : Fin shape.length) (keepDims : Bool) (x : Tile .real shape) :
    Option (Tile .real (TileShape.reduceShape shape axis keepDims)) := by
  cases keepDims
  · exact Tile.reduceMaxDrop axis x
  · exact Tile.reduceMaxKeep axis x

/-! ### `simp` lemmas for the structural cases of reduceMax -/

@[simp] theorem Tile.reduceMax_false {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis false x = Tile.reduceMaxDrop axis x := rfl

@[simp] theorem Tile.reduceMax_true {shape : TileShape}
    (axis : Fin shape.length) (x : Tile .real shape) :
    Tile.reduceMax axis true x = Tile.reduceMaxKeep axis x := rfl

def Tile.natToReal {shape} (x : Tile .nat shape) : Tile .real shape :=
  ⟨fun i => some ((x.data i : ℝ))⟩

/-- Block-level (possibly batched) matrix multiply (Triton's `tl.dot`):
`c[…, m, n] = ∑_k a[…, m, k] * b[…, k, n]`.

Operates on the trailing two dims; any batch prefix `batch : TileShape` is
broadcast pointwise. Structural recursion on the batch fixes one outer
index per level and recurses on the remaining batch.

`⊥`-propagating: if any factor in any pair is `⊥` the corresponding
summand is `⊥`, and `Finset.sum` over `WithBot ℝ` propagates `⊥` through
the whole entry. Well-formed kernels load real (non-`⊥`) values into the
operands, so the dot product collapses to a regular `↑(∑ a*b)` term and
the `tl.store` demotion via `unbotD 0` recovers the underlying ℝ value. -/
noncomputable def Tile.dot : (batch : TileShape) → {M K N : Nat} →
    Tile .real (batch ++ [M, K]) → Tile .real (batch ++ [K, N]) →
    Tile .real (batch ++ [M, N])
  | [], _, K, _, a, b =>
      ⟨fun (m, n, _) =>
        @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
          (fun k => Option.map₂ (· * ·)
                      (a.data (m, k, PUnit.unit))
                      (b.data (k, n, PUnit.unit)))⟩
  | _ :: rest, M, K, N, a, b =>
      ⟨fun (i, restIdx) =>
        (Tile.dot rest (M := M) (K := K) (N := N)
            ⟨fun rIdx => a.data (i, rIdx)⟩
            ⟨fun rIdx => b.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.dot` at the rank-2 base case (`batch = []`); lets `simp`
expose the `Finset.sum` form for kernel proofs. -/
@[simp] theorem Tile.dot_nil_data {M K N : Nat}
    (a : Tile .real [M, K]) (b : Tile .real [K, N])
    (m : Fin M) (n : Fin N) (rest : TileIndex []) :
    (Tile.dot [] a b).data (m, n, rest) =
      @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun k => Option.map₂ (· * ·)
                    (a.data (m, k, PUnit.unit))
                    (b.data (k, n, PUnit.unit))) := rfl

/-- Recursive step: `(Tile.dot (b :: rest) a b').data (i, restIdx)` slices
each operand at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.dot_cons_data {b : Nat} {rest : TileShape} {M K N : Nat}
    (a : Tile .real ((b :: rest) ++ [M, K]))
    (b' : Tile .real ((b :: rest) ++ [K, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [M, N])) :
    (Tile.dot (b :: rest) a b').data (i, restIdx) =
      (Tile.dot rest (M := M) (K := K) (N := N)
          ⟨fun rIdx => a.data (i, rIdx)⟩
          ⟨fun rIdx => b'.data (i, rIdx)⟩).data restIdx := rfl

/-- Trailing-two-axes transpose (`Tile.dot`-style framework: matrix dims
are the trailing two, leading dims are an unchanged `batch` prefix). -/
def Tile.transpose : {dtype : TileDType} → (batch : TileShape) →
    {M N : Nat} →
    Tile dtype (batch ++ [M, N]) → Tile dtype (batch ++ [N, M])
  | _, [],         _, _, x =>
      ⟨fun (n, m, _) => x.data (m, n, PUnit.unit)⟩
  | _, _ :: rest,  _, _, x =>
      ⟨fun (i, restIdx) =>
        (Tile.transpose rest
            ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx⟩

/-- Body of `Tile.transpose` at the rank-2 base case: swap `(m, n) ↔ (n, m)`. -/
@[simp] theorem Tile.transpose_nil_data {dtype : TileDType} {M N : Nat}
    (x : Tile dtype [M, N]) (n : Fin N) (m : Fin M) (rest : TileIndex []) :
    (Tile.transpose [] x).data (n, m, rest) =
      x.data (m, n, PUnit.unit) := rfl

/-- Recursive step: `(Tile.transpose (b :: rest) x).data (i, restIdx)`
slices at outer index `i` and recurses on the remaining batch. -/
@[simp] theorem Tile.transpose_cons_data {dtype : TileDType}
    {b : Nat} {rest : TileShape} {M N : Nat}
    (x : Tile dtype ((b :: rest) ++ [M, N]))
    (i : Fin b) (restIdx : TileIndex (rest ++ [N, M])) :
    (Tile.transpose (b :: rest) x).data (i, restIdx) =
      (Tile.transpose rest (M := M) (N := N)
          ⟨fun rIdx => x.data (i, rIdx)⟩).data restIdx := rfl

/-- Insert a unit-size axis at position `axis`. The output index is
projected back to the input by `dropInsertedIndex`, which throws away
the inserted slot's `Fin 1` coordinate. -/
def Tile.expandDim {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape) :
    Tile dtype (TileShape.insertAxis shape axis 1) :=
  ⟨fun idx => x.data (TileShape.dropInsertedIndex shape axis 1 idx)⟩

@[simp] theorem Tile.expandDim_data {dtype : TileDType} {shape : TileShape}
    (axis : Fin (shape.length + 1)) (x : Tile dtype shape)
    (idx : TileIndex (TileShape.insertAxis shape axis 1)) :
    (Tile.expandDim axis x).data idx =
      x.data (TileShape.dropInsertedIndex shape axis 1 idx) := rfl

/-- Lift a plain ℝ-valued tile-shaped function into a `Tile .real`. Useful
at the spec / boundary layer: `BlockState.mem` reads ℝ, never `⊥`, so a
`tl.load`-fed kernel input is naturally a `TileIndex shape → ℝ`. The
`⊥` sentinel of `WithBot ℝ` is reserved for `-inf` / masked-off /
`tl.full(_, -inf)` values introduced *inside* a kernel; spec-level
inputs and outputs should round-trip through `ofReal` / `unbotD 0`. -/
def Tile.ofReal {shape : TileShape} (x : TileIndex shape → ℝ) :
    Tile .real shape :=
  ⟨fun i => some (x i)⟩

@[simp] theorem Tile.ofReal_data {shape : TileShape}
    (x : TileIndex shape → ℝ) (i : TileIndex shape) :
    (Tile.ofReal x).data i = some (x i) := rfl

/-- Element-wise select (`tl.where(cond, a, b)`): per-cell, pick from `a`
when `cond` is `true`, else from `b`. Same-shape; broadcast lifting is
done at the DSL layer. Named `select` to avoid the Lean `where` keyword
in definition position; the AST constructor `Op.where` and DSL surface
`tl.where(...)` keep the user-facing Triton spelling. -/
def Tile.select {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) :
    Tile dtype shape :=
  ⟨fun idx => if c.data idx then a.data idx else b.data idx⟩

@[simp] theorem Tile.select_data {dtype : TileDType} {shape : TileShape}
    (c : Tile .bool shape) (a b : Tile dtype shape) (idx : TileIndex shape) :
    (Tile.select c a b).data idx = if c.data idx then a.data idx else b.data idx := rfl

noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar (some c : WithBot ℝ))
  | .constNat n, _ => some (Tile.scalar n)
  | .constBool b, _ => some (Tile.scalar b)
  | .negInf, _ => some (Tile.scalar (none : WithBot ℝ))
  | .programId axis, s => some (Tile.scalar (s.pids axis))
  | .ref dtype shape name, s => s.regs dtype shape name
  | .arange n, _ => some (Tile.vec (fun i => i.val))
  | .broadcast e shape, s => do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .full shape e, s => do
      let v ← evalOp e s
      some ⟨fun _ => v.data PUnit.unit⟩
  | .add h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.add bc va vb)
  | .sub h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.sub bc va vb)
  | .mul h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.mul bc va vb)
  | .div h bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop h.div bc va vb)
  | .exp a, s => return Tile.uop WithBot.realExp (← evalOp a s)
  | .log a, s => return Tile.uop WithBot.realLog (← evalOp a s)
  | .sigmoid a, s => return Tile.uop WithBot.realSigmoid (← evalOp a s)
  | .sqrt a, s => return Tile.uop WithBot.realSqrt (← evalOp a s)
  | .tanh a, s => return Tile.uop WithBot.realTanh (← evalOp a s)
  | .lt h bc a b, s => return Tile.cop h.lt bc (← evalOp a s) (← evalOp b s)
  | .le h bc a b, s => return Tile.cop h.le bc (← evalOp a s) (← evalOp b s)
  | .eq h bc a b, s => return Tile.cop h.eq bc (← evalOp a s) (← evalOp b s)
  | .gt h bc a b, s => return Tile.cop h.gt bc (← evalOp a s) (← evalOp b s)
  | .ge h bc a b, s => return Tile.cop h.ge bc (← evalOp a s) (← evalOp b s)
  | .ne h bc a b, s => return Tile.cop h.ne bc (← evalOp a s) (← evalOp b s)
  | .boolAnd bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop (fun x y : Bool => x && y) bc va vb)
  | .max2 bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop max bc va vb)
  | .reduceMax axis keepDims a, s => do
      let va ← evalOp a s
      Tile.reduceMax axis keepDims va
  | .reduceSum axis keepDims a, s => return Tile.reduceSum axis keepDims (← evalOp a s)
  | .dot (batch := batch) a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.dot batch va vb)
  | .expandDim axis a, s => return Tile.expandDim axis (← evalOp a s)
  | .where c a b, s => do
      let vc ← evalOp c s
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.select vc va vb)
  | .transpose (batch := batch) a, s => do
      let va ← evalOp a s
      some (Tile.transpose batch va)
  | .ptrBase region, _ => some (Tile.scalar (region, 0))
  | .ptrAdd bc ptr off, s => do
      let ptrs ← evalOp ptr s
      let offs ← evalOp off s
      some (Tile.ptrAdd bc ptrs offs)
  | .load region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => some (s.readMem region (offsets.data i))⟩
  | .loadMask region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else some (s.undef region addr)⟩
  | .loadMaskOther region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then some (s.readMem region addr) else others.data i⟩
  | .loadPtr ptr, s => do
      let ptrs ← evalOp ptr s
      some ⟨fun i =>
        let p := ptrs.data i
        some (s.readMem p.1 p.2)⟩
  | .loadPtrMask ptr mask, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else some (s.undef p.1 p.2)⟩
  | .loadPtrMaskOther ptr mask other, s => do
      let ptrs ← evalOp ptr s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let p := ptrs.data i
        if masks.data i then some (s.readMem p.1 p.2) else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)

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
    simp [BlockState.foldl_writeMem_masked_pid]
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
