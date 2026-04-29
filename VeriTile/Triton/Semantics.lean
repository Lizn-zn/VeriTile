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
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import VeriTile.Triton.Core

namespace VeriTile.Triton

namespace Tile

@[simp] theorem scalar_data {dtype : TileDType} (x : TileCarrier dtype) :
    (Tile.scalar x).data PUnit.unit = x := rfl

@[simp] theorem scalar_data_index {dtype : TileDType}
    (x : TileCarrier dtype) (i : TileIndex .scalar) :
    (Tile.scalar x).data i = x := by
  cases i
  rfl

@[simp] theorem scalar_eta {dtype : TileDType} (x : TileCarrier dtype) :
    ({ data := fun _ : TileIndex .scalar => x } : Tile dtype .scalar) =
      Tile.scalar x := rfl

@[simp] theorem vec_data {dtype : TileDType} {n : Nat}
    (f : Fin n → TileCarrier dtype) (i : Fin n) :
    (Tile.vec f).data i = f i := rfl

end Tile

namespace NumericDType

def add : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => x + y
  | .nat, x, y => x + y

def sub : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => x - y
  | .nat, x, y => x - y

def mul : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => x * y
  | .nat, x, y => x * y

noncomputable def div : NumericDType dtype → TileCarrier dtype → TileCarrier dtype → TileCarrier dtype
  | .real, x, y => x / y
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
  pid   : Nat
  undef : RegionName → Nat → ℝ := fun _ _ => 0

instance : Inhabited BlockState :=
  ⟨{ mem := fun _ _ => 0
   , regs := fun _ _ _ => none
   , pid := 0
   , undef := fun _ _ => 0 }⟩

namespace BlockState

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

theorem scatter_readback {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).mem
        region (offsetFn i)
    = valueFn i := by
  have h_nodup : (List.finRange n).Nodup := List.nodup_finRange n
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (List.mem_finRange i)
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

theorem scatter_readback_masked {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (mask : Fin n → Bool)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k =>
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).mem
        region (offsetFn i)
    = if mask i then valueFn i else s.mem region (offsetFn i) := by
  have h_nodup : (List.finRange n).Nodup := List.nodup_finRange n
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem (List.mem_finRange i)
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

def Tile.uop {shape} (op : ℝ → ℝ) (x : Tile .real shape) : Tile .real shape :=
  ⟨fun i => op (x.data i)⟩

noncomputable def Tile.reduceSum {n} (x : Tile .real (.vec n)) : Tile .real .scalar :=
  Tile.scalar (∑ i, x.data i)

noncomputable def Tile.reduceMax {n} (x : Tile .real (.vec n)) : Option (Tile .real .scalar) :=
  match n with
  | 0 => none
  | n' + 1 =>
      some (Tile.scalar
        ((Finset.univ : Finset (Fin (n' + 1))).sup' Finset.univ_nonempty
          (fun i => x.data i)))

def Tile.natToReal {shape} (x : Tile .nat shape) : Tile .real shape :=
  ⟨fun i => (x.data i : ℝ)⟩

noncomputable def evalOp : Op dtype shape → BlockState → Option (Tile dtype shape)
  | .const c, _ => some (Tile.scalar c)
  | .constNat n, _ => some (Tile.scalar n)
  | .constBool b, _ => some (Tile.scalar b)
  | .negInf, _ => some (Tile.scalar (-1e38))
  | .programId, s => some (Tile.scalar s.pid)
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
  | .exp a, s => return Tile.uop Real.exp (← evalOp a s)
  | .log a, s => return Tile.uop Real.log (← evalOp a s)
  | .sigmoid a, s => return Tile.uop Real.sigmoid (← evalOp a s)
  | .sqrt a, s => return Tile.uop Real.sqrt (← evalOp a s)
  | .lt h bc a b, s => return Tile.cop h.lt bc (← evalOp a s) (← evalOp b s)
  | .le h bc a b, s => return Tile.cop h.le bc (← evalOp a s) (← evalOp b s)
  | .eq h bc a b, s => return Tile.cop h.eq bc (← evalOp a s) (← evalOp b s)
  | .gt h bc a b, s => return Tile.cop h.gt bc (← evalOp a s) (← evalOp b s)
  | .ge h bc a b, s => return Tile.cop h.ge bc (← evalOp a s) (← evalOp b s)
  | .ne h bc a b, s => return Tile.cop h.ne bc (← evalOp a s) (← evalOp b s)
  | .max2 bc a b, s => do
      let va ← evalOp a s
      let vb ← evalOp b s
      some (Tile.bop max bc va vb)
  | .reduceMax a, s => do
      let va ← evalOp a s
      va.reduceMax
  | .reduceSum a, s => return Tile.reduceSum (← evalOp a s)
  | .load region off, s => do
      let offsets ← evalOp off s
      some ⟨fun i => s.readMem region (offsets.data i)⟩
  | .loadMask region off mask, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then s.readMem region addr else s.undef region addr⟩
  | .loadMaskOther region off mask other, s => do
      let offsets ← evalOp off s
      let masks ← evalOp mask s
      let others ← evalOp other s
      some ⟨fun i =>
        let addr := offsets.data i
        if masks.data i then s.readMem region addr else others.data i⟩
  | .natToReal a, s => return Tile.natToReal (← evalOp a s)

mutual

noncomputable def stepStmt : Stmt → BlockState → Option BlockState
  | .assign dtype shape name e, s => do
      let v ← evalOp e s
      some (s.setReg name dtype shape v)
  | .store region shape off val, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      match shape with
      | .scalar =>
          some (s.writeMem region (offsets.data PUnit.unit) (values.data PUnit.unit))
      | .vec n =>
          some ((List.finRange n).foldl
            (fun acc i => acc.writeMem region (offsets.data i) (values.data i)) s)
      | .mat m n =>
          let idxs : List (Fin m × Fin n) :=
            (List.finRange m).flatMap (fun i => (List.finRange n).map (fun j => (i, j)))
          some (idxs.foldl
            (fun acc i => acc.writeMem region (offsets.data i) (values.data i)) s)
  | .storeMask region shape off val mask, s => do
      let offsets ← evalOp off s
      let values ← evalOp val s
      let masks ← evalOp mask s
      match shape with
      | .scalar =>
          some (if masks.data PUnit.unit then
            s.writeMem region (offsets.data PUnit.unit) (values.data PUnit.unit)
          else s)
      | .vec n =>
          some ((List.finRange n).foldl
            (fun acc i =>
              if masks.data i then acc.writeMem region (offsets.data i) (values.data i)
              else acc) s)
      | .mat m n =>
          let idxs : List (Fin m × Fin n) :=
            (List.finRange m).flatMap (fun i => (List.finRange n).map (fun j => (i, j)))
          some (idxs.foldl
            (fun acc i =>
              if masks.data i then acc.writeMem region (offsets.data i) (values.data i)
              else acc) s)
  | .forLoop idx n body, s =>
      stepForLoopAux idx 0 n body s
termination_by st _ => (sizeOf st, 0)
decreasing_by
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
        match stepStmts body (s.setReg idx .nat .scalar (Tile.scalar start)) with
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
      = (stepStmts body (s.setReg idx .nat .scalar (Tile.scalar start))).bind
          (stepForLoopAux idx (start + 1) n body) := by
  conv_lhs => unfold stepForLoopAux
  simp [h]
  cases hbody : stepStmts body (s.setReg idx .nat .scalar (Tile.scalar start)) <;> rfl

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
    cases shape <;> simp at h
    · cases h
      simp
    · cases h
      simp [BlockState.foldl_writeMem_pid]
    · cases h
      simp [BlockState.foldl_writeMem_pid]
  case storeMask region shape off val mask =>
    cases hoff : evalOp off s <;> simp [stepStmt, hoff] at h
    rename_i offsets
    cases hval : evalOp val s <;> simp [hval] at h
    rename_i values
    cases hmask : evalOp mask s <;> simp [hmask] at h
    rename_i masks
    cases shape <;> simp at h
    · split at h
      · cases h
        simp
      · cases h
        rfl
    · cases h
      simp [BlockState.foldl_writeMem_masked_pid]
    · cases h
      simp [BlockState.foldl_writeMem_masked_pid]
  case forLoop idx n body =>
    simp at h
    exact stepForLoopAux_pid h

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
    cases hbody : stepStmts body (s.setReg idx .nat .scalar (Tile.scalar start)) <;>
      simp [hbody] at h
    rename_i mid
    exact (stepForLoopAux_pid h).trans
      ((stepStmts_pid hbody).trans (BlockState.setReg_pid s idx .nat .scalar (Tile.scalar start)))
  · have hge : n ≤ start := Nat.le_of_not_gt hlt
    rw [stepForLoopAux.step_ge hge] at h
    simp_all

end

theorem exec_pid {k : Kernel} {s s' : BlockState}
    (h : exec k s = some s') :
    s'.pid = s.pid := by
  exact stepStmts_pid h

example : evalOp (.const 5) default = some (Tile.scalar 5) := by
  unfold evalOp
  rfl

example : evalOp (.constNat 7) default = some (Tile.scalar 7) := by
  unfold evalOp
  rfl

example : evalOp .programId default = some (Tile.scalar 0) := by
  unfold evalOp
  rfl

example : evalOp (.add .real .same (.const 1) (.const 2)) default
    = some (Tile.scalar 3) := by
  show some (Tile.scalar ((1 : ℝ) + 2)) = some (Tile.scalar 3)
  norm_num

example : evalOp (.add .nat .same (.constNat 2) (.constNat 3)) default
    = some (Tile.scalar 5) := by
  unfold evalOp Tile.bop NumericDType.add
  rfl

end VeriTile.Triton
