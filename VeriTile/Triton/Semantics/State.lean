/-
VeriTile.Triton.Semantics.State

Execution state, register updates, memory writes, and readback lemmas.
-/

import VeriTile.Triton.Semantics.Scalar

namespace VeriTile.Triton

/-- A typed register file. The same register name can only be observed at the
    dtype/shape requested by the typed AST node that references it. -/
abbrev RegFile :=
  (dtype : TileDType) → (shape : TileShape) → RegName → Option (Tile dtype shape)

/-- Dynamically typed scalar memory cell.

Memory is typed at the cell boundary, while proof-facing Real contracts continue
to use `BlockState.readMem` / `BlockState.writeMem`. -/
abbrev MemCell := Sigma TileCarrier

namespace MemCell

def of (dtype : TileDType) (value : TileCarrier dtype) : MemCell :=
  ⟨dtype, value⟩

def real (value : ℝ) : MemCell :=
  of .real (some value)

instance (n : Nat) : OfNat MemCell n where
  ofNat := real n

def readAs (cell : MemCell) (dtype : TileDType) : Option (TileCarrier dtype) :=
  match cell with
  | ⟨cellDType, value⟩ =>
      if h : cellDType = dtype then
        some (cast (by rw [h]) value)
      else
        none

@[simp] theorem readAs_of_same (dtype : TileDType) (value : TileCarrier dtype) :
    (MemCell.of dtype value).readAs dtype = some value := by
  cases dtype <;> rfl

def eraseDType : MemCell → MemCell
  | ⟨.real, value⟩ => of .real value
  | ⟨.fp32, value⟩ => of .real (FloatDType.fp32.toWithBot value)
  | ⟨.fp16, value⟩ => of .real (FloatDType.fp16.toWithBot value)
  | ⟨.bf16, value⟩ => of .real (FloatDType.bf16.toWithBot value)
  | ⟨.int, value⟩ => of .int value
  | ⟨.nat, value⟩ => of .nat value
  | ⟨.bool, value⟩ => of .bool value
  | ⟨.ptr, value⟩ => of .ptr value
  | ⟨.blockPtr, value⟩ => of .blockPtr value

@[simp] theorem readAs_real_real (value : ℝ) :
    (MemCell.real value).readAs .real = some (some value : WithBot ℝ) := rfl

@[simp] theorem eraseDType_real (value : ℝ) :
    (MemCell.real value).eraseDType = MemCell.real value := rfl

end MemCell

/--
A block-level execution state.

`regs` is typed directly; there is no dynamically tagged runtime value.
`undef` models Triton's masked load with `other=None`.
-/
structure BlockState where
  mem   : RegionName → Nat → MemCell
  regs  : RegFile
  /-- Per-axis program IDs. `pids axis` is the value of
  `tl.program_id(axis)`. Total over `Nat`; out-of-range axes default to `0`
  (or whatever the launch model picks). FA-1's batch/head launch grid uses
  axes 0/1/2 for `(q_block, head, batch)`. -/
  pids  : Nat → Nat
  undef : RegionName → Nat → ℝ := fun _ _ => 0

instance : Inhabited BlockState :=
  ⟨{ mem := fun _ _ => MemCell.real 0
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
      if r = region ∧ o = offset then MemCell.real v else s.mem r o }

/-- Write a typed floating scalar to memory. Stores use Triton's finite fallback
for `⊥`, matching the previous Real-only semantics. -/
def writeMemAs (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) : BlockState :=
  { s with mem := fun r o =>
      if r = region ∧ o = offset then
        MemCell.of dtype.toTileDType (dtype.ofReal (dtype.storeValue v))
      else
        s.mem r o }

/-- Write a typed scalar to memory.

Floating channels use the same finite fallback for `⊥` as the legacy Real
path. Non-floating channels write their carrier exactly. -/
def writeMemTyped (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) : BlockState :=
  match dtype with
  | .real => s.writeMemAs .real region offset v
  | .fp32 => s.writeMemAs .fp32 region offset v
  | .fp16 => s.writeMemAs .fp16 region offset v
  | .bf16 => s.writeMemAs .bf16 region offset v
  | .int =>
      { s with mem := fun r o =>
          if r = region ∧ o = offset then MemCell.of .int v else s.mem r o }
  | .nat =>
      { s with mem := fun r o =>
          if r = region ∧ o = offset then MemCell.of .nat v else s.mem r o }
  | .bool =>
      { s with mem := fun r o =>
          if r = region ∧ o = offset then MemCell.of .bool v else s.mem r o }
  | .ptr =>
      { s with mem := fun r o =>
          if r = region ∧ o = offset then MemCell.of .ptr v else s.mem r o }
  | .blockPtr =>
      { s with mem := fun r o =>
          if r = region ∧ o = offset then MemCell.of .blockPtr v else s.mem r o }

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

@[simp] theorem writeMemAs_regs (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType)
    (dtype' : TileDType) (shape : TileShape) (name : RegName) :
    (s.writeMemAs dtype region offset v).regs dtype' shape name =
      s.regs dtype' shape name := rfl

@[simp] theorem writeMemAs_pids (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) :
    (s.writeMemAs dtype region offset v).pids = s.pids := rfl

@[simp] theorem writeMemAs_pid (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType) :
    (s.writeMemAs dtype region offset v).pid = s.pid := rfl

@[simp] theorem writeMemAs_undef (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType)
    (r : RegionName) (o : Nat) :
    (s.writeMemAs dtype region offset v).undef r o = s.undef r o := rfl

@[simp] theorem writeMemTyped_regs (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype)
    (dtype' : TileDType) (shape : TileShape) (name : RegName) :
    (s.writeMemTyped dtype region offset v).regs dtype' shape name =
      s.regs dtype' shape name := by
  cases dtype <;> rfl

@[simp] theorem writeMemTyped_pids (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) :
    (s.writeMemTyped dtype region offset v).pids = s.pids := by
  cases dtype <;> rfl

@[simp] theorem writeMemTyped_pid (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype) :
    (s.writeMemTyped dtype region offset v).pid = s.pid := by
  cases dtype <;> rfl

@[simp] theorem writeMemTyped_undef (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype)
    (r : RegionName) (o : Nat) :
    (s.writeMemTyped dtype region offset v).undef r o = s.undef r o := by
  cases dtype <;> rfl

/-- Read a single scalar from memory. -/
@[inline] def readMem (s : BlockState) (region : RegionName) (offset : Nat) : ℝ :=
  match (s.mem region offset).readAs .real with
  | some (some value) => value
  | _ => 0

/-- Read a typed floating scalar from memory. Mismatched cells read as zero,
which preserves the existing total-memory behavior at the proof surface. -/
@[inline] def readMemAs (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) : TileCarrier dtype.toTileDType :=
  match (s.mem region offset).readAs dtype.toTileDType with
  | some value => dtype.ofReal (dtype.storeValue value)
  | none => dtype.ofReal 0

/-- Read a typed scalar cell exactly. Unlike `readMem` / `readMemAs`, this keeps
dtype mismatch observable. Use this for typed-memory examples and tests rather
than projecting `BlockState.mem` directly. -/
@[inline] def readMemTyped (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) : Option (TileCarrier dtype) :=
  (s.mem region offset).readAs dtype

def defaultCarrier : (dtype : TileDType) → TileCarrier dtype
  | .real => some 0
  | .fp32 => some 0
  | .fp16 => some 0
  | .bf16 => some 0
  | .int => 0
  | .nat => 0
  | .bool => false
  | .ptr => ("", 0)
  | .blockPtr =>
      { region := "", baseOffset := 0, parentShape := [], blockShape := [],
        strides := [], offsets := [] }

/-- Read a typed scalar for operational `tl.load`. Floating channels keep the
current finite-fallback semantics; non-floating channels read exact typed cells
and use the dtype default on mismatch. -/
@[inline] def readMemValue (s : BlockState) (dtype : TileDType)
    (region : RegionName) (offset : Nat) : TileCarrier dtype :=
  match dtype with
  | .real => s.readMemAs .real region offset
  | .fp32 => s.readMemAs .fp32 region offset
  | .fp16 => s.readMemAs .fp16 region offset
  | .bf16 => s.readMemAs .bf16 region offset
  | .int =>
      match s.readMemTyped .int region offset with
      | some value => value
      | none => defaultCarrier .int
  | .nat =>
      match s.readMemTyped .nat region offset with
      | some value => value
      | none => defaultCarrier .nat
  | .bool =>
      match s.readMemTyped .bool region offset with
      | some value => value
      | none => defaultCarrier .bool
  | .ptr =>
      match s.readMemTyped .ptr region offset with
      | some value => value
      | none => defaultCarrier .ptr
  | .blockPtr =>
      match s.readMemTyped .blockPtr region offset with
      | some value => value
      | none => defaultCarrier .blockPtr

@[simp] theorem setReg_readMem (s : BlockState) (name : RegName)
    (dtype : TileDType) (shape : TileShape) (v : Tile dtype shape)
    (region : RegionName) (offset : Nat) :
    (s.setReg name dtype shape v).readMem region offset = s.readMem region offset := rfl

@[simp] theorem setReg_readMemAs (s : BlockState) (name : RegName)
    (regDType : TileDType) (shape : TileShape) (v : Tile regDType shape)
    (memDType : FloatDType) (region : RegionName) (offset : Nat) :
    (s.setReg name regDType shape v).readMemAs memDType region offset =
      s.readMemAs memDType region offset := rfl

@[simp] theorem setReg_readMemTyped (s : BlockState) (name : RegName)
    (regDType : TileDType) (shape : TileShape) (v : Tile regDType shape)
    (memDType : TileDType) (region : RegionName) (offset : Nat) :
    (s.setReg name regDType shape v).readMemTyped memDType region offset =
      s.readMemTyped memDType region offset := rfl

@[simp] theorem setReg_readMemValue (s : BlockState) (name : RegName)
    (regDType : TileDType) (shape : TileShape) (v : Tile regDType shape)
    (memDType : TileDType) (region : RegionName) (offset : Nat) :
    (s.setReg name regDType shape v).readMemValue memDType region offset =
      s.readMemValue memDType region offset := by
  cases memDType <;> rfl

@[simp] theorem readMemValue_real (s : BlockState) (region : RegionName) (offset : Nat) :
    s.readMemValue .real region offset = some (s.readMem region offset) := by
  unfold readMemValue
  unfold readMemAs readMem
  cases h : (s.mem region offset).readAs .real with
  | none => simp [h]
  | some value =>
      cases value
      · simp [h]
        change some (0 : ℝ) = some 0
        rfl
      · rename_i value
        simp [h]
        change some value = some value
        rfl

@[simp] theorem readMemAs_real (s : BlockState) (region : RegionName) (offset : Nat) :
    s.readMemAs .real region offset = some (s.readMem region offset) := by
  unfold readMemAs readMem
  cases h : (s.mem region offset).readAs .real with
  | none => simp [h]
  | some value =>
      cases value
      · simp [h]
        change some (0 : ℝ) = some 0
        rfl
      · rename_i value
        simp [h]
        change some value = some value
        rfl

@[simp] theorem writeMem_readMem (s : BlockState) (region : RegionName)
    (offset : Nat) (v : ℝ) (r : RegionName) (o : Nat) :
    (s.writeMem region offset v).readMem r o =
      if r = region ∧ o = offset then v else s.readMem r o := by
  unfold writeMem readMem
  by_cases h : r = region ∧ o = offset
  · rcases h with ⟨rfl, rfl⟩
    dsimp
    rw [if_pos (show r = r ∧ o = o from ⟨rfl, rfl⟩),
      if_pos (show r = r ∧ o = o from ⟨rfl, rfl⟩)]
    change (match (MemCell.real v).readAs .real with
      | some (some value) => value
      | _ => 0) = v
    rfl
  · simp [h]

@[simp] theorem writeMemAs_readMemAs (s : BlockState) (dtype : FloatDType)
    (region : RegionName) (offset : Nat) (v : TileCarrier dtype.toTileDType)
    (r : RegionName) (o : Nat) :
    (s.writeMemAs dtype region offset v).readMemAs dtype r o =
      if r = region ∧ o = offset then dtype.ofReal (dtype.storeValue v)
      else s.readMemAs dtype r o := by
  unfold writeMemAs readMemAs
  by_cases h : r = region ∧ o = offset
  · rcases h with ⟨rfl, rfl⟩
    dsimp
    rw [if_pos (show r = r ∧ o = o from ⟨rfl, rfl⟩),
      if_pos (show r = r ∧ o = o from ⟨rfl, rfl⟩)]
    change (match (MemCell.of dtype.toTileDType
        (dtype.ofReal (dtype.storeValue v))).readAs dtype.toTileDType with
      | some value => dtype.ofReal (dtype.storeValue value)
      | none => dtype.ofReal 0) = dtype.ofReal (dtype.storeValue v)
    cases dtype <;> rfl
  · simp [h]

@[simp] theorem writeMemAs_real (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier FloatDType.real.toTileDType) :
    s.writeMemAs .real region offset v = s.writeMem region offset (FloatDType.real.storeValue v) := by
  unfold writeMemAs writeMem
  rfl

@[simp] theorem writeMemTyped_real (s : BlockState)
    (region : RegionName) (offset : Nat) (v : TileCarrier TileDType.real) :
    s.writeMemTyped .real region offset v =
      s.writeMem region offset (FloatDType.real.storeValue v) := by
  unfold writeMemTyped
  simp

private theorem foldl_writeMem_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, offsetFn k ≠ o) →
      ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem region o)
      = s.readMem region o := by
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s h
    have hhd : offsetFn hd ≠ o := h hd (List.mem_cons_self)
    have htl : ∀ k ∈ tl, offsetFn k ≠ o :=
      fun k hk => h k (List.mem_cons_of_mem hd hk)
    rw [List.foldl_cons, ih _ htl]
    rw [writeMem_readMem]
    show (if region = region ∧ o = offsetFn hd then valueFn hd else s.readMem region o)
        = s.readMem region o
    rw [if_neg]
    rintro ⟨_, h_eq⟩
    exact hhd h_eq.symm

theorem scatter_readback_list {α : Type} {region : RegionName}
    (l : List α) (s : BlockState) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (i : α) (h_nodup : l.Nodup) (h_mem : i ∈ l)
    (h_inj : Function.Injective offsetFn) :
    (l.foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem region (offsetFn i)
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
  rw [writeMem_readMem]
  show (if region = region ∧ offsetFn i = offsetFn i then valueFn i else _)
      = valueFn i
  exact if_pos ⟨rfl, rfl⟩

theorem scatter_readback {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem region (offsetFn i)
      = valueFn i := by
  exact scatter_readback_list (List.finRange n) s offsetFn valueFn
    i (List.nodup_finRange n) (List.mem_finRange i) h_inj

theorem scatter_readback_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem region (offsetFn i)
    = valueFn i :=
  scatter_readback_list (TileShape.allIndices shape) s offsetFn valueFn
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) h_inj

private theorem foldl_writeMem_masked_preserves {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α) :
    ∀ (s : BlockState), (∀ k ∈ l, mask k = true → offsetFn k ≠ o) →
      ((l.foldl
          (fun acc k =>
            if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region o)
      = s.readMem region o := by
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
      rw [writeMem_readMem]
      show (if region = region ∧ o = offsetFn hd then valueFn hd else s.readMem region o)
          = s.readMem region o
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
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if mask i then valueFn i else s.readMem region (offsetFn i) := by
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
    rw [writeMem_readMem]
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
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if mask i then valueFn i else s.readMem region (offsetFn i) :=
  scatter_readback_masked_list (List.finRange n) s offsetFn valueFn mask
    i (List.nodup_finRange n) (List.mem_finRange i) h_inj

theorem scatter_readback_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (mask : TileIndex shape → Bool)
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k =>
         if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if mask i then valueFn i else s.readMem region (offsetFn i) :=
  scatter_readback_masked_list (TileShape.allIndices shape) s offsetFn valueFn mask
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) h_inj

theorem scatter_readback_prop_masked {region : RegionName} {n : Nat}
    (s : BlockState) (offsetFn : Fin n → Nat) (valueFn : Fin n → ℝ)
    (P : Fin n → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : Fin n) :
    ((List.finRange n).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if P i then valueFn i else s.readMem region (offsetFn i) := by
  have h := scatter_readback_masked (region := region) s offsetFn valueFn
              (fun k => decide (P k)) h_inj i
  simp only [decide_eq_true_eq] at h
  exact h

theorem scatter_readback_prop_masked_nd {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (P : TileIndex shape → Prop) [DecidablePred P]
    (h_inj : Function.Injective offsetFn) (i : TileIndex shape) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = if P i then valueFn i else s.readMem region (offsetFn i) := by
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
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
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
      st l₂).readMem region (offsetFn i) = valueFn i
  rw [show List.foldl
      (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st l₂ =
    List.foldl
      (fun acc k => if decide (P k) then acc.writeMem region (offsetFn k) (valueFn k) else acc)
      st l₂ by
        rw [hstep]]
  rw [hpres]
  simp [BlockState.readMem, BlockState.writeMem, st]

theorem scatter_readback_prop_masked_nd_of_true {region : RegionName} {shape : TileShape}
    (s : BlockState) (offsetFn : TileIndex shape → Nat)
    (valueFn : TileIndex shape → ℝ) (P : TileIndex shape → Prop) [DecidablePred P]
    (i : TileIndex shape) (hPi : P i)
    (h_no_collision :
      ∀ k, P k → offsetFn k = offsetFn i → k = i) :
    ((TileShape.allIndices shape).foldl
       (fun acc k => if P k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region (offsetFn i)
    = valueFn i :=
  scatter_readback_prop_masked_list_of_true (TileShape.allIndices shape) s offsetFn valueFn P
    i (TileShape.allIndices_nodup shape) (TileShape.mem_allIndices shape i) hPi
    (fun k _hk hPk heq => h_no_collision k hPk heq)

theorem scatter_preserves_other_region {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (R : RegionName) (h_ne : R ≠ region) (off : Nat) :
    ∀ (l : List α) (s : BlockState),
      ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).readMem R off)
      = s.readMem R off := by
  intro l
  induction l with
  | nil => intros; rfl
  | cons hd tl ih =>
    intro s
    rw [List.foldl_cons, ih]
    show (s.writeMem region (offsetFn hd) (valueFn hd)).readMem R off = s.readMem R off
    rw [writeMem_readMem]
    show (if R = region ∧ off = offsetFn hd then valueFn hd else s.readMem R off)
        = s.readMem R off
    rw [if_neg]
    rintro ⟨h_R, _⟩
    exact h_ne h_R

@[simp] theorem foldl_writeMem_regs {α : Type}
    (region : RegionName) (offsetFn : α → Nat) (valueFn : α → ℝ)
    (l : List α) (s : BlockState) (dtype : TileDType) (shape : TileShape)
    (name : RegName) :
    ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn k)) s).regs
      dtype shape name) = s.regs dtype shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      rfl

@[simp] theorem foldl_writeMem_dependent_regs {α : Type}
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : BlockState → α → ℝ)
    (l : List α) (s : BlockState) (dtype : TileDType) (shape : TileShape)
    (name : RegName) :
    ((l.foldl (fun acc k => acc.writeMem region (offsetFn k) (valueFn acc k)) s).regs
      dtype shape name) = s.regs dtype shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      rfl

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

theorem foldl_writeMemAs_pid {α : Type} (dtype : FloatDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype.toTileDType)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMemAs dtype region (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMemAs_masked_pid {α : Type} (dtype : FloatDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype.toTileDType)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemAs dtype region (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemAsAt_pid {α : Type} (dtype : FloatDType)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype.toTileDType)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMemAs dtype (regionFn k) (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMemAsAt_masked_pid {α : Type} (dtype : FloatDType)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype.toTileDType)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemAs dtype (regionFn k) (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemTyped_pid {α : Type} (dtype : TileDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMemTyped dtype region (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMem_masked_regs {α : Type}
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → ℝ)
    (mask : α → Bool) (l : List α) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    ((l.foldl
        (fun acc k => if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).regs
        dtype shape name)
      = s.regs dtype shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemTyped_regs {α : Type} (dtype : TileDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (l : List α) (s : BlockState)
    (dtype' : TileDType) (shape : TileShape) (name : RegName) :
    ((l.foldl (fun acc k => acc.writeMemTyped dtype region (offsetFn k) (valueFn k)) s).regs
        dtype' shape name)
      = s.regs dtype' shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMemTyped_masked_pid {α : Type} (dtype : TileDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemTyped_masked_regs {α : Type} (dtype : TileDType)
    (region : RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (mask : α → Bool) (l : List α) (s : BlockState)
    (dtype' : TileDType) (shape : TileShape) (name : RegName) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemTyped dtype region (offsetFn k) (valueFn k) else acc) s).regs
        dtype' shape name)
      = s.regs dtype' shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

theorem foldl_writeMemTypedAt_pid {α : Type} (dtype : TileDType)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (l : List α) (s : BlockState) :
    ((l.foldl (fun acc k => acc.writeMemTyped dtype (regionFn k) (offsetFn k) (valueFn k)) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      simp

theorem foldl_writeMemTypedAt_masked_pid {α : Type} (dtype : TileDType)
    (regionFn : α → RegionName) (offsetFn : α → Nat)
    (valueFn : α → TileCarrier dtype)
    (mask : α → Bool) (l : List α) (s : BlockState) :
    ((l.foldl
        (fun acc k =>
          if mask k then acc.writeMemTyped dtype (regionFn k) (offsetFn k) (valueFn k) else acc) s).pid)
      = s.pid := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
      rw [List.foldl_cons, ih]
      by_cases hmask : mask hd
      · simp [hmask]
      · simp [hmask]

end BlockState

end VeriTile.Triton
