/-
VeriTile.Triton.Memory.Flatten

Flat-memory bridge (stage 2 of the address-layout realism roadmap in
`documents/GpuMemoryModel.md`).

The operational model addresses memory as `(RegionName, Nat)` with a total
`mem : RegionName → Nat → MemCell`; distinct tensors live in distinct regions,
so non-aliasing holds *by construction*. Real kernels receive raw pointers
into one flat address space. This file closes that gap without re-proving any
kernel: a `FlatAlloc` places finitely many regions at disjoint base offsets
inside one designated flat region, and the bridge theorems show that the
*translated* kernel (every region access rewritten to `flat` + shifted
offsets) run on the *flattened* state steps to the flattened result. Every
region-model theorem then transports to a flat-memory corollary.

Design notes:

- Pointer and block-pointer values are first-class (`TileCarrier .ptr =
  RegionName × Nat`, `TileCarrier .blockPtr = BlockPtr`) and can sit in
  registers and memory cells, so flattening translates *values* as well as
  syntax: `(r, o) ↦ (flat, base r + o)` and a `BlockPtr` shifts its
  `baseOffset` into the flat interval of its region. On the four float
  channels and `.int`/`.nat`/`.bool` the value translation is the identity.

- The in-bounds side conditions come from the #48 memory-safety layer
  (`RegionBounds`, `Op.MemorySafe`, `MemAccess.ActiveAddressSafe` in
  `VeriTile.Triton.Memory.Bounds`): the allocation's `extent` *is* a
  `RegionBounds`, and active-lane addresses below the extent are exactly
  what keeps the flat images of distinct regions from colliding.

- Masked loads read `s.undef region offset` on inactive real lanes, and
  inactive-lane offsets are *not* bounded by the safety contracts, so the
  bridge assumes the default `undef = 0` (which every practical state uses);
  the flattened state's `undef` is definitionally `0`.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory.Bounds

namespace VeriTile.Triton

/-! ## Allocation -/

/-- An allocation of finitely many named regions into one flat address
space: region `r` occupies the interval `[base r, base r + extent r)` of the
designated `flat` region. `extent` doubles as the #48 `RegionBounds` map used
for the in-bounds side conditions. -/
structure FlatAlloc where
  /-- The single region every translated access lands in. -/
  flat : RegionName
  /-- Finite support: the regions the kernel may touch. -/
  regions : List RegionName
  /-- Base offset of each region inside `flat`. -/
  base : RegionName → Nat
  /-- Capacity of each region (also the `RegionBounds` for #48 contracts). -/
  extent : RegionBounds

namespace FlatAlloc

variable (A : FlatAlloc)

/-- Flat address of `(r, o)`. -/
def addr (r : RegionName) (o : Nat) : Nat := A.base r + o

/-- The allocated intervals of distinct listed regions do not overlap. -/
def Disjoint : Prop :=
  ∀ r ∈ A.regions, ∀ r' ∈ A.regions, r ≠ r' →
    A.base r + A.extent r ≤ A.base r' ∨ A.base r' + A.extent r' ≤ A.base r

/-- Decode a flat offset back to the unique listed region covering it. -/
def decode (o' : Nat) : Option (RegionName × Nat) :=
  (A.regions.find? fun r =>
    decide (A.base r ≤ o' ∧ o' < A.base r + A.extent r)).map
    fun r => (r, o' - A.base r)

/-- Under disjointness, decoding the flat address of an in-bounds `(r, o)`
recovers `(r, o)`. -/
theorem decode_addr (hd : A.Disjoint) {r : RegionName} (hr : r ∈ A.regions)
    {o : Nat} (ho : o < A.extent r) :
    A.decode (A.addr r o) = some (r, o) := by
  unfold decode addr
  have hex : ∃ x ∈ A.regions,
      (fun r' => decide (A.base r' ≤ A.base r + o ∧
        A.base r + o < A.base r' + A.extent r')) x = true :=
    ⟨r, hr, by simp; omega⟩
  obtain ⟨r', hfind⟩ := Option.isSome_iff_exists.mp
    (List.find?_isSome.mpr hex)
  have hr' : r' ∈ A.regions := List.mem_of_find?_eq_some hfind
  have hpred := List.find?_some hfind
  simp only [decide_eq_true_eq] at hpred
  have hrr : r' = r := by
    by_contra hne
    rcases hd r' hr' r hr hne with h | h <;> omega
  subst hrr
  rw [hfind]
  simp

/-! ## Value translation

Pointer-shaped values are relocated into the flat region; every other carrier
is untouched. -/

/-- Translate one carrier value: `.ptr` pairs and `.blockPtr` records move to
the flat region; all data channels are the identity. -/
def trCarrier : (d : TileDType) → TileCarrier d → TileCarrier d
  | .real, v => v
  | .fp32, v => v
  | .fp16, v => v
  | .bf16, v => v
  | .int,  v => v
  | .nat,  v => v
  | .bool, v => v
  | .ptr,  p => (A.flat, A.addr p.1 p.2)
  | .blockPtr, bp =>
      { bp with region := A.flat, baseOffset := A.addr bp.region bp.baseOffset }

/-- Translate a memory cell (keeps the dtype tag). -/
def trCell : MemCell → MemCell
  | ⟨d, v⟩ => ⟨d, A.trCarrier d v⟩

/-- Translate a tile pointwise. -/
def trTile {d : TileDType} {sh : TileShape} (t : Tile d sh) : Tile d sh :=
  ⟨fun i => A.trCarrier d (t.data i)⟩

/-- Translate a register file pointwise. -/
def trRegs (regs : RegFile) : RegFile :=
  fun d sh n => (regs d sh n).map A.trTile

/-! ## State flattening -/

/-- One flat cell: decode the offset and read the translated source cell
(default cell outside every allocated interval). -/
noncomputable def readFlat (s : BlockState) (o' : Nat) : MemCell :=
  match A.decode o' with
  | some (r, o) => A.trCell (s.mem r o)
  | none => MemCell.real 0

/-- The flattened state: all listed regions' cells are laid out inside `flat`
at their allocated intervals (values translated); every other flat offset and
every other region reads as the default cell. Registers are translated
pointwise; `pids`/`numPids` are preserved; `undef` is the default `0` (see the
design note in the module docstring). -/
noncomputable def flattenState (s : BlockState) : BlockState where
  mem := fun r' o' => if r' = A.flat then A.readFlat s o' else MemCell.real 0
  regs := A.trRegs s.regs
  pids := s.pids
  undef := fun _ _ => 0
  numPids := s.numPids

/-- Reading the flat image of an in-bounds `(r, o)` sees the translated
source cell. -/
theorem flattenState_mem_addr (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r) :
    (A.flattenState s).mem A.flat (A.addr r o) = A.trCell (s.mem r o) := by
  show (if A.flat = A.flat then A.readFlat s (A.addr r o) else MemCell.real 0)
      = A.trCell (s.mem r o)
  rw [if_pos rfl]
  unfold readFlat
  rw [A.decode_addr hd hr ho]

/-! ## Syntax translation -/

/-- Broadcast shape for adding a scalar base offset onto a `sh`-shaped
offset tile. -/
def shiftBroadcast : (sh : TileShape) → Broadcast [] sh sh
  | [] => .nil
  | _ :: _ => .scalarL

set_option maxHeartbeats 1600000 in
/-- Translate an `Op`: direct-region loads land in `flat` with shifted
offsets, pointer/block-pointer constructors relocate their region argument,
and everything else is a congruence. -/
def flattenOp (A : FlatAlloc) :
    {d : TileDType} → {sh : TileShape} → Op d sh → Op d sh
  | _, _, .const x => .const x
  | _, _, .constFloat dt x => .constFloat dt x
  | _, _, .constNat n => .constNat n
  | _, _, .constInt n => .constInt n
  | _, _, .constBool b => .constBool b
  | _, _, .negInf => .negInf
  | _, _, .programId ax => .programId ax
  | _, _, .numPrograms ax => .numPrograms ax
  | _, _, .ref d sh n => .ref d sh n
  | _, _, .arange n => .arange n
  | _, _, .broadcast e sh => .broadcast (A.flattenOp e) sh
  | _, _, .full sh e => .full sh (A.flattenOp e)
  | _, _, .castFloat src dst a => .castFloat src dst (A.flattenOp a)
  | _, _, .castNatToInt a => .castNatToInt (A.flattenOp a)
  | _, _, .castIntToNat a => .castIntToNat (A.flattenOp a)
  | _, _, .castRealToInt8 a => .castRealToInt8 (A.flattenOp a)
  | _, _, .add nd bc a b => .add nd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .sub nd bc a b => .sub nd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .mul nd bc a b => .mul nd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .div nd bc a b => .div nd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .floorDiv id bc a b => .floorDiv id bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .mod id bc a b => .mod id bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .bitAnd bc a b => .bitAnd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .bitOr bc a b => .bitOr bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .bitXor bc a b => .bitXor bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .shiftLeft bc a b => .shiftLeft bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .shiftRight bc a b => .shiftRight bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .exp a => .exp (A.flattenOp a)
  | _, _, .exp2 a => .exp2 (A.flattenOp a)
  | _, _, .log a => .log (A.flattenOp a)
  | _, _, .log2 a => .log2 (A.flattenOp a)
  | _, _, .sigmoid a => .sigmoid (A.flattenOp a)
  | _, _, .sqrt a => .sqrt (A.flattenOp a)
  | _, _, .rsqrt a => .rsqrt (A.flattenOp a)
  | _, _, .tanh a => .tanh (A.flattenOp a)
  | _, _, .sin a => .sin (A.flattenOp a)
  | _, _, .cos a => .cos (A.flattenOp a)
  | _, _, .tan a => .tan (A.flattenOp a)
  | _, _, .atan a => .atan (A.flattenOp a)
  | _, _, .cosh a => .cosh (A.flattenOp a)
  | _, _, .sinh a => .sinh (A.flattenOp a)
  | _, _, .erf a => .erf (A.flattenOp a)
  | _, _, .lt cd bc a b => .lt cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .le cd bc a b => .le cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .eq cd bc a b => .eq cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .gt cd bc a b => .gt cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .ge cd bc a b => .ge cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .ne cd bc a b => .ne cd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .boolAnd bc a b => .boolAnd bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .boolOr bc a b => .boolOr bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .boolNot a => .boolNot (A.flattenOp a)
  | _, _, .max2 bc a b => .max2 bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .pow bc a b => .pow bc (A.flattenOp a) (A.flattenOp b)
  | _, _, .where c a b => .where (A.flattenOp c) (A.flattenOp a) (A.flattenOp b)
  | _, _, .ite c a b => .ite (A.flattenOp c) (A.flattenOp a) (A.flattenOp b)
  | _, _, .reduceMax ax kd a => .reduceMax ax kd (A.flattenOp a)
  | _, _, .reduceMaxNat ax kd a => .reduceMaxNat ax kd (A.flattenOp a)
  | _, _, .reduceSum ax kd a => .reduceSum ax kd (A.flattenOp a)
  | _, _, .scan op ax dir a => .scan op ax dir (A.flattenOp a)
  | _, _, .argMax ax a => .argMax ax (A.flattenOp a)
  | _, _, .argMin ax a => .argMin ax (A.flattenOp a)
  | _, _, .sort ax a => .sort ax (A.flattenOp a)
  | _, _, .dot a b => .dot (A.flattenOp a) (A.flattenOp b)
  | _, _, .transpose a => .transpose (A.flattenOp a)
  | _, _, .reshape outSh a => .reshape outSh (A.flattenOp a)
  | _, _, .remap outSh f a => .remap outSh f (A.flattenOp a)
  | _, _, .join a b => .join (A.flattenOp a) (A.flattenOp b)
  | _, _, .split side a => .split side (A.flattenOp a)
  | _, _, .expandDim ax a => .expandDim ax (A.flattenOp a)
  | _, _, .ptrBase r =>
      .ptrAdd .nil (.ptrBase A.flat) (.constNat (A.base (Region.cast r)))
  | _, _, .ptrAdd bc p o => .ptrAdd bc (A.flattenOp p) (A.flattenOp o)
  | _, _, .ptrSub bc p o => .ptrSub bc (A.flattenOp p) (A.flattenOp o)
  | _, _, .makeBlockPtr r bo ps bs strides offs =>
      .makeBlockPtr A.flat (A.addr (Region.cast r) bo) ps bs strides offs
  | _, _, .makeBlockPtrDyn r bo ps bs strides offs =>
      .makeBlockPtrDyn A.flat
        (.add .nat .nil (.constNat (A.base r)) (A.flattenOp bo)) ps bs strides offs
  | _, _, .makeBlockPtrDynOffsets r bo ps bs strides offs =>
      .makeBlockPtrDynOffsets A.flat
        (.add .nat .nil (.constNat (A.base r)) (A.flattenOp bo)) ps bs strides
        (offs.attach.map fun ⟨o, _⟩ => A.flattenOp o)
  | _, _, .advanceBlockPtr p ds => .advanceBlockPtr (A.flattenOp p) ds
  | _, _, .load d mem mask =>
      .load d
        (match mem with
         | .region r off =>
             .region (Region.cast A.flat)
               (.add .nat (shiftBroadcast _)
                 (.constNat (A.base (Region.cast r))) (A.flattenOp off))
         | .ptr p => .ptr (A.flattenOp p)
         | .blockPtr p bc => .blockPtr (A.flattenOp p) bc)
        (match mask with
         | .none => .none
         | .mask m => .mask (A.flattenOp m)
         | .maskOther m o => .maskOther (A.flattenOp m) (A.flattenOp o))
  | _, _, .natToReal a => .natToReal (A.flattenOp a)

/-- Translate a memory access (the named form of the `.load` case above,
shared with the store/atomic statements). -/
def flattenAccess (A : FlatAlloc) :
    {d : TileDType} → {sh : TileShape} → MemAccess d sh → MemAccess d sh
  | _, _, .region r off =>
      .region (Region.cast A.flat)
        (.add .nat (shiftBroadcast _)
          (.constNat (A.base (Region.cast r))) (A.flattenOp off))
  | _, _, .ptr p => .ptr (A.flattenOp p)
  | _, _, .blockPtr p bc => .blockPtr (A.flattenOp p) bc

/-- Translate a mask option. -/
def flattenMask (A : FlatAlloc) :
    {d : TileDType} → {sh : TileShape} → MaskOpt d sh → MaskOpt d sh
  | _, _, .none => .none
  | _, _, .mask m => .mask (A.flattenOp m)
  | _, _, .maskOther m o => .maskOther (A.flattenOp m) (A.flattenOp o)

/-- The `.load` case of `flattenOp` in terms of the named access/mask
translations. -/
@[simp] theorem flattenOp_load (A : FlatAlloc) {d : TileDType}
    {sh : TileShape} (mem : MemAccess d sh) (mask : MaskOpt d sh) :
    A.flattenOp (.load d mem mask)
      = .load d (A.flattenAccess mem) (A.flattenMask mask) := by
  cases mem <;> cases mask <;>
    simp only [flattenOp, flattenAccess, flattenMask]

/-! ## Statement and kernel translation -/

mutual

/-- Translate one statement. -/
def flattenStmt (A : FlatAlloc) : Stmt → Stmt
  | .assign d sh n e => .assign d sh n (A.flattenOp e)
  | .store d sh mem v mask =>
      .store d sh (A.flattenAccess mem) (A.flattenOp v) (A.flattenMask mask)
  | .atomicAdd nd sh mem v mask =>
      .atomicAdd nd sh (A.flattenAccess mem) (A.flattenOp v) (A.flattenMask mask)
  | .atomicRMW op d sh mem input extra mask dest =>
      .atomicRMW op d sh (A.flattenAccess mem) (A.flattenOp input)
        (extra.map A.flattenOp) (A.flattenMask mask) dest
  | .forLoop idx n body => .forLoop idx n (A.flattenStmts body)
  | .forRange idx start stop step body =>
      .forRange idx start stop step (A.flattenStmts body)
  | .forRangeDyn idx start stop step body =>
      .forRangeDyn idx (A.flattenOp start) (A.flattenOp stop) (A.flattenOp step)
        (A.flattenStmts body)
  | .ifThen c body => .ifThen (A.flattenOp c) (A.flattenStmts body)
  | .ifThenElse c tb eb =>
      .ifThenElse (A.flattenOp c) (A.flattenStmts tb) (A.flattenStmts eb)

/-- Translate a statement list. -/
def flattenStmts (A : FlatAlloc) : List Stmt → List Stmt
  | [] => []
  | st :: rest => A.flattenStmt st :: A.flattenStmts rest

end

/-- Translate a whole kernel: the body is translated and the region metadata
collapses to the single flat region. -/
def flattenKernel (A : FlatAlloc) (k : Kernel) : Kernel where
  inputs := [A.flat]
  outputs := [A.flat]
  body := A.flattenStmts k.body

/-! ## Value-translation identities

On every data channel the carrier translation is the identity; only the two
pointer-shaped channels move. -/

@[simp] theorem trCarrier_real (v : TileCarrier .real) : A.trCarrier .real v = v := rfl
@[simp] theorem trCarrier_fp32 (v : TileCarrier .fp32) : A.trCarrier .fp32 v = v := rfl
@[simp] theorem trCarrier_fp16 (v : TileCarrier .fp16) : A.trCarrier .fp16 v = v := rfl
@[simp] theorem trCarrier_bf16 (v : TileCarrier .bf16) : A.trCarrier .bf16 v = v := rfl
@[simp] theorem trCarrier_int (v : TileCarrier .int) : A.trCarrier .int v = v := rfl
@[simp] theorem trCarrier_nat (v : TileCarrier .nat) : A.trCarrier .nat v = v := rfl
@[simp] theorem trCarrier_bool (v : TileCarrier .bool) : A.trCarrier .bool v = v := rfl

@[simp] theorem trTile_real {sh : TileShape} (t : Tile .real sh) : A.trTile t = t := rfl
@[simp] theorem trTile_fp32 {sh : TileShape} (t : Tile .fp32 sh) : A.trTile t = t := rfl
@[simp] theorem trTile_fp16 {sh : TileShape} (t : Tile .fp16 sh) : A.trTile t = t := rfl
@[simp] theorem trTile_bf16 {sh : TileShape} (t : Tile .bf16 sh) : A.trTile t = t := rfl
@[simp] theorem trTile_int {sh : TileShape} (t : Tile .int sh) : A.trTile t = t := rfl
@[simp] theorem trTile_nat {sh : TileShape} (t : Tile .nat sh) : A.trTile t = t := rfl
@[simp] theorem trTile_bool {sh : TileShape} (t : Tile .bool sh) : A.trTile t = t := rfl

/-! ## Reading through the flattening -/

/-- `readAs` commutes with the cell translation: the dtype tag is preserved,
so the tag test is unchanged and a matching payload comes back translated. -/
theorem readAs_trCell (c : MemCell) (d : TileDType) :
    (A.trCell c).readAs d = (c.readAs d).map (A.trCarrier d) := by
  obtain ⟨cd, v⟩ := c
  unfold trCell MemCell.readAs
  by_cases h : cd = d
  · subst h; simp
  · simp [h]

/-- Float-channel reads are unchanged by flattening at in-bounds addresses
(the float carriers translate identically). -/
theorem flattenState_readMemAs (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (fd : FloatDType) :
    (A.flattenState s).readMemAs fd A.flat (A.addr r o)
      = s.readMemAs fd r o := by
  unfold BlockState.readMemAs
  rw [A.flattenState_mem_addr hd s hr ho, A.readAs_trCell]
  cases hcell : (s.mem r o).readAs fd.toTileDType with
  | none => simp
  | some v =>
      cases fd <;> simp

/-- Exact typed reads come back translated at in-bounds addresses. -/
theorem flattenState_readMemTyped (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (d : TileDType) :
    (A.flattenState s).readMemTyped d A.flat (A.addr r o)
      = (s.readMemTyped d r o).map (A.trCarrier d) := by
  unfold BlockState.readMemTyped
  rw [A.flattenState_mem_addr hd s hr ho, A.readAs_trCell]

/-- The operational `tl.load` read commutes with flattening at in-bounds
addresses on every **data** channel. (`.ptr`/`.blockPtr` element loads are
excluded from the bridge: their dtype-mismatch fallback `defaultCarrier` does
not relocate.) -/
theorem flattenState_readMemValue (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (d : TileDType) (hdt : d ≠ .ptr ∧ d ≠ .blockPtr) :
    (A.flattenState s).readMemValue d A.flat (A.addr r o)
      = s.readMemValue d r o := by
  unfold BlockState.readMemValue
  cases d with
  | ptr => exact absurd rfl hdt.1
  | blockPtr => exact absurd rfl hdt.2
  | real => exact A.flattenState_readMemAs hd s hr ho .real
  | fp32 => exact A.flattenState_readMemAs hd s hr ho .fp32
  | fp16 => exact A.flattenState_readMemAs hd s hr ho .fp16
  | bf16 => exact A.flattenState_readMemAs hd s hr ho .bf16
  | int =>
      rw [A.flattenState_readMemTyped hd s hr ho .int]
      cases s.readMemTyped .int r o <;> simp
  | nat =>
      rw [A.flattenState_readMemTyped hd s hr ho .nat]
      cases s.readMemTyped .nat r o <;> simp
  | bool =>
      rw [A.flattenState_readMemTyped hd s hr ho .bool]
      cases s.readMemTyped .bool r o <;> simp

end FlatAlloc

/-! ## The flattenable fragment

Two constructs are excluded from the v1 bridge (see the module docstring and
the address-layout roadmap in `documents/GpuMemoryModel.md`):

- `Op.ptrSub` is covered under a per-state no-underflow side condition
  (truncated `Nat` subtraction diverges under relocation when it underflows:
  `base + (o ∸ k) ≠ (base + o) ∸ k` for `k > o`).
- element loads at `.ptr`/`.blockPtr` dtype: their dtype-mismatch fallback
  `BlockState.defaultCarrier` is a fixed junk pointer that does not relocate.
  `Op.reshape` at pointer dtype is excluded for the same reason (its
  out-of-range lanes fall back to `defaultCarrier`).


Everything else — including `advanceBlockPtr`, whose offset arithmetic is
orthogonal to the relocation — is covered. The predicate mirrors the #48
`Op.MemorySafe` recursion structure. -/

set_option maxHeartbeats 1600000 in
/-- The `Op` fragment covered by the flat-memory bridge: no `ptrSub`, no
pointer-dtype element loads; recursive everywhere else. -/
def Op.FlattenOk : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .numPrograms _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.FlattenOk
  | .full _ e => e.FlattenOk
  | .castFloat _ _ e => e.FlattenOk
  | .castNatToInt e => e.FlattenOk
  | .castIntToNat e => e.FlattenOk
  | .castRealToInt8 e => e.FlattenOk
  | .add _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .sub _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .mul _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .div _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .floorDiv _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .mod _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .bitAnd _ a b => a.FlattenOk ∧ b.FlattenOk
  | .bitOr _ a b => a.FlattenOk ∧ b.FlattenOk
  | .bitXor _ a b => a.FlattenOk ∧ b.FlattenOk
  | .shiftLeft _ a b => a.FlattenOk ∧ b.FlattenOk
  | .shiftRight _ a b => a.FlattenOk ∧ b.FlattenOk
  | .exp a => a.FlattenOk
  | .exp2 a => a.FlattenOk
  | .log a => a.FlattenOk
  | .log2 a => a.FlattenOk
  | .sigmoid a => a.FlattenOk
  | .sqrt a => a.FlattenOk
  | .rsqrt a => a.FlattenOk
  | .tanh a => a.FlattenOk
  | .sin a => a.FlattenOk
  | .cos a => a.FlattenOk
  | .tan a => a.FlattenOk
  | .atan a => a.FlattenOk
  | .cosh a => a.FlattenOk
  | .sinh a => a.FlattenOk
  | .erf a => a.FlattenOk
  | .lt _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .le _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .eq _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .gt _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .ge _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .ne _ _ a b => a.FlattenOk ∧ b.FlattenOk
  | .boolAnd _ a b => a.FlattenOk ∧ b.FlattenOk
  | .boolOr _ a b => a.FlattenOk ∧ b.FlattenOk
  | .boolNot a => a.FlattenOk
  | .max2 _ a b => a.FlattenOk ∧ b.FlattenOk
  | .pow _ a b => a.FlattenOk ∧ b.FlattenOk
  | .where c a b => c.FlattenOk ∧ a.FlattenOk ∧ b.FlattenOk
  | .ite c a b => c.FlattenOk ∧ a.FlattenOk ∧ b.FlattenOk
  | .reduceMax _ _ a => a.FlattenOk
  | .reduceMaxNat _ _ a => a.FlattenOk
  | .reduceSum _ _ a => a.FlattenOk
  | .scan _ _ _ a => a.FlattenOk
  | .argMax _ a => a.FlattenOk
  | .argMin _ a => a.FlattenOk
  | .sort _ a => a.FlattenOk
  | .dot a b => a.FlattenOk ∧ b.FlattenOk
  | .transpose a => a.FlattenOk
  | .reshape _ a => (dtype ≠ .ptr ∧ dtype ≠ .blockPtr) ∧ a.FlattenOk
  | .remap _ _ a => a.FlattenOk
  | .join a b => a.FlattenOk ∧ b.FlattenOk
  | .split _ a => a.FlattenOk
  | .expandDim _ a => a.FlattenOk
  | .ptrBase _ => True
  | .ptrAdd _ ptr off => ptr.FlattenOk ∧ off.FlattenOk
  | .ptrSub bc p off => p.FlattenOk ∧ off.FlattenOk ∧
      ∀ s ptrs offs, evalOp p s = some ptrs → evalOp off s = some offs →
        ∀ i, offs.data (bc.rightIndex i) ≤ (ptrs.data (bc.leftIndex i)).2
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => base.FlattenOk
  | .makeBlockPtrDynOffsets _ base _ _ _ offsets =>
      base.FlattenOk ∧ ∀ off ∈ offsets, off.FlattenOk
  | .advanceBlockPtr ptr _ => ptr.FlattenOk
  | .load d mem mask =>
      (d ≠ .ptr ∧ d ≠ .blockPtr) ∧
      (match mem with
        | .region _ off => off.FlattenOk
        | .ptr ptr => ptr.FlattenOk
        | .blockPtr ptr _ => ptr.FlattenOk) ∧
      (match mask with
        | .none => True
        | .mask m => m.FlattenOk
        | .maskOther m other => m.FlattenOk ∧ other.FlattenOk)
  | .natToReal a => a.FlattenOk

/-! ## Tile-level commutation of the value translation -/

namespace FlatAlloc

variable (A : FlatAlloc)

/-- On every data channel the tile translation is the identity function. -/
theorem trTileFun_data {d : TileDType} (h1 : d ≠ .ptr) (h2 : d ≠ .blockPtr)
    {sh : TileShape} : (A.trTile (d := d) (sh := sh)) = id := by
  cases d
  case ptr => exact absurd rfl h1
  case blockPtr => exact absurd rfl h2
  all_goals rfl

/-- `trTile` commutes with lane-wise select (the condition is untouched). -/
theorem trTile_select {d : TileDType} {sh : TileShape}
    (c : Tile .bool sh) (a b : Tile d sh) :
    A.trTile (Tile.select c a b) = Tile.select c (A.trTile a) (A.trTile b) := by
  apply Tile.ext
  intro i
  show A.trCarrier d (if c.data i then a.data i else b.data i)
      = if c.data i then A.trCarrier d (a.data i) else A.trCarrier d (b.data i)
  exact apply_ite (A.trCarrier d) _ _ _

/-- `trTile` commutes with the final-axis join. -/
theorem trTile_join {d : TileDType} {sh : TileShape} (a b : Tile d sh) :
    A.trTile (Tile.join a b) = Tile.join (A.trTile a) (A.trTile b) := by
  apply Tile.ext
  intro i
  show A.trCarrier d (if (TileShape.splitLastIndex sh i).2.val = 0
        then a.data (TileShape.splitLastIndex sh i).1
        else b.data (TileShape.splitLastIndex sh i).1) = _
  exact apply_ite (A.trCarrier d) _ _ _

/-- `trTile` commutes with the batched transpose. -/
theorem trTile_transpose {d : TileDType} (batch : TileShape) {M N : Nat}
    (x : Tile d (batch ++ [M, N])) :
    A.trTile (Tile.transpose batch x) = Tile.transpose batch (A.trTile x) := by
  induction batch with
  | nil => rfl
  | cons b rest ih =>
      apply Tile.ext
      rintro ⟨i, restIdx⟩
      have h := congrArg (fun t => t.data restIdx)
        (ih ⟨fun rIdx => x.data (i, rIdx)⟩)
      simpa [trTile, Tile.transpose_cons_data] using h

/-- `trTile` commutes with `Tile.ptrAdd`: relocating the base then adding an
offset is adding then relocating (`Nat.add_assoc`). -/
theorem trTile_ptrAdd {a b out : TileShape} (bc : Broadcast a b out)
    (p : Tile .ptr a) (o : Tile .nat b) :
    A.trTile (Tile.ptrAdd bc p o) = Tile.ptrAdd bc (A.trTile p) o := by
  apply Tile.ext
  intro i
  show (A.flat, A.addr (p.data (bc.leftIndex i)).1
        ((p.data (bc.leftIndex i)).2 + o.data (bc.rightIndex i))) = _
  simp [Tile.ptrAdd, trTile, trCarrier, addr, Nat.add_assoc]

/-- Relocation commutes with `BlockPtr.advance` (they touch disjoint
fields). -/
theorem trCarrier_advance (bp : BlockPtr) (ds : List Int) :
    A.trCarrier .blockPtr (bp.advance ds)
      = (A.trCarrier .blockPtr bp).advance ds := rfl

/-- On every data channel the carrier translation is the identity. -/
theorem trCarrier_data {d : TileDType} (h1 : d ≠ .ptr) (h2 : d ≠ .blockPtr)
    (v : TileCarrier d) : A.trCarrier d v = v := by
  cases d
  case ptr => exact absurd rfl h1
  case blockPtr => exact absurd rfl h2
  all_goals rfl

/-- The relocated block pointer lives in the flat region. -/
theorem trCarrier_blockPtr_region (bp : BlockPtr) :
    (A.trCarrier .blockPtr bp).region = A.flat := rfl

/-- Relocation shifts a block pointer's flat address by its region's base. -/
theorem trCarrier_blockPtr_address (bp : BlockPtr) (idx : List Nat) :
    (A.trCarrier .blockPtr bp).address idx
      = A.base bp.region + bp.address idx := by
  simp [trCarrier, BlockPtr.address, addr, Nat.add_assoc]

/-- Relocation does not change a block pointer's boundary check (it only
touches `region`/`baseOffset`). -/
theorem trCarrier_blockPtr_inBounds (bp : BlockPtr) (idx checkedAxes : List Nat) :
    (A.trCarrier .blockPtr bp).inBounds idx checkedAxes
      = bp.inBounds idx checkedAxes := rfl

end FlatAlloc

/-! ## dtype side facts -/

@[simp] theorem Region.cast_self {d : TileDType} (r : Region d) :
    (Region.cast r : Region d) = r := rfl


theorem NumericDType.ne_ptr {d : TileDType} (nd : NumericDType d) :
    d ≠ .ptr := by cases nd <;> simp
theorem NumericDType.ne_blockPtr {d : TileDType} (nd : NumericDType d) :
    d ≠ .blockPtr := by cases nd <;> simp
theorem ComparableDType.ne_ptr {d : TileDType} (cd : ComparableDType d) :
    d ≠ .ptr := by cases cd <;> simp
theorem ComparableDType.ne_blockPtr {d : TileDType} (cd : ComparableDType d) :
    d ≠ .blockPtr := by cases cd <;> simp
theorem IntegralDType.ne_ptr {d : TileDType} (id' : IntegralDType d) :
    d ≠ .ptr := by cases id' <;> simp
theorem IntegralDType.ne_blockPtr {d : TileDType} (id' : IntegralDType d) :
    d ≠ .blockPtr := by cases id' <;> simp
theorem FloatDType.toTileDType_ne_ptr (fd : FloatDType) :
    fd.toTileDType ≠ .ptr := by cases fd <;> simp [FloatDType.toTileDType]
theorem FloatDType.toTileDType_ne_blockPtr (fd : FloatDType) :
    fd.toTileDType ≠ .blockPtr := by cases fd <;> simp [FloatDType.toTileDType]

/-! ## Address-shift evaluation -/

/-- Evaluating the translated offset (`base + off`, as built by
`flattenAccess`) shifts every lane of the source offsets by `b`. -/
theorem evalOp_shiftAdd {sh : TileShape} (b : Nat) (off : Op .nat sh)
    (s : BlockState) :
    evalOp (.add .nat (FlatAlloc.shiftBroadcast sh) (.constNat b) off) s
      = (evalOp off s).map fun t => ⟨fun i => b + t.data i⟩ := by
  cases hoff : evalOp off s with
  | none => cases sh <;> simp [hoff, FlatAlloc.shiftBroadcast]
  | some t =>
      cases sh with
      | nil =>
          simp [hoff, FlatAlloc.shiftBroadcast, Tile.bop, NumericDType.nat_add]
          rfl
      | cons n r =>
          simp [hoff, FlatAlloc.shiftBroadcast, Tile.bop, NumericDType.nat_add]


/-! ## Per-state safety (bridge v1.2)

`Op.MemorySafe` quantifies its active-address bounds over **all** states,
which is dischargeable only when masks and addresses are visibly coupled
(inline addressing, block pointers) — not for the register-indirect
`offs := ...; tl.load(x + offs, ...)` style every DSL kernel uses. The
bridge therefore takes the *per-execution* contracts below: `Op.SafeAt`
bounds the active addresses in ONE state, and `Stmt.TraceSafe` threads it
along the actual execution (each step safe in the state it actually runs
in). `∀`-state `MemorySafe` users can always specialize. -/

set_option maxHeartbeats 1600000 in
/-- Single-state analogue of `Op.MemorySafe`: every load's active lanes are
in bounds **in the given state** (subterms recurse at the same state, since
expression evaluation never changes it). -/
def Op.SafeAt (bounds : RegionBounds) (s : BlockState) : Op dtype shape → Prop
  | .const _ => True
  | .constFloat _ _ => True
  | .constNat _ => True
  | .constInt _ => True
  | .constBool _ => True
  | .negInf => True
  | .programId _ => True
  | .numPrograms _ => True
  | .ref _ _ _ => True
  | .arange _ => True
  | .broadcast e _ => e.SafeAt bounds s
  | .full _ e => e.SafeAt bounds s
  | .castFloat _ _ e => e.SafeAt bounds s
  | .castNatToInt e => e.SafeAt bounds s
  | .castIntToNat e => e.SafeAt bounds s
  | .castRealToInt8 e => e.SafeAt bounds s
  | .add _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .sub _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .mul _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .div _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .floorDiv _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .mod _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .bitAnd _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .bitOr _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .bitXor _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .shiftLeft _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .shiftRight _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .exp a => a.SafeAt bounds s
  | .exp2 a => a.SafeAt bounds s
  | .log a => a.SafeAt bounds s
  | .log2 a => a.SafeAt bounds s
  | .sigmoid a => a.SafeAt bounds s
  | .sqrt a => a.SafeAt bounds s
  | .rsqrt a => a.SafeAt bounds s
  | .tanh a => a.SafeAt bounds s
  | .sin a => a.SafeAt bounds s
  | .cos a => a.SafeAt bounds s
  | .tan a => a.SafeAt bounds s
  | .atan a => a.SafeAt bounds s
  | .cosh a => a.SafeAt bounds s
  | .sinh a => a.SafeAt bounds s
  | .erf a => a.SafeAt bounds s
  | .lt _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .le _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .eq _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .gt _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .ge _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .ne _ _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .boolAnd _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .boolOr _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .boolNot a => a.SafeAt bounds s
  | .max2 _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .pow _ a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .where c a b => c.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .ite c a b => c.SafeAt bounds s ∧ a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .reduceMax _ _ a => a.SafeAt bounds s
  | .reduceMaxNat _ _ a => a.SafeAt bounds s
  | .reduceSum _ _ a => a.SafeAt bounds s
  | .scan _ _ _ a => a.SafeAt bounds s
  | .argMax _ a => a.SafeAt bounds s
  | .argMin _ a => a.SafeAt bounds s
  | .sort _ a => a.SafeAt bounds s
  | .dot a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .transpose a => a.SafeAt bounds s
  | .reshape _ a => a.SafeAt bounds s
  | .remap _ _ a => a.SafeAt bounds s
  | .join a b => a.SafeAt bounds s ∧ b.SafeAt bounds s
  | .split _ a => a.SafeAt bounds s
  | .expandDim _ a => a.SafeAt bounds s
  | .ptrBase _ => True
  | .ptrAdd _ ptr off => ptr.SafeAt bounds s ∧ off.SafeAt bounds s
  | .ptrSub _ ptr off => ptr.SafeAt bounds s ∧ off.SafeAt bounds s
  | .makeBlockPtr _ _ _ _ _ _ => True
  | .makeBlockPtrDyn _ base _ _ _ _ => base.SafeAt bounds s
  | .makeBlockPtrDynOffsets _ base _ _ _ offsets =>
      base.SafeAt bounds s ∧ ∀ off ∈ offsets, off.SafeAt bounds s
  | .advanceBlockPtr ptr _ => ptr.SafeAt bounds s
  | .load _ mem mask =>
      (match mem with
        | .region _ off => off.SafeAt bounds s
        | .ptr ptr => ptr.SafeAt bounds s
        | .blockPtr ptr _ => ptr.SafeAt bounds s) ∧
      (match mask with
        | .none => True
        | .mask m => m.SafeAt bounds s
        | .maskOther m other => m.SafeAt bounds s ∧ other.SafeAt bounds s) ∧
      mem.ActiveAddressSafe bounds s (mask.Active s)
  | .natToReal a => a.SafeAt bounds s


/-- `mapM`-level commutation for dynamic block-pointer offset lists: if every
member op evaluates identically after translation, so does the scalar-offset
`mapM`. -/
private theorem mapM_data_flatten (A : FlatAlloc) (s' s : BlockState) :
    ∀ (l : List (Op TileDType.nat [])),
      (∀ off ∈ l, evalOp (A.flattenOp off) s' = evalOp off s) →
      (l.map A.flattenOp).mapM (fun off => do
          let v ← evalOp off s'
          some (v.data PUnit.unit))
        = l.mapM (fun off => do
            let v ← evalOp off s
            some (v.data PUnit.unit))
  | [], _ => rfl
  | o :: rest, h => by
      simp only [List.map_cons, List.mapM_cons]
      rw [h o (List.mem_cons_self ..)]
      cases evalOp o s with
      | none => rfl
      | some v =>
          rw [mapM_data_flatten A s' s rest
            (fun off hoff => h off (List.mem_cons_of_mem _ hoff))]

set_option maxHeartbeats 3200000 in
/-- **Op-level bridge**: on the `FlattenOk` fragment, evaluating the
translated op in the flattened state returns the translated value. The
in-bounds obligations come from the #48 `Op.MemorySafe` contract at the
allocation's extents; `hcov` turns "active address below extent" into
membership in the allocation's region list (regions outside it have extent
`0`, so no active access can touch them). -/
theorem FlatAlloc.evalOp_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) :
    ∀ {d : TileDType} {sh : TileShape} (e : Op d sh) (s : BlockState),
      e.SafeAt A.extent s → e.FlattenOk → s.undef = (fun _ _ => 0) →
      evalOp (A.flattenOp e) (A.flattenState s)
        = (evalOp e s).map A.trTile
  | _, _, .const c, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .constFloat fd c, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]
      rw [A.trTileFun_data fd.toTileDType_ne_ptr fd.toTileDType_ne_blockPtr]; rfl
  | _, _, .constNat n, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .constInt n, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .constBool b, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .negInf, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .programId ax, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .numPrograms ax, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .ref d sh n, s, _, _, _ => by
      simp only [flattenOp, evalOp]; rfl
  | _, _, .arange n, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .broadcast e sh, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov e s hms hok hu]
      cases evalOp e s <;> rfl
  | _, _, .full sh e, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov e s hms hok hu]
      cases evalOp e s <;> rfl
  | _, _, .castFloat src dst a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data src.toTileDType_ne_ptr src.toTileDType_ne_blockPtr,
        A.trTileFun_data dst.toTileDType_ne_ptr dst.toTileDType_ne_blockPtr,
        Option.map_id, id_eq]
  | _, _, .add nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .sub nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .mul nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .div nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .floorDiv nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .mod nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
  | _, _, .lt nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .le nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .eq nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .gt nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .ge nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .ne nd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data nd.ne_ptr nd.ne_blockPtr,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitAnd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitOr bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .bitXor bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .shiftLeft bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .shiftRight bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolAnd bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolOr bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .max2 bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .pow bc a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .dot a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .exp a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .exp2 a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .log a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .log2 a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sigmoid a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sqrt a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .rsqrt a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .tanh a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sin a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .cos a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .tan a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .atan a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .cosh a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sinh a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .erf a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .boolNot a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .natToReal a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castNatToInt a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        A.trTileFun_data (d := .int) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castIntToNat a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .int) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .castRealToInt8 a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .int) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .sort ax a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .scan op ax dir a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .argMax ax a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .argMin ax a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceMax ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceMaxNat ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .reduceSum ax kd a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu,
        A.trTileFun_data (d := .real) (by decide) (by decide),
        Option.map_id, id_eq]
  | _, _, .where c a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hmc, hma, hmb⟩ := hms
      obtain ⟨hkc, hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov c s hmc hkc hu,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide), Option.map_id, id_eq]
      cases evalOp c s <;> cases evalOp a s <;> cases evalOp b s <;>
        simp [A.trTile_select]
  | _, _, .ite c a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hmc, hma, hmb⟩ := hms
      obtain ⟨hkc, hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov c s hmc hkc hu,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu,
        A.trTileFun_data (d := .bool) (by decide) (by decide), Option.map_id, id_eq]
      cases hvc : evalOp c s with
      | none => rfl
      | some vc => by_cases hb : vc.data PUnit.unit = true <;> simp [hb]
  | _, _, .transpose a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu]
      cases evalOp a s <;> simp [A.trTile_transpose]
  | _, _, .reshape outSh a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨⟨h1, h2⟩, hka⟩ := hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hka hu,
        A.trTileFun_data h1 h2, Option.map_id, id_eq]
  | _, _, .remap outSh f a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu]
      cases evalOp a s <;> rfl
  | _, _, .join a b, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hma, hmb⟩ := hms
      obtain ⟨hka, hkb⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov a s hma hka hu,
        evalOp_flatten A hd hcov b s hmb hkb hu]
      cases evalOp a s <;> cases evalOp b s <;> simp [A.trTile_join]
  | _, _, .split side a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu]
      cases evalOp a s <;> rfl
  | _, _, .expandDim ax a, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov a s hms hok hu]
      cases evalOp a s <;> rfl
  | _, _, .ptrBase r, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]
      exact congrArg some (Tile.ext fun i => by
        show ((Region.cast A.flat : RegionName), 0 + A.base (Region.cast r))
            = (A.flat, A.addr (Region.cast r) 0)
        simp [FlatAlloc.addr])
  | _, _, .ptrAdd bc p o, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hmp, hmo⟩ := hms
      obtain ⟨hkp, hko⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov p s hmp hkp hu,
        evalOp_flatten A hd hcov o s hmo hko hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide), Option.map_id, id_eq]
      cases evalOp p s <;> cases evalOp o s <;> simp [A.trTile_ptrAdd]
  | _, _, .ptrSub bc p o, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hkp, hko, hnuf⟩ := hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov p s hms.1 hkp hu,
        evalOp_flatten A hd hcov o s hms.2 hko hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
      cases hps : evalOp p s with
      | none => rfl
      | some ps =>
          cases hos : evalOp o s with
          | none => rfl
          | some os =>
              refine congrArg some (Tile.ext fun i => ?_)
              have hle := hnuf s ps os hps hos i
              show (A.flat,
                  A.addr (ps.data (bc.leftIndex i)).1
                      (ps.data (bc.leftIndex i)).2
                    - os.data (bc.rightIndex i))
                = (A.flat, A.addr (ps.data (bc.leftIndex i)).1
                    ((ps.data (bc.leftIndex i)).2 - os.data (bc.rightIndex i)))
              have harith :
                  A.addr (ps.data (bc.leftIndex i)).1
                      (ps.data (bc.leftIndex i)).2
                    - os.data (bc.rightIndex i)
                  = A.addr (ps.data (bc.leftIndex i)).1
                      ((ps.data (bc.leftIndex i)).2
                        - os.data (bc.rightIndex i)) := by
                simp only [FlatAlloc.addr]
                exact Nat.add_sub_assoc hle _
              rw [harith]
  | _, _, .makeBlockPtr r bo ps bs strides offs, s, _, _, _ => by
      simp only [flattenOp, evalOp, Option.map_some]; rfl
  | _, _, .makeBlockPtrDyn r bo ps bs strides offs, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov bo s hms hok hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide), Option.map_id, id_eq]
      cases hbo : evalOp bo s with
      | none => rfl
      | some b =>
          simp [Tile.bop, NumericDType.nat_add, FlatAlloc.trTile,
            FlatAlloc.trCarrier, FlatAlloc.addr]
  | _, _, .makeBlockPtrDynOffsets r bo ps bs strides offs, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      obtain ⟨hkb, hkoffs⟩ := hok
      obtain ⟨hmb, hmoffs⟩ := hms
      have hmem : ∀ off ∈ offs,
          evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s :=
        fun off hoff => by
          rw [evalOp_flatten A hd hcov off s (hmoffs off hoff)
              (hkoffs off hoff) hu,
            A.trTileFun_data (d := .nat) (by decide) (by decide),
            Option.map_id, id_eq]
      have hattach : (offs.attach.map fun o => A.flattenOp o.1)
          = offs.map A.flattenOp := by
        simp
      simp only [flattenOp, evalOp,
        evalOp_flatten A hd hcov bo s hmb hkb hu,
        A.trTileFun_data (d := .nat) (by decide) (by decide),
        Option.map_id, id_eq]
      cases hbo : evalOp bo s with
      | none => rfl
      | some b =>
          rw [show (offs.attach.map fun o => A.flattenOp o.1)
              = offs.map A.flattenOp from hattach,
            mapM_data_flatten A (A.flattenState s) s offs hmem]
          cases hoffl : offs.mapM (fun off => do
              let v ← evalOp off s
              some (v.data PUnit.unit)) with
          | none => rfl
          | some offlist =>
              simp [Tile.bop, NumericDType.nat_add, FlatAlloc.trTile,
                FlatAlloc.trCarrier, FlatAlloc.addr]
  | _, _, .advanceBlockPtr p ds, s, hms, hok, hu => by
      simp only [Op.SafeAt] at hms
      simp only [Op.FlattenOk] at hok
      simp only [flattenOp, evalOp, evalOp_flatten A hd hcov p s hms hok hu]
      cases evalOp p s <;>
        simp [FlatAlloc.trTile, FlatAlloc.trCarrier_advance]
  | _, _, .load d mem mask, s, hms, hok, hu => by
      rw [FlatAlloc.flattenOp_load]
      cases mem with
      | region r off =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHoff : evalOp (A.flattenOp off) (A.flattenState s)
                  = evalOp off s := by
                rw [evalOp_flatten A hd hcov off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hoffs : evalOp off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    evalOp_shiftAdd, IHoff, hoffs]
              | some offs =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOp, evalOp_shiftAdd, IHoff, hoffs, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  have hbound : offs.data i < A.extent (Region.cast r) :=
                    haddr offs hoffs i trivial
                  have hreg : (Region.cast r : RegionName) ∈ A.regions := by
                    by_contra hnr
                    rw [hcov _ hnr] at hbound
                    exact absurd hbound (Nat.not_lt_zero _)
                  rw [show A.base (Region.cast r) + offs.data i
                      = A.addr (Region.cast r) (offs.data i) from rfl]
                  rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                    A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHoff : evalOp (A.flattenOp off) (A.flattenState s)
                  = evalOp off s := by
                rw [evalOp_flatten A hd hcov off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hoffs : evalOp off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    evalOp_shiftAdd, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, evalOp_shiftAdd, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, evalOp_shiftAdd, IHoff, hoffs, IHm, hm,
                        Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      by_cases hact : ms.data i = true
                      · have hbound : offs.data i < A.extent (Region.cast r) :=
                          haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hreg : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hbound
                          exact absurd hbound (Nat.not_lt_zero _)
                        simp only [hact, if_true]
                        rw [show A.base (Region.cast r) + offs.data i
                            = A.addr (Region.cast r) (offs.data i) from rfl]
                        rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                          A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHoff : evalOp (A.flattenOp off) (A.flattenState s)
                  = evalOp off s := by
                rw [evalOp_flatten A hd hcov off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOp (A.flattenOp other) (A.flattenState s)
                  = evalOp other s := by
                rw [evalOp_flatten A hd hcov other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hoffs : evalOp off s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    evalOp_shiftAdd, IHoff, hoffs]
              | some offs =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, evalOp_shiftAdd, IHoff, hoffs, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOp, evalOp_shiftAdd, IHoff, hoffs, IHm, hm,
                            IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOp, evalOp_shiftAdd,
                            IHoff, hoffs, IHm, hm, IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          by_cases hact : ms.data i = true
                          · have hbound : offs.data i
                                < A.extent (Region.cast r) :=
                              haddr offs hoffs i ⟨ms, hm, hact⟩
                            have hreg : (Region.cast r : RegionName)
                                ∈ A.regions := by
                              by_contra hnr
                              rw [hcov _ hnr] at hbound
                              exact absurd hbound (Nat.not_lt_zero _)
                            simp only [hact, if_true]
                            rw [show A.base (Region.cast r) + offs.data i
                                = A.addr (Region.cast r) (offs.data i) from rfl]
                            rw [A.flattenState_readMemValue hd s hreg hbound
                                d hdt,
                              A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]
      | ptr p =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOp, IHp, hps, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  have hbound : (ps.data i).2 < A.extent (ps.data i).1 :=
                    haddr.2 ps hps i trivial
                  have hreg : (ps.data i).1 ∈ A.regions := by
                    by_contra hnr
                    rw [hcov _ hnr] at hbound
                    exact absurd hbound (Nat.not_lt_zero _)
                  show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                  rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                    A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm, Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      by_cases hact : ms.data i = true
                      · have hbound : (ps.data i).2 < A.extent (ps.data i).1 :=
                          haddr.2 ps hps i ⟨ms, hm, hact⟩
                        have hreg : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hbound
                          exact absurd hbound (Nat.not_lt_zero _)
                        simp only [hact, if_true]
                        show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                        rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                          A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOp (A.flattenOp other) (A.flattenState s)
                  = evalOp other s := by
                rw [evalOp_flatten A hd hcov other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOp, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOp, IHp, hps, IHm, hm,
                            IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          by_cases hact : ms.data i = true
                          · have hbound : (ps.data i).2
                                < A.extent (ps.data i).1 :=
                              haddr.2 ps hps i ⟨ms, hm, hact⟩
                            have hreg : (ps.data i).1 ∈ A.regions := by
                              by_contra hnr
                              rw [hcov _ hnr] at hbound
                              exact absurd hbound (Nat.not_lt_zero _)
                            simp only [hact, if_true]
                            show (A.flattenState s).readMemValue d A.flat (A.addr (ps.data i).1 (ps.data i).2) = A.trCarrier d (s.readMemValue d (ps.data i).1 (ps.data i).2)
                            rw [A.flattenState_readMemValue hd s hreg hbound
                                d hdt,
                              A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]
      | blockPtr p bc =>
          cases mask with
          | none =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                    evalOp, IHp, hps, Option.map_some]
                  refine congrArg some (Tile.ext fun i => ?_)
                  simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                  rw [A.trCarrier_blockPtr_region, A.trCarrier_blockPtr_address,
                    A.trCarrier_blockPtr_inBounds]
                  by_cases hib : (ps.data i).inBounds
                      (TileShape.indexToList _ i) bc = true
                  · have hbound := haddr.2 ps hps i trivial hib
                    have hreg : (ps.data i).region ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hbound
                      exact absurd hbound (Nat.not_lt_zero _)
                    simp only [hib, if_true]
                    rw [show A.base (ps.data i).region
                          + (ps.data i).address (TileShape.indexToList _ i)
                        = A.addr (ps.data i).region
                          ((ps.data i).address (TileShape.indexToList _ i))
                        from rfl]
                    rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                      A.trCarrier_data hdt.1 hdt.2]
                  · simp only [Bool.not_eq_true] at hib
                    simp only [hib, Bool.false_eq_true, if_false]
                    rw [A.trCarrier_data hdt.1 hdt.2]
          | mask m =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hokmask⟩ := hok
              obtain ⟨hmsmem, hmsmask, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm, Option.map_some]
                      refine congrArg some (Tile.ext fun i => ?_)
                      simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                      rw [A.trCarrier_blockPtr_region,
                        A.trCarrier_blockPtr_address,
                        A.trCarrier_blockPtr_inBounds]
                      by_cases hact : ms.data i = true
                      · simp only [hact, if_true]
                        by_cases hib : (ps.data i).inBounds
                            (TileShape.indexToList _ i) bc = true
                        · have hbound := haddr.2 ps hps i ⟨ms, hm, hact⟩ hib
                          have hreg : (ps.data i).region ∈ A.regions := by
                            by_contra hnr
                            rw [hcov _ hnr] at hbound
                            exact absurd hbound (Nat.not_lt_zero _)
                          simp only [hib, if_true]
                          rw [show A.base (ps.data i).region
                                + (ps.data i).address (TileShape.indexToList _ i)
                              = A.addr (ps.data i).region
                                ((ps.data i).address (TileShape.indexToList _ i))
                              from rfl]
                          rw [A.flattenState_readMemValue hd s hreg hbound d hdt,
                            A.trCarrier_data hdt.1 hdt.2]
                        · simp only [Bool.not_eq_true] at hib
                          simp only [hib, Bool.false_eq_true, if_false]
                          rw [A.trCarrier_data hdt.1 hdt.2]
                      · simp only [Bool.not_eq_true] at hact
                        simp only [hact, Bool.false_eq_true, if_false]
                        cases d
                        all_goals first
                          | exact absurd rfl hdt.1
                          | exact absurd rfl hdt.2
                          | simp [hu, FlatAlloc.flattenState, FlatAlloc.trCarrier]
          | maskOther m other =>
              simp only [Op.SafeAt] at hms
              simp only [Op.FlattenOk] at hok
              obtain ⟨hdt, hokmem, hkm, hko⟩ := hok
              obtain ⟨hmsmem, ⟨hmm, hmo⟩, haddr⟩ := hms
              simp only [MemAccess.ActiveAddressSafe,
                memAccessActiveAddressSafe, Op.PointerAddressesSafeOn] at haddr
              have IHp := evalOp_flatten A hd hcov p s hmsmem hokmem hu
              have IHm : evalOp (A.flattenOp m) (A.flattenState s)
                  = evalOp m s := by
                rw [evalOp_flatten A hd hcov m s hmm hkm hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]
              have IHo : evalOp (A.flattenOp other) (A.flattenState s)
                  = evalOp other s := by
                rw [evalOp_flatten A hd hcov other s hmo hko hu,
                  A.trTileFun_data hdt.1 hdt.2, Option.map_id, id_eq]
              cases hps : evalOp p s with
              | none =>
                  simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask, evalOp,
                    IHp, hps]
              | some ps =>
                  cases hm : evalOp m s with
                  | none =>
                      simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                        evalOp, IHp, hps, IHm, hm]
                  | some ms =>
                      cases ho : evalOp other s with
                      | none =>
                          simp [FlatAlloc.flattenAccess, FlatAlloc.flattenMask,
                            evalOp, IHp, hps, IHm, hm, IHo, ho]
                      | some os =>
                          simp only [FlatAlloc.flattenAccess,
                            FlatAlloc.flattenMask, evalOp, IHp, hps, IHm, hm,
                            IHo, ho, Option.map_some]
                          refine congrArg some (Tile.ext fun i => ?_)
                          simp only [reduceIte, FlatAlloc.trTile, Region.cast_cast, Region.cast_self]
                          rw [A.trCarrier_blockPtr_region,
                            A.trCarrier_blockPtr_address,
                            A.trCarrier_blockPtr_inBounds]
                          by_cases hact : ms.data i = true
                          · simp only [hact, if_true]
                            by_cases hib : (ps.data i).inBounds
                                (TileShape.indexToList _ i) bc = true
                            · have hbound := haddr.2 ps hps i ⟨ms, hm, hact⟩ hib
                              have hreg : (ps.data i).region ∈ A.regions := by
                                by_contra hnr
                                rw [hcov _ hnr] at hbound
                                exact absurd hbound (Nat.not_lt_zero _)
                              simp only [hib, if_true]
                              rw [show A.base (ps.data i).region
                                    + (ps.data i).address
                                      (TileShape.indexToList _ i)
                                  = A.addr (ps.data i).region
                                    ((ps.data i).address
                                      (TileShape.indexToList _ i))
                                  from rfl]
                              rw [A.flattenState_readMemValue hd s hreg hbound
                                  d hdt,
                                A.trCarrier_data hdt.1 hdt.2]
                            · simp only [Bool.not_eq_true] at hib
                              simp only [hib, Bool.false_eq_true, if_false]
                              rw [A.trCarrier_data hdt.1 hdt.2]
                          · simp only [Bool.not_eq_true] at hact
                            simp only [hact, Bool.false_eq_true, if_false]
                            rw [A.trCarrier_data hdt.1 hdt.2]

/-! ## State-level commutation (B3a)

The write-side counterparts of the read lemmas above: flattening commutes
with `setReg` and with a single in-bounds `writeMemTyped`. -/

namespace FlatAlloc

variable (A : FlatAlloc)

/-- Soundness of `decode`: a successful decode names a listed region, the
offset is in bounds, and re-encoding recovers the flat offset. -/
theorem decode_sound {o' : Nat} {r : RegionName} {o : Nat}
    (h : A.decode o' = some (r, o)) :
    r ∈ A.regions ∧ o' = A.addr r o ∧ o < A.extent r := by
  unfold decode at h
  cases hfind : A.regions.find? fun r2 =>
      decide (A.base r2 ≤ o' ∧ o' < A.base r2 + A.extent r2) with
  | none => rw [hfind] at h; simp at h
  | some r2 =>
      rw [hfind] at h
      simp only [Option.map_some, Option.some_inj] at h
      have hmem := List.mem_of_find?_eq_some hfind
      have hpred := List.find?_some hfind
      simp only [decide_eq_true_eq] at hpred
      rw [Prod.mk.injEq] at h
      obtain ⟨hr2, ho2⟩ := h
      subst hr2
      subst ho2
      refine ⟨hmem, ?_, ?_⟩
      · show o' = A.base r2 + (o' - A.base r2)
        omega
      · omega

/-- Reading the flat image of a single-cell update: the updated flat offset
sees the translated new cell; every other offset reads as before. -/
theorem readFlat_update (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (cell : MemCell) (o' : Nat) :
    A.readFlat { s with mem := fun rr oo =>
        if rr = r ∧ oo = o then cell else s.mem rr oo } o'
      = if o' = A.addr r o then A.trCell cell else A.readFlat s o' := by
  unfold readFlat
  cases hdec : A.decode o' with
  | none =>
      have hne : o' ≠ A.addr r o := fun he => by
        rw [he, A.decode_addr hd hr ho] at hdec
        simp at hdec
      rw [if_neg hne]
  | some ro =>
      obtain ⟨r2, o2⟩ := ro
      obtain ⟨hr2, henc, ho2⟩ := A.decode_sound hdec
      by_cases he : o' = A.addr r o
      · rw [if_pos he]
        rw [he, A.decode_addr hd hr ho] at hdec
        have h2 := Option.some_inj.mp hdec
        rw [Prod.mk.injEq] at h2
        obtain ⟨rfl, rfl⟩ := h2
        simp
      · rw [if_neg he]
        have hcell : ¬(r2 = r ∧ o2 = o) := by
          rintro ⟨rfl, rfl⟩
          exact he henc
        simp [hcell]

/-- Flattening commutes with `setReg`. -/
theorem flattenState_setReg (s : BlockState) (n : RegName)
    (d : TileDType) (sh : TileShape) (t : Tile d sh) :
    A.flattenState (s.setReg n d sh t)
      = (A.flattenState s).setReg n d sh (A.trTile t) := by
  refine BlockState.ext (fun _ _ => rfl) (fun d' sh' n' => ?_) (fun _ => rfl)
    (fun _ _ => rfl) (fun _ => rfl)
  show A.trRegs ((s.setReg n d sh t).regs) d' sh' n' = _
  simp only [BlockState.setReg, trRegs]
  split
  · split
    · split
      · subst_vars; rfl
      · rfl
    · rfl
  · rfl

/-- Flattening commutes with a single in-bounds typed write: writing `(r, o)`
in the source is writing `(flat, base r + o)` (with the translated value) in
the image. -/
theorem flattenState_writeMemTyped (hd : A.Disjoint) (s : BlockState)
    {r : RegionName} (hr : r ∈ A.regions) {o : Nat} (ho : o < A.extent r)
    (d : TileDType) (v : TileCarrier d) :
    A.flattenState (s.writeMemTyped d r o v)
      = (A.flattenState s).writeMemTyped d A.flat (A.addr r o)
          (A.trCarrier d v) := by
  cases d <;>
  · refine BlockState.ext (fun r' o' => ?_) (fun _ _ _ => rfl) (fun _ => rfl)
      (fun _ _ => rfl) (fun _ => rfl)
    show (if r' = A.flat then A.readFlat _ o' else MemCell.real 0) = _
    simp only [BlockState.writeMemTyped, BlockState.writeMemAs]
    by_cases hrf : r' = A.flat
    · subst hrf
      rw [if_pos rfl, A.readFlat_update hd s hr ho]
      by_cases heq : o' = A.addr r o
      · subst heq
        rw [if_pos rfl, if_pos ⟨rfl, rfl⟩]
        rfl
      · rw [if_neg heq, if_neg (fun h => heq h.2)]
        show A.readFlat s o'
            = if A.flat = A.flat then A.readFlat s o' else MemCell.real 0
        rw [if_pos rfl]
    · rw [if_neg hrf, if_neg (fun h => hrf h.1)]
      show MemCell.real 0
          = if r' = A.flat then A.readFlat s o' else MemCell.real 0
      rw [if_neg hrf]

end FlatAlloc




/-- Address-form children are safe at `s`. -/
def MemAccess.SafeAt (bounds : RegionBounds) (s : BlockState) :
    {d : TileDType} → {sh : TileShape} → MemAccess d sh → Prop
  | _, _, .region _ off => off.SafeAt bounds s
  | _, _, .ptr p => p.SafeAt bounds s
  | _, _, .blockPtr p _ => p.SafeAt bounds s

/-- Mask children are safe at `s`. -/
def MaskOpt.SafeAt (bounds : RegionBounds) (s : BlockState) :
    {d : TileDType} → {sh : TileShape} → MaskOpt d sh → Prop
  | _, _, .none => True
  | _, _, .mask m => m.SafeAt bounds s
  | _, _, .maskOther m o => m.SafeAt bounds s ∧ o.SafeAt bounds s

/-! ## Per-execution statement safety (bridge v1.2, continued) -/

mutual

/-- Per-execution safety: the statement's own accesses are in bounds in the
state it runs in, and every successor state reached by the actual execution
is recursively safe. A failing step has no successors to constrain. -/
def Stmt.TraceSafe (bounds : RegionBounds) : Stmt → BlockState → Prop
  | .assign _ _ _ e, s => e.SafeAt bounds s
  | .store _ _ mem val mask, s =>
      mem.SafeAt bounds s ∧ val.SafeAt bounds s ∧ mask.SafeAt bounds s ∧
      mem.ActiveAddressSafe bounds s (mask.Active s)
  | .atomicAdd _ _ mem val mask, s =>
      mem.SafeAt bounds s ∧ val.SafeAt bounds s ∧ mask.SafeAt bounds s ∧
      mem.ActiveAddressSafe bounds s (mask.Active s)
  | .atomicRMW _ _ _ _ _ _ _ _, _ => False
  | .forLoop idx n body, s => Stmt.forLoopTraceSafe bounds idx 0 n body s
  | .forRange idx start stop step body, s =>
      Stmt.forRangeTraceSafe bounds idx start stop step body s
  | .forRangeDyn idx start stop step body, s =>
      start.SafeAt bounds s ∧ stop.SafeAt bounds s ∧ step.SafeAt bounds s ∧
      (match evalOp start s, evalOp stop s, evalOp step s with
        | some a, some b, some c =>
            Stmt.forRangeTraceSafe bounds idx (a.data PUnit.unit)
              (b.data PUnit.unit) (c.data PUnit.unit) body s
        | _, _, _ => True)
  | .ifThen c body, s =>
      c.SafeAt bounds s ∧
      (match evalOp c s with
        | some vc => if vc.data PUnit.unit then Stmt.TraceSafeList bounds body s
                     else True
        | none => True)
  | .ifThenElse c tb eb, s =>
      c.SafeAt bounds s ∧
      (match evalOp c s with
        | some vc => if vc.data PUnit.unit then Stmt.TraceSafeList bounds tb s
                     else Stmt.TraceSafeList bounds eb s
        | none => True)
  termination_by st _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- Per-execution safety for a statement list. -/
def Stmt.TraceSafeList (bounds : RegionBounds) : List Stmt → BlockState → Prop
  | [], _ => True
  | st :: rest, s =>
      st.TraceSafe bounds s ∧
      (match stepStmt st s with
        | some s' => Stmt.TraceSafeList bounds rest s'
        | none => True)
  termination_by l _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- Per-execution safety through a `forLoop` unrolling. -/
def Stmt.forLoopTraceSafe (bounds : RegionBounds) (idx : RegName)
    (start n : Nat) (body : List Stmt) (s : BlockState) : Prop :=
  if start < n then
    Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar start)) ∧
    (match stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
      | some s' => Stmt.forLoopTraceSafe bounds idx (start + 1) n body s'
      | none => True)
  else True
  termination_by (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- Per-execution safety through a `forRange` unrolling. -/
def Stmt.forRangeTraceSafe (bounds : RegionBounds) (idx : RegName)
    (cur stop step : Nat) (body : List Stmt) (s : BlockState) : Prop :=
  if step = 0 then True
  else if cur < stop then
    Stmt.TraceSafeList bounds body (s.setReg idx .nat [] (Tile.scalar cur)) ∧
    (match stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
      | some s' =>
          Stmt.forRangeTraceSafe bounds idx (cur + step) stop step body s'
      | none => True)
  else True
  termination_by (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end

/-- Kernel-level per-execution safety. -/
def Kernel.TraceSafe (bounds : RegionBounds) (k : Kernel)
    (s : BlockState) : Prop :=
  Stmt.TraceSafeList bounds k.body s

/-! ## The flattenable statement fragment (B3b)

`store`/`assign`/control flow are covered. The two atomics are deferred:
`atomicAdd` is mechanical to readmit (its `NumericDType` constraint makes the
value translation the identity; only the address shifts), `atomicRMW`
additionally threads an event trace. -/

mutual

/-- Statements covered by the flat-memory bridge. -/
def Stmt.FlattenOk : Stmt → Prop
  | .assign _ _ _ e => e.FlattenOk
  | .store _ _ mem val mask =>
      (match mem with
        | .region _ off => off.FlattenOk
        | .ptr p => p.FlattenOk
        | .blockPtr p _ => p.FlattenOk) ∧
      val.FlattenOk ∧
      (match mask with
        | .none => True
        | .mask m => m.FlattenOk
        | .maskOther m o => m.FlattenOk ∧ o.FlattenOk)
  | .atomicAdd _ _ mem val mask =>
      (match mem with
        | .region _ off => off.FlattenOk
        | .ptr p => p.FlattenOk
        | .blockPtr p _ => p.FlattenOk) ∧
      val.FlattenOk ∧
      (match mask with
        | .none => True
        | .mask m => m.FlattenOk
        | .maskOther m o => m.FlattenOk ∧ o.FlattenOk)
  | .atomicRMW _ _ _ _ _ _ _ _ => False
  | .forLoop _ _ body => StmtList.FlattenOk body
  | .forRange _ _ _ _ body => StmtList.FlattenOk body
  | .forRangeDyn _ start stop step body =>
      start.FlattenOk ∧ stop.FlattenOk ∧ step.FlattenOk
        ∧ StmtList.FlattenOk body
  | .ifThen c body => c.FlattenOk ∧ StmtList.FlattenOk body
  | .ifThenElse c tb eb =>
      c.FlattenOk ∧ StmtList.FlattenOk tb ∧ StmtList.FlattenOk eb

/-- Statement lists covered by the flat-memory bridge. -/
def StmtList.FlattenOk : List Stmt → Prop
  | [] => True
  | st :: rest => st.FlattenOk ∧ StmtList.FlattenOk rest

end

/-- Kernels covered by the flat-memory bridge. -/
def Kernel.FlattenOk (k : Kernel) : Prop := StmtList.FlattenOk k.body

/-! ## Fold-level write commutation -/

namespace FlatAlloc

variable (A : FlatAlloc)

/-- A guarded `writeMemTyped` fold (the shape of every store) commutes with
flattening, provided every active write is in bounds. The per-lane
`reg`/`off`/`v`/`act` functions cover all three access forms at once. -/
theorem flattenState_foldl_store (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    {ι : Type _} {d : TileDType} (l : List ι) (s : BlockState)
    (act : ι → Bool) (reg : ι → RegionName) (off : ι → Nat)
    (v : ι → TileCarrier d)
    (hbound : ∀ i ∈ l, act i = true → off i < A.extent (reg i)) :
    A.flattenState (l.foldl
      (fun acc i => if act i
        then acc.writeMemTyped d (reg i) (off i) (v i) else acc) s)
      = l.foldl
        (fun acc i => if act i
          then acc.writeMemTyped d A.flat (A.addr (reg i) (off i))
            (A.trCarrier d (v i)) else acc) (A.flattenState s) := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · have hb := hbound j (List.mem_cons_self ..) hj
        have hrj : reg j ∈ A.regions := by
          by_contra hnr
          rw [hcov _ hnr] at hb
          exact absurd hb (Nat.not_lt_zero _)
        rw [if_pos hj, if_pos hj,
          ← A.flattenState_writeMemTyped hd s hrj hb d (v j)]
        exact ih _ fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact
      · rw [if_neg hj, if_neg hj]
        exact ih _ fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact

/-- Read-modify-write variant of `flattenState_foldl_store`: the written
value may depend on the running accumulator (the `atomicAdd` shape). The
`hv` hypothesis relates the two per-lane value functions across the
flattening at every intermediate state. -/
theorem flattenState_foldl_rmw (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0)
    {ι : Type _} {d : TileDType} (l : List ι) (s : BlockState)
    (act : ι → Bool) (reg : ι → RegionName) (off : ι → Nat)
    (v vf : BlockState → ι → TileCarrier d)
    (hbound : ∀ i ∈ l, act i = true → off i < A.extent (reg i))
    (hv : ∀ acc i, i ∈ l → act i = true →
      vf (A.flattenState acc) i = A.trCarrier d (v acc i)) :
    A.flattenState (l.foldl
      (fun acc i => if act i
        then acc.writeMemTyped d (reg i) (off i) (v acc i) else acc) s)
      = l.foldl
        (fun acc i => if act i
          then acc.writeMemTyped d A.flat (A.addr (reg i) (off i))
            (vf acc i) else acc) (A.flattenState s) := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · have hb := hbound j (List.mem_cons_self ..) hj
        have hrj : reg j ∈ A.regions := by
          by_contra hnr
          rw [hcov _ hnr] at hb
          exact absurd hb (Nat.not_lt_zero _)
        rw [if_pos hj, if_pos hj, hv s j (List.mem_cons_self ..) hj,
          ← A.flattenState_writeMemTyped hd s hrj hb d (v s j)]
        exact ih _ (fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact)
          (fun acc i hi hact => hv acc i (List.mem_cons_of_mem _ hi) hact)
      · rw [if_neg hj, if_neg hj]
        exact ih _ (fun i hi hact => hbound i (List.mem_cons_of_mem _ hi) hact)
          (fun acc i hi hact => hv acc i (List.mem_cons_of_mem _ hi) hact)

end FlatAlloc

/-! ## Execution preserves `undef` on the covered fragment -/

private theorem foldl_guarded_write_undef {ι : Type _} {d : TileDType}
    (l : List ι) (s : BlockState) (act : ι → Bool) (reg : ι → RegionName)
    (off : ι → Nat) (v : BlockState → ι → TileCarrier d) :
    (l.foldl (fun acc i => if act i
      then acc.writeMemTyped d (reg i) (off i) (v acc i) else acc) s).undef
      = s.undef := by
  induction l generalizing s with
  | nil => rfl
  | cons j rest ih =>
      simp only [List.foldl_cons]
      by_cases hj : act j = true
      · rw [if_pos hj, ih]
        funext rr oo
        exact BlockState.writeMemTyped_undef ..
      · rw [if_neg hj]
        exact ih _

mutual

/-- On the covered fragment, stepping a statement never changes `undef`. -/
theorem stepStmt_undef : ∀ (st : Stmt) (s s' : BlockState),
    st.FlattenOk → stepStmt st s = some s' → s'.undef = s.undef
  | .assign d sh n e, s, s', _, h => by
      simp only [stepStmt] at h
      cases hv : evalOp e s with
      | none => rw [hv] at h; exact absurd h (by simp)
      | some v =>
          rw [hv] at h
          replace h : some (s.setReg n d sh v) = some s' := h
          obtain rfl := Option.some_inj.mp h
          rfl
  | .store d sh mem val mask, s, s', _, h => by
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOp off s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOp p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hx : evalOp p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hv : evalOp val s with
        | none => rw [hv] at h; exact absurd h (by simp)
        | some values =>
            rw [hv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
  | .atomicAdd nd sh mem val mask, s, s', _, h => by
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOp off s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOp p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hx : evalOp p s with
            | none => rw [hx] at h; exact absurd h (by simp)
            | some xs =>
                rw [hx] at h
                replace h : some ((TileShape.allIndices sh).foldl _ s)
                    = some s' := h
                obtain rfl := Option.some_inj.mp h
                exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp off s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
      · simp only [stepStmt] at h
        cases hvv : evalOp val s with
        | none => rw [hvv] at h; exact absurd h (by simp)
        | some values =>
            rw [hvv] at h
            cases hm : evalOp m s with
            | none => rw [hm] at h; exact absurd h (by simp)
            | some ms =>
                rw [hm] at h
                cases hx : evalOp p s with
                | none => rw [hx] at h; exact absurd h (by simp)
                | some xs =>
                    rw [hx] at h
                    replace h : some ((TileShape.allIndices sh).foldl _ s)
                        = some s' := h
                    obtain rfl := Option.some_inj.mp h
                    exact foldl_guarded_write_undef ..
  | .atomicRMW op d sh mem input extra mask dest, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
  | .forLoop idx n body, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
      simp only [stepStmt] at h
      exact stepForLoopAux_undef idx 0 n body s s' hok h
  | .forRange idx start stop step body, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
      simp only [stepStmt] at h
      exact stepForRangeAux_undef idx start stop step body s s' hok h
  | .forRangeDyn idx start stop step body, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
      simp only [stepStmt] at h
      cases h1 : evalOp start s with
      | none => rw [h1] at h; exact absurd h (by simp)
      | some a =>
          rw [h1] at h
          cases h2 : evalOp stop s with
          | none => rw [h2] at h; exact absurd h (by simp)
          | some b =>
              rw [h2] at h
              cases h3 : evalOp step s with
              | none => rw [h3] at h; exact absurd h (by simp)
              | some c =>
                  rw [h3] at h
                  replace h : stepForRangeAux idx (a.data PUnit.unit)
                      (b.data PUnit.unit) (c.data PUnit.unit) body s
                      = some s' := h
                  exact stepForRangeAux_undef idx _ _ _ body s s' hok.2.2.2 h
  | .ifThen c body, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
      simp only [stepStmt] at h
      cases hc : evalOp c s with
      | none => rw [hc] at h; exact absurd h (by simp)
      | some vc =>
          rw [hc] at h
          replace h : (if vc.data PUnit.unit = true then stepStmts body s
              else some s) = some s' := h
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb] at h
            exact stepStmts_undef body s s' hok.2 h
          · rw [if_neg hb] at h
            obtain rfl := Option.some_inj.mp h
            rfl
  | .ifThenElse c tb eb, s, s', hok, h => by
      simp only [Stmt.FlattenOk] at hok
      simp only [stepStmt] at h
      cases hc : evalOp c s with
      | none => rw [hc] at h; exact absurd h (by simp)
      | some vc =>
          rw [hc] at h
          replace h : (if vc.data PUnit.unit = true then stepStmts tb s
              else stepStmts eb s) = some s' := h
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb] at h
            exact stepStmts_undef tb s s' hok.2.1 h
          · rw [if_neg hb] at h
            exact stepStmts_undef eb s s' hok.2.2 h
  termination_by st _ _ _ _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- List version of `stepStmt_undef`. -/
theorem stepStmts_undef : ∀ (l : List Stmt) (s s' : BlockState),
    StmtList.FlattenOk l → stepStmts l s = some s' → s'.undef = s.undef
  | [], s, s', _, h => by
      simp only [stepStmts] at h
      obtain rfl := Option.some_inj.mp h
      rfl
  | st :: rest, s, s', hok, h => by
      simp only [StmtList.FlattenOk] at hok
      simp only [stepStmts] at h
      cases h1 : stepStmt st s with
      | none => rw [h1] at h; exact absurd h (by simp)
      | some s1 =>
          rw [h1] at h
          replace h : stepStmts rest s1 = some s' := h
          rw [stepStmts_undef rest s1 s' hok.2 h,
            stepStmt_undef st s s1 hok.1 h1]
  termination_by l _ _ _ _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary version of `stepStmt_undef`. -/
theorem stepForLoopAux_undef : ∀ (idx : RegName) (start n : Nat)
    (body : List Stmt) (s s' : BlockState),
    StmtList.FlattenOk body →
    stepForLoopAux idx start n body s = some s' → s'.undef = s.undef
  | idx, start, n, body, s, s', hok, h => by
      rw [stepForLoopAux] at h
      split at h
      · cases h1 : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rw [h1] at h; exact absurd h (by simp)
        | some s1 =>
            rw [h1] at h
            replace h : stepForLoopAux idx (start + 1) n body s1 = some s' := h
            rw [stepForLoopAux_undef idx (start + 1) n body s1 s' hok h,
              stepStmts_undef body _ s1 hok h1]
            rfl
      · obtain rfl := Option.some_inj.mp h
        rfl
  termination_by _ start n body _ _ _ _ => (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- `forRange` auxiliary version of `stepStmt_undef`. -/
theorem stepForRangeAux_undef : ∀ (idx : RegName) (cur stop step : Nat)
    (body : List Stmt) (s s' : BlockState),
    StmtList.FlattenOk body →
    stepForRangeAux idx cur stop step body s = some s' → s'.undef = s.undef
  | idx, cur, stop, step, body, s, s', hok, h => by
      rw [stepForRangeAux] at h
      split at h
      · obtain rfl := Option.some_inj.mp h
        rfl
      · split at h
        · cases h1 : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rw [h1] at h; exact absurd h (by simp)
          | some s1 =>
              rw [h1] at h
              replace h : stepForRangeAux idx (cur + step) stop step body s1
                  = some s' := h
              rw [stepForRangeAux_undef idx (cur + step) stop step body s1 s'
                  hok h,
                stepStmts_undef body _ s1 hok h1]
              rfl
        · obtain rfl := Option.some_inj.mp h
          rfl
  termination_by _ cur stop step body _ _ _ _ => (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end

/-! ## Statement-level bridge (B3c) -/

mutual

/-- **Statement-level bridge**: on the covered fragment, stepping the
translated statement in the flattened state is the flattening of the source
step. -/
theorem FlatAlloc.stepStmt_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) :
    ∀ (st : Stmt) (s : BlockState),
      st.TraceSafe A.extent s → st.FlattenOk → s.undef = (fun _ _ => 0) →
      stepStmt (A.flattenStmt st) (A.flattenState s)
        = (stepStmt st s).map A.flattenState
  | .assign d sh n e, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      rw [A.evalOp_flatten hd hcov e s hms hok hu]
      cases hv : evalOp e s with
      | none => rfl
      | some v =>
          refine congrArg some ?_
          exact (A.flattenState_setReg s n d sh v).symm
  | .store d sh mem val mask, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt]
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩ <;>
        simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask]
      -- none.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [evalOp_shiftAdd,
              show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hoffs : evalOp off s with
            | none => rfl
            | some offs =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                  (fun i _ _ => haddr offs hoffs i trivial)]
                rfl
      -- none.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
            cases hps : evalOp p s with
            | none => rfl
            | some ps =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                  (fun i _ _ => haddr.2 ps hps i trivial)]
                rfl
      -- none.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
            cases hps : evalOp p s with
            | none => rfl
            | some ps =>
                refine congrArg some ?_
                rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                  (fun i _ hact => haddr.2 ps hps i trivial (by simpa using hact))]
                congr 1
                funext acc i
                simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                  FlatAlloc.trCarrier_blockPtr_address,
                  FlatAlloc.trCarrier_blockPtr_inBounds]
                rfl
      -- mask.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [evalOp_shiftAdd,
                  show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                    rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOp off s with
                | none => rfl
                | some offs =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => haddr.2 ps hps i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2)]
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile,
                      FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
      -- mask.region
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [evalOp_shiftAdd,
                  show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                    rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOp off s with
                | none => rfl
                | some offs =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.ptr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => haddr.2 ps hps i ⟨ms, hm, hact⟩)]
                    rfl
      -- mask.blockPtr
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu]
        cases hv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    refine congrArg some ?_
                    rw [A.flattenState_foldl_store hd hcov _ s _ _ _ _
                      (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2)]
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile,
                      FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
  | .atomicAdd nd sh mem val mask, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt]
      rcases mask with _ | m | ⟨m, o⟩ <;> rcases mem with ⟨r, off⟩ | p | ⟨p, bc⟩ <;>
        simp only [FlatAlloc.flattenAccess, FlatAlloc.flattenMask]
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [evalOp_shiftAdd,
              show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                  A.trTileFun_data (d := .nat) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hoffs : evalOp off s with
            | none => rfl
            | some offs =>
                have hfold := A.flattenState_foldl_rmw hd hcov
                  (TileShape.allIndices sh) s
                  (act := fun _ => true)
                  (reg := fun _ => (Region.cast r : RegionName))
                  (off := fun i => offs.data i)
                  (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                  (fun i _ _ => haddr offs hoffs i trivial)
                  (fun acc i _ _ => by
                    simp only []
                    have hb2 := haddr offs hoffs i trivial
                    have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
            cases hps : evalOp p s with
            | none => rfl
            | some ps =>
                have hfold := A.flattenState_foldl_rmw hd hcov
                  (TileShape.allIndices sh) s
                  (act := fun _ => true)
                  (reg := fun i => (ps.data i).1)
                  (off := fun i => (ps.data i).2)
                  (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                  (fun i _ _ => haddr.2 ps hps i trivial)
                  (fun acc i _ _ => by
                    simp only []
                    have hb2 := haddr.2 ps hps i trivial
                    have hr2 : (ps.data i).1 ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
            cases hps : evalOp p s with
            | none => rfl
            | some ps =>
                have hfold := A.flattenState_foldl_rmw hd hcov
                  (TileShape.allIndices sh) s
                  (act := fun i => true && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                  (reg := fun i => (ps.data i).region)
                  (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                  (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                  (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                  (fun i _ hact => by
                    simp only [Bool.and_eq_true] at hact
                    exact haddr.2 ps hps i trivial hact.2)
                  (fun acc i _ hact => by
                    simp only []
                    simp only [Bool.and_eq_true] at hact
                    have hb2 := haddr.2 ps hps i trivial hact.2
                    have hr2 : (ps.data i).region ∈ A.regions := by
                      by_contra hnr
                      rw [hcov _ hnr] at hb2
                      exact absurd hb2 (Nat.not_lt_zero _)
                    rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                        ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                      A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                refine congrArg some ?_
                refine Eq.trans ?_ hfold.symm
                congr 1
                funext acc i
                simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                  FlatAlloc.trCarrier_blockPtr_address,
                  FlatAlloc.trCarrier_blockPtr_inBounds]
                rfl
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [evalOp_shiftAdd,
                  show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                    rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOp off s with
                | none => rfl
                | some offs =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun _ => (Region.cast r : RegionName))
                      (off := fun i => offs.data i)
                      (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun i => (ps.data i).1)
                      (off := fun i => (ps.data i).2)
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                      (fun i _ hact => haddr.2 ps hps i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr.2 ps hps i ⟨ms, hm, hact⟩
                        have hr2 : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask hokmask hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      (reg := fun i => (ps.data i).region)
                      (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2)
                      (fun acc i _ hact => by
                        simp only []
                        simp only [Bool.and_eq_true] at hact
                        have hb2 := haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2
                        have hr2 : (ps.data i).region ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    refine Eq.trans ?_ hfold.symm
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [evalOp_shiftAdd,
                  show evalOp (A.flattenOp off) (A.flattenState s) = evalOp off s from by
                    rw [A.evalOp_flatten hd hcov off s hmsmem hokmem hu,
                      A.trTileFun_data (d := .nat) (by decide) (by decide),
                      Option.map_id, id_eq]]
                cases hoffs : evalOp off s with
                | none => rfl
                | some offs =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun _ => (Region.cast r : RegionName))
                      (off := fun i => offs.data i)
                      (v := fun acc i => nd.add (acc.readMemValue _ (Region.cast r) (offs.data i)) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (Region.cast r) (offs.data i))) (values.data i))
                      (fun i _ hact => haddr offs hoffs i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr offs hoffs i ⟨ms, hm, hact⟩
                        have hr2 : (Region.cast r : RegionName) ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i)
                      (reg := fun i => (ps.data i).1)
                      (off := fun i => (ps.data i).2)
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).1 (ps.data i).2) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).1 (ps.data i).2)) (values.data i))
                      (fun i _ hact => haddr.2 ps hps i ⟨ms, hm, hact⟩)
                      (fun acc i _ hact => by
                        simp only []
                        have hb2 := haddr.2 ps hps i ⟨ms, hm, hact⟩
                        have hr2 : (ps.data i).1 ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    exact hfold.symm
      · obtain ⟨hmsmem, hmsval, hmsmask, haddr⟩ := hms
        obtain ⟨hokmem, hokval, hokmask⟩ := hok
        simp only [MemAccess.ActiveAddressSafe, memAccessActiveAddressSafe,
          Op.PointerAddressesSafeOn] at haddr
        simp only [stepStmt]
        rw [A.evalOp_flatten hd hcov val s hmsval hokval hu,
          A.trTileFun_data nd.ne_ptr nd.ne_blockPtr, Option.map_id, id_eq]
        cases hvv : evalOp val s with
        | none => rfl
        | some values =>
            rw [show evalOp (A.flattenOp m) (A.flattenState s) = evalOp m s from by
                rw [A.evalOp_flatten hd hcov m s hmsmask.1 hokmask.1 hu,
                  A.trTileFun_data (d := .bool) (by decide) (by decide),
                  Option.map_id, id_eq]]
            cases hm : evalOp m s with
            | none => rfl
            | some ms =>
                rw [A.evalOp_flatten hd hcov p s hmsmem hokmem hu]
                cases hps : evalOp p s with
                | none => rfl
                | some ps =>
                    have hfold := A.flattenState_foldl_rmw hd hcov
                      (TileShape.allIndices sh) s
                      (act := fun i => ms.data i && (ps.data i).inBounds (TileShape.indexToList sh i) bc)
                      (reg := fun i => (ps.data i).region)
                      (off := fun i => (ps.data i).address (TileShape.indexToList sh i))
                      (v := fun acc i => nd.add (acc.readMemValue _ (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i))) (values.data i))
                      (vf := fun acc i => nd.add (acc.readMemValue _ A.flat (A.addr (ps.data i).region ((ps.data i).address (TileShape.indexToList sh i)))) (values.data i))
                      (fun i _ hact => by
                        simp only [Bool.and_eq_true] at hact
                        exact haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2)
                      (fun acc i _ hact => by
                        simp only []
                        simp only [Bool.and_eq_true] at hact
                        have hb2 := haddr.2 ps hps i ⟨ms, hm, hact.1⟩ hact.2
                        have hr2 : (ps.data i).region ∈ A.regions := by
                          by_contra hnr
                          rw [hcov _ hnr] at hb2
                          exact absurd hb2 (Nat.not_lt_zero _)
                        rw [A.flattenState_readMemValue hd acc hr2 hb2 _
                            ⟨nd.ne_ptr, nd.ne_blockPtr⟩,
                          A.trCarrier_data nd.ne_ptr nd.ne_blockPtr])
                    refine congrArg some ?_
                    refine Eq.trans ?_ hfold.symm
                    congr 1
                    funext acc i
                    simp only [FlatAlloc.trTile, FlatAlloc.trCarrier_blockPtr_region,
                      FlatAlloc.trCarrier_blockPtr_address,
                      FlatAlloc.trCarrier_blockPtr_inBounds]
                    rfl
  | .atomicRMW op d sh mem input extra mask dest, s, hms, hok, hu => by
      simp only [Stmt.FlattenOk] at hok
  | .forLoop idx n body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      exact A.stepForLoopAux_flatten hd hcov idx 0 n body s hms hok hu
  | .forRange idx start stop step body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      exact A.stepForRangeAux_flatten hd hcov idx start stop step body s hms hok hu
  | .forRangeDyn idx start stop step body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      obtain ⟨hms1, hms2, hms3, hmsb⟩ := hms
      obtain ⟨hok1, hok2, hok3, hokb⟩ := hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      rw [show evalOp (A.flattenOp start) (A.flattenState s) = evalOp start s from by
          rw [A.evalOp_flatten hd hcov start s hms1 hok1 hu,
            A.trTileFun_data (d := .nat) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases h1 : evalOp start s with
      | none => rfl
      | some a =>
          rw [show evalOp (A.flattenOp stop) (A.flattenState s) = evalOp stop s from by
              rw [A.evalOp_flatten hd hcov stop s hms2 hok2 hu,
                A.trTileFun_data (d := .nat) (by decide) (by decide),
                Option.map_id, id_eq]]
          cases h2 : evalOp stop s with
          | none => rfl
          | some b =>
              rw [show evalOp (A.flattenOp step) (A.flattenState s)
                  = evalOp step s from by
                  rw [A.evalOp_flatten hd hcov step s hms3 hok3 hu,
                    A.trTileFun_data (d := .nat) (by decide) (by decide),
                    Option.map_id, id_eq]]
              cases h3 : evalOp step s with
              | none => rfl
              | some c =>
                  rw [h1, h2, h3] at hmsb
                  replace hmsb : Stmt.forRangeTraceSafe A.extent idx
                      (a.data PUnit.unit) (b.data PUnit.unit)
                      (c.data PUnit.unit) body s := hmsb
                  exact A.stepForRangeAux_flatten hd hcov idx _ _ _ body s
                    hmsb hokb hu
  | .ifThen c body, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      rw [show evalOp (A.flattenOp c) (A.flattenState s) = evalOp c s from by
          rw [A.evalOp_flatten hd hcov c s hms.1 hok.1 hu,
            A.trTileFun_data (d := .bool) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases hc : evalOp c s with
      | none => rfl
      | some vc =>
          show (if vc.data PUnit.unit = true
              then stepStmts (A.flattenStmts body) (A.flattenState s)
              else some (A.flattenState s))
            = (if vc.data PUnit.unit = true then stepStmts body s
              else some s).map A.flattenState
          have hbody := hms.2
          rw [hc] at hbody
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            replace hbody : (if vc.data PUnit.unit = true
                then Stmt.TraceSafeList A.extent body s else True) := hbody
            rw [if_pos hb] at hbody
            exact A.stepStmts_flatten hd hcov body s hbody hok.2 hu
          · rw [if_neg hb, if_neg hb]
            rfl
  | .ifThenElse c tb eb, s, hms, hok, hu => by
      simp only [Stmt.TraceSafe] at hms
      simp only [Stmt.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmt, stepStmt]
      rw [show evalOp (A.flattenOp c) (A.flattenState s) = evalOp c s from by
          rw [A.evalOp_flatten hd hcov c s hms.1 hok.1 hu,
            A.trTileFun_data (d := .bool) (by decide) (by decide),
            Option.map_id, id_eq]]
      cases hc : evalOp c s with
      | none => rfl
      | some vc =>
          show (if vc.data PUnit.unit = true
              then stepStmts (A.flattenStmts tb) (A.flattenState s)
              else stepStmts (A.flattenStmts eb) (A.flattenState s))
            = (if vc.data PUnit.unit = true then stepStmts tb s
              else stepStmts eb s).map A.flattenState
          have hbody := hms.2
          rw [hc] at hbody
          replace hbody : (if vc.data PUnit.unit = true
              then Stmt.TraceSafeList A.extent tb s
              else Stmt.TraceSafeList A.extent eb s) := hbody
          by_cases hb : vc.data PUnit.unit = true
          · rw [if_pos hb, if_pos hb]
            rw [if_pos hb] at hbody
            exact A.stepStmts_flatten hd hcov tb s hbody hok.2.1 hu
          · rw [if_neg hb, if_neg hb]
            rw [if_neg hb] at hbody
            exact A.stepStmts_flatten hd hcov eb s hbody hok.2.2 hu
  termination_by st _ _ _ _ => (sizeOf st, 0)
  decreasing_by
    all_goals simp_wf
    all_goals (try omega)
    all_goals (have : 0 < sizeOf idx := by cases idx; simp)
    all_goals omega

/-- List version of the statement bridge. -/
theorem FlatAlloc.stepStmts_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) :
    ∀ (l : List Stmt) (s : BlockState),
      Stmt.TraceSafeList A.extent l s → StmtList.FlattenOk l →
      s.undef = (fun _ _ => 0) →
      stepStmts (A.flattenStmts l) (A.flattenState s)
        = (stepStmts l s).map A.flattenState
  | [], s, _, _, _ => by
      simp only [FlatAlloc.flattenStmts, stepStmts, Option.map_some]
  | st :: rest, s, hms, hok, hu => by
      simp only [Stmt.TraceSafeList] at hms
      simp only [StmtList.FlattenOk] at hok
      simp only [FlatAlloc.flattenStmts, stepStmts]
      rw [A.stepStmt_flatten hd hcov st s hms.1 hok.1 hu]
      cases h1 : stepStmt st s with
      | none => rfl
      | some s1 =>
          have hu1 : s1.undef = (fun _ _ => 0) := by
            rw [stepStmt_undef st s s1 hok.1 h1, hu]
          have hrest := hms.2
          rw [h1] at hrest
          replace hrest : Stmt.TraceSafeList A.extent rest s1 := hrest
          exact A.stepStmts_flatten hd hcov rest s1 hrest hok.2 hu1
  termination_by l _ _ _ _ => (sizeOf l, 0)
  decreasing_by all_goals (simp_wf; omega)

/-- `forLoop` auxiliary bridge. -/
theorem FlatAlloc.stepForLoopAux_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) :
    ∀ (idx : RegName) (start n : Nat) (body : List Stmt) (s : BlockState),
      Stmt.forLoopTraceSafe A.extent idx start n body s →
      StmtList.FlattenOk body → s.undef = (fun _ _ => 0) →
      stepForLoopAux idx start n (A.flattenStmts body) (A.flattenState s)
        = (stepForLoopAux idx start n body s).map A.flattenState
  | idx, start, n, body, s, hms, hok, hu => by
      rw [stepForLoopAux, stepForLoopAux]
      rw [Stmt.forLoopTraceSafe] at hms
      split
      next hlt =>
        rw [if_pos hlt] at hms
        obtain ⟨hbody, hnext⟩ := hms
        have hset : (A.flattenState s).setReg idx .nat [] (Tile.scalar start)
            = A.flattenState (s.setReg idx .nat [] (Tile.scalar start)) :=
          (A.flattenState_setReg s idx .nat [] (Tile.scalar start)).symm
        rw [hset, A.stepStmts_flatten hd hcov body
          (s.setReg idx .nat [] (Tile.scalar start)) hbody hok hu]
        cases h1 : stepStmts body (s.setReg idx .nat [] (Tile.scalar start)) with
        | none => rfl
        | some s1 =>
            have hu1 : s1.undef = (fun _ _ => 0) := by
              rw [stepStmts_undef body _ s1 hok h1]
              exact hu
            rw [h1] at hnext
            replace hnext : Stmt.forLoopTraceSafe A.extent idx (start + 1) n
                body s1 := hnext
            exact A.stepForLoopAux_flatten hd hcov idx (start + 1) n body s1
              hnext hok hu1
      next => rfl
  termination_by _ start n body _ _ _ _ => (sizeOf body + 1, n - start)
  decreasing_by all_goals (simp_wf; omega)

/-- `forRange` auxiliary bridge. -/
theorem FlatAlloc.stepForRangeAux_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) :
    ∀ (idx : RegName) (cur stop step : Nat) (body : List Stmt)
      (s : BlockState),
      Stmt.forRangeTraceSafe A.extent idx cur stop step body s →
      StmtList.FlattenOk body → s.undef = (fun _ _ => 0) →
      stepForRangeAux idx cur stop step (A.flattenStmts body)
          (A.flattenState s)
        = (stepForRangeAux idx cur stop step body s).map A.flattenState
  | idx, cur, stop, step, body, s, hms, hok, hu => by
      rw [stepForRangeAux, stepForRangeAux]
      rw [Stmt.forRangeTraceSafe] at hms
      split
      next => rfl
      next hz =>
        rw [if_neg hz] at hms
        split
        next hlt =>
          rw [if_pos hlt] at hms
          obtain ⟨hbody, hnext⟩ := hms
          have hset : (A.flattenState s).setReg idx .nat [] (Tile.scalar cur)
              = A.flattenState (s.setReg idx .nat [] (Tile.scalar cur)) :=
            (A.flattenState_setReg s idx .nat [] (Tile.scalar cur)).symm
          rw [hset, A.stepStmts_flatten hd hcov body
            (s.setReg idx .nat [] (Tile.scalar cur)) hbody hok hu]
          cases h1 : stepStmts body (s.setReg idx .nat [] (Tile.scalar cur)) with
          | none => rfl
          | some s1 =>
              have hu1 : s1.undef = (fun _ _ => 0) := by
                rw [stepStmts_undef body _ s1 hok h1]
                exact hu
              rw [h1] at hnext
              replace hnext : Stmt.forRangeTraceSafe A.extent idx (cur + step)
                  stop step body s1 := hnext
              exact A.stepForRangeAux_flatten hd hcov idx (cur + step) stop step
                body s1 hnext hok hu1
        next => rfl
  termination_by _ cur stop step body _ _ _ _ => (sizeOf body + 1, stop - cur)
  decreasing_by all_goals (simp_wf; omega)

end

/-- **The flat-memory bridge**: on the covered fragment, running the
translated kernel on the flattened state is the flattening of the source
run. Every region-model execution theorem transports along this equation to
a single flat address space. -/
theorem FlatAlloc.exec_flatten (A : FlatAlloc) (hd : A.Disjoint)
    (hcov : ∀ r, r ∉ A.regions → A.extent r = 0) (k : Kernel)
    (s : BlockState) (hms : k.TraceSafe A.extent s) (hok : k.FlattenOk)
    (hu : s.undef = (fun _ _ => 0)) :
    exec (A.flattenKernel k) (A.flattenState s)
      = (exec k s).map A.flattenState := by
  simp only [exec, FlatAlloc.flattenKernel]
  exact A.stepStmts_flatten hd hcov k.body s hms hok hu

end VeriTile.Triton
