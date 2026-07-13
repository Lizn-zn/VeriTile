/-
VeriTile.Triton.Memory.Denotation

The generic denotation combinator (#487): ⟦kernel⟧ as "array in, array out".

A denotation-style specification states a kernel's behavior with **no**
memory concept in the statement: plain arrays in, `Option ℝ` out. Every
pointer, region, layout, allocation, and start-state concept is packaged —
once, here — into `denoteKernel`, and each kernel's ⟦·⟧ becomes a one-line
instantiation supplying

* a **region table** (`List (RegionName × Nat)`): the named regions the
  kernel touches with their extents; `FlatAlloc.ofList` lays them end to
  end (prefix-sum bases) inside one flat address space,
* a **slot table** (`List DenoteSlot`): which region holds which input
  array at which window of offsets; `DenoteSlot.state` is the canonical
  start state loaded with exactly those arrays, and
* one output region/offset to read back after running the **flattened**
  kernel (real pointer arithmetic) on the flattened canonical state.

**Audit surface.** This module is the audited trust point of the denotation
program (#487): a reviewer audits `denoteKernel` — canonical allocation,
canonical loaded state, flattening translation, execution, read-back — once
and for all. Per kernel, the audit surface is only the **slot and region
tables** of the one-line instantiation (plus the output address); everything
else a denotation headline mentions is elementary mathematics.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.Memory.Flatten

namespace VeriTile.Triton

/-! ## The canonical allocation: a region table, laid end to end -/

namespace FlatAlloc

/-- Base offset of region `r` in the end-to-end layout of the region table
`regs`: the sum of the extents of the entries strictly before the first
entry named `r` (and the total size if `r` is absent — irrelevant, since
absent regions get extent `0`). -/
def listBase : List (RegionName × Nat) → RegionName → Nat
  | [], _ => 0
  | (r', e) :: rest, r => if r = r' then 0 else e + listBase rest r

/-- Extent of region `r` in the region table `regs`: the extent of the first
entry named `r`, and `0` if `r` is absent. -/
def listExtent : List (RegionName × Nat) → RegionName → Nat
  | [], _ => 0
  | (r', e) :: rest, r => if r = r' then e else listExtent rest r

/-- The canonical flat allocation of a region table: the listed regions laid
end to end (prefix-sum bases) inside the single flat region `flat`. This is
the allocation `denoteKernel` runs over; the region table is part of the
per-kernel audit surface. -/
def ofList (flat : RegionName) (regs : List (RegionName × Nat)) : FlatAlloc where
  flat := flat
  regions := regs.map Prod.fst
  base := listBase regs
  extent := listExtent regs

@[simp] theorem ofList_flat (flat : RegionName) (regs : List (RegionName × Nat)) :
    (ofList flat regs).flat = flat := rfl

@[simp] theorem ofList_regions (flat : RegionName) (regs : List (RegionName × Nat)) :
    (ofList flat regs).regions = regs.map Prod.fst := rfl

@[simp] theorem ofList_base (flat : RegionName) (regs : List (RegionName × Nat)) :
    (ofList flat regs).base = listBase regs := rfl

@[simp] theorem ofList_extent (flat : RegionName) (regs : List (RegionName × Nat)) :
    (ofList flat regs).extent = listExtent regs := rfl

/-- Listed entries are regions of the canonical allocation. -/
theorem ofList_mem_regions {flat : RegionName} {regs : List (RegionName × Nat)}
    {r : RegionName} {e : Nat} (hmem : (r, e) ∈ regs) :
    r ∈ (ofList flat regs).regions :=
  List.mem_map.mpr ⟨(r, e), hmem, rfl⟩

/-- Extent lookup: under a duplicate-free region table, a listed entry's
extent is its table entry. -/
theorem listExtent_of_mem {regs : List (RegionName × Nat)}
    (hnd : (regs.map Prod.fst).Nodup)
    {r : RegionName} {e : Nat} (hmem : (r, e) ∈ regs) :
    listExtent regs r = e := by
  induction regs with
  | nil => cases hmem
  | cons p tl ih =>
      obtain ⟨r0, e0⟩ := p
      rw [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨h0, hnd⟩ := hnd
      rcases List.mem_cons.mp hmem with heq | hmem'
      · rw [Prod.mk.injEq] at heq
        obtain ⟨rfl, rfl⟩ := heq
        simp [listExtent]
      · have hrtl : r ∈ tl.map Prod.fst := List.mem_map.mpr ⟨(r, e), hmem', rfl⟩
        have hne : r ≠ r0 := fun h => h0 (h ▸ hrtl)
        simp only [listExtent, if_neg hne]
        exact ih hnd hmem'

/-- Extent closure: regions outside the table have extent `0`. -/
theorem listExtent_eq_zero {regs : List (RegionName × Nat)}
    {r : RegionName} (hr : r ∉ regs.map Prod.fst) :
    listExtent regs r = 0 := by
  induction regs with
  | nil => rfl
  | cons p tl ih =>
      obtain ⟨r0, e0⟩ := p
      rw [List.map_cons, List.mem_cons, not_or] at hr
      obtain ⟨h1, h2⟩ := hr
      simp only [listExtent, if_neg h1]
      exact ih h2

/-- Extent closure for the canonical allocation (the `hcov` side condition of
the flat-memory bridge). -/
theorem ofList_closed (flat : RegionName) (regs : List (RegionName × Nat)) :
    ∀ r, r ∉ (ofList flat regs).regions → (ofList flat regs).extent r = 0 :=
  fun _ hr => listExtent_eq_zero hr

/-- End-to-end intervals of a duplicate-free region table are disjoint: an
entry's interval `[listBase, listBase + listExtent)` ends before the base of
every later entry. Generic prefix-sum disjointness, by induction on the
table. -/
private theorem listBase_disjoint (regs : List (RegionName × Nat))
    (hnd : (regs.map Prod.fst).Nodup) :
    ∀ r ∈ regs.map Prod.fst, ∀ r' ∈ regs.map Prod.fst, r ≠ r' →
      listBase regs r + listExtent regs r ≤ listBase regs r' ∨
      listBase regs r' + listExtent regs r' ≤ listBase regs r := by
  induction regs with
  | nil => intro r hr; simp at hr
  | cons p tl ih =>
      obtain ⟨r0, e0⟩ := p
      rw [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨h0, hnd⟩ := hnd
      intro r hr r' hr' hne
      rw [List.map_cons, List.mem_cons] at hr hr'
      rcases hr with rfl | hr
      · rcases hr' with rfl | hr'
        · exact absurd rfl hne
        · left
          have hne' : r' ≠ r := fun h => hne h.symm
          simp only [listBase, listExtent, if_true, if_neg hne']
          omega
      · have hner0 : r ≠ r0 := fun h => h0 (h ▸ hr)
        rcases hr' with rfl | hr'
        · right
          simp only [listBase, listExtent, if_true, if_neg hner0]
          omega
        · have hner0' : r' ≠ r0 := fun h => h0 (h ▸ hr')
          simp only [listBase, listExtent, if_neg hner0, if_neg hner0']
          rcases ih hnd r hr r' hr' hne with h | h
          · left; omega
          · right; omega

/-- Disjointness of the canonical allocation: under a duplicate-free region
table, the end-to-end intervals do not overlap (the `hd` side condition of
the flat-memory bridge). -/
theorem ofList_disjoint (flat : RegionName) (regs : List (RegionName × Nat))
    (hnd : (regs.map Prod.fst).Nodup) : (ofList flat regs).Disjoint :=
  fun r hr r' hr' hne => listBase_disjoint regs hnd r hr r' hr' hne

/-- Scalar `readMem` at a translated in-bounds address reads the region-model
cell (the `.real` carrier translates identically). The generic read-back step
of every denotation instantiation. -/
theorem flattenState_readMem (A : FlatAlloc) (hd : A.Disjoint)
    (s : BlockState) {r : RegionName} (hr : r ∈ A.regions) {o : Nat}
    (ho : o < A.extent r) :
    (A.flattenState s).readMem A.flat (A.addr r o) = s.readMem r o := by
  unfold BlockState.readMem
  rw [A.flattenState_mem_addr hd s hr ho, A.readAs_trCell]
  cases hcell : (s.mem r o).readAs .real with
  | none => rfl
  | some v => cases v <;> rfl

end FlatAlloc

/-! ## The canonical start state: a slot table -/

/-- One denotation input slot: a named 1-D contiguous window of region
`region`, holding `data k` at offset `off + k` for every lane `k < len`
(cells outside every slot's window read `0`). Program-tile inputs use
`off := pid * B`; shared parameters use `off := 0`. The slot table of a
denotation instantiation is part of the per-kernel audit surface. -/
structure DenoteSlot where
  /-- The region this slot loads. -/
  region : RegionName
  /-- First offset of the loaded window. -/
  off : Nat
  /-- Length of the loaded window. -/
  len : Nat
  /-- Lane values: `data k` sits at offset `off + k` for `k < len`. -/
  data : Nat → ℝ

namespace DenoteSlot

/-- Slot holding the `Fin`-indexed array `arr` at offsets
`[off, off + len)` of `region` — the ergonomic constructor for the arrays a
denotation headline quantifies over. -/
def ofFin (region : RegionName) (off : Nat) {len : Nat}
    (arr : Fin len → ℝ) : DenoteSlot :=
  { region := region, off := off, len := len,
    data := fun k => if h : k < len then arr ⟨k, h⟩ else 0 }

@[simp] theorem ofFin_region (region : RegionName) (off : Nat) {len : Nat}
    (arr : Fin len → ℝ) : (ofFin region off arr).region = region := rfl

/-- Memory contents of a slot table: the first slot claiming a region wins;
inside its window it holds `data`, outside it holds `0`; regions no slot
claims hold `0` everywhere. -/
noncomputable def memOf : List DenoteSlot → RegionName → Nat → MemCell
  | [], _, _ => MemCell.real 0
  | sl :: rest, r, o =>
      if r = sl.region then
        MemCell.real (if o - sl.off < sl.len then sl.data (o - sl.off) else 0)
      else memOf rest r o

/-- Canonical region-model start state of a slot table for program `pid`:
memory holds exactly the slots (everything else `0`), registers are empty,
`undef` is clean, and every program-id axis reads `pid`. -/
noncomputable def state (pid : Nat) (slots : List DenoteSlot) : BlockState where
  mem := memOf slots
  regs := fun _ _ _ => none
  pids := fun _ => pid
  undef := fun _ _ => 0
  numPids := fun _ => 1

/-- `pid` projection of the canonical state. -/
@[simp] theorem state_pid (pid : Nat) (slots : List DenoteSlot) :
    (state pid slots).pid = pid := rfl

/-- The canonical state has the clean `undef` the flat-memory bridge
assumes. -/
@[simp] theorem state_undef (pid : Nat) (slots : List DenoteSlot) :
    (state pid slots).undef = fun _ _ => 0 := rfl

/-- Reading a slot's window in the slot-table memory: under a duplicate-free
slot table, lane `k` of slot `sl` reads `sl.data k`. -/
theorem memOf_read {slots : List DenoteSlot} {sl : DenoteSlot}
    (hmem : sl ∈ slots) (hnd : (slots.map DenoteSlot.region).Nodup)
    {k : Nat} (hk : k < sl.len) :
    memOf slots sl.region (sl.off + k) = MemCell.real (sl.data k) := by
  induction slots with
  | nil => cases hmem
  | cons hd tl ih =>
      rw [List.map_cons, List.nodup_cons] at hnd
      obtain ⟨h0, hnd⟩ := hnd
      rcases List.mem_cons.mp hmem with rfl | hmem'
      · simp only [memOf, Nat.add_sub_cancel_left, if_pos hk, if_true]
      · have hne : sl.region ≠ hd.region := fun h =>
          h0 (h ▸ List.mem_map.mpr ⟨sl, hmem', rfl⟩)
        simp only [memOf, if_neg hne]
        exact ih hmem' hnd

/-- `InputLoadedAt`-style loaded lemma, raw form: in the canonical state,
lane `k` of slot `sl` reads back `sl.data k`. -/
theorem state_read {pid : Nat} {slots : List DenoteSlot} {sl : DenoteSlot}
    (hmem : sl ∈ slots) (hnd : (slots.map DenoteSlot.region).Nodup)
    {k : Nat} (hk : k < sl.len) :
    (state pid slots).readMem sl.region (sl.off + k) = sl.data k := by
  simp [BlockState.readMem, state, memOf_read hmem hnd hk]

/-- `InputLoadedAt`-style loaded lemma, `Fin` form: in the canonical state, a
`Fin`-slot's region holds its array at the slot's window. -/
theorem state_read_ofFin {pid : Nat} {slots : List DenoteSlot}
    {region : RegionName} {off : Nat} {len : Nat} {arr : Fin len → ℝ}
    (hmem : ofFin region off arr ∈ slots)
    (hnd : (slots.map DenoteSlot.region).Nodup) (i : Fin len) :
    (state pid slots).readMem region (off + i.val) = arr i := by
  have h := state_read (pid := pid) hmem hnd (k := i.val) i.isLt
  simpa [ofFin, i.isLt] using h

end DenoteSlot

/-! ## The generic denotation -/

/-- **The generic denotation combinator** (#487): flatten `k` over the
canonical end-to-end allocation of the region table `regs`, run it from the
flattened canonical state loaded with the slot table `slots` for program
`pid`, and read one flat output address back — offset `outOff` of region
`outRegion`. `none` exactly when the (flattened) kernel gets stuck.

This definition is the audited trust point of a denotation-style
specification: it packages the canonical allocation (`FlatAlloc.ofList`),
the canonical loaded start state (`DenoteSlot.state`), the flat-memory
translation (`FlatAlloc.flattenKernel` / `flattenState` — real pointer
arithmetic `base + offset` in one flat address space), the execution, and
the read-back. A per-kernel ⟦·⟧ instantiates it with the kernel, its region
and slot tables, and the output address — those tables are the whole
per-kernel audit surface. -/
noncomputable def denoteKernel (k : ComputeKernel) (flat : RegionName)
    (regs : List (RegionName × Nat)) (pid : Nat) (slots : List DenoteSlot)
    (outRegion : RegionName) (outOff : Nat) : Option ℝ :=
  (exec ((FlatAlloc.ofList flat regs).flattenKernel k.toAlgKernel)
      ((FlatAlloc.ofList flat regs).flattenState
        (DenoteSlot.state pid slots))).map
    (fun sF => sF.readMem (FlatAlloc.ofList flat regs).flat
      ((FlatAlloc.ofList flat regs).addr outRegion outOff))

end VeriTile.Triton
