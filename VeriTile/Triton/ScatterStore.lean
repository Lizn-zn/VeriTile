/-
VeriTile.Triton.ScatterStore

Reusable lemmas about masked list-`foldl` scatter stores, shared by the bench
kernels whose writeback loop stores a tile lane-by-lane via
`l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc)`.

These are fully generic over the lane type `α`, the region, and the
offset / value / mask functions — no kernel layout binding — so they live in the
shared library instead of being re-proved in every masked-store kernel
(`rmsnorm_triton`, `bgmv_expand_slice`, …). They sit alongside the per-lane
scatter readback lemmas in `Semantics`.
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

set_option linter.unusedSimpArgs false

/-- A masked scatter-store `foldl` leaves an offset `o` untouched if no active
(`mask k`) lane writes to it. -/
theorem foldl_store_preserve {α : Type} {region : RegionName}
    (offsetFn : α → Nat) (valueFn : α → ℝ) (mask : α → Bool) (o : Nat) (l : List α)
    (s : BlockState) (hnot : ∀ k ∈ l, mask k → offsetFn k ≠ o) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (offsetFn k) (valueFn k) else acc) s).readMem region o
      = s.readMem region o := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]
    cases hm : mask hd
    · simp only [hm, Bool.false_eq_true, if_false]
      exact ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)
    · simp only [hm, if_true]
      rw [ih _ (fun k hk hmk => hnot k (List.mem_cons_of_mem hd hk) hmk)]
      exact BlockState.writeMem_readMem_of_ne_offset s region (offsetFn hd) (valueFn hd) region o
        (hnot hd (List.mem_cons_self) (by rw [hm])).symm

/-- Readback at offset `O` of a masked scatter-store `foldl`: if a unique active
lane `a` writes `O`, the readback is its value `vfn a`. -/
theorem foldl_store_at {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (O : Nat) (l : List α)
    (s : BlockState) (a : α) (ha : a ∈ l) (hma : mask a) (hoa : ofn a = O)
    (huniq : ∀ b ∈ l, mask b → ofn b = O → b = a)
    (hnodup : l.Nodup) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).readMem region O
      = vfn a := by
  obtain ⟨l₁, l₂, hl⟩ := List.append_of_mem ha
  subst hl
  rw [List.foldl_append, List.foldl_cons]
  rw [List.nodup_append, List.nodup_cons] at hnodup
  obtain ⟨hnd1, ⟨ha_notin2, hnd2⟩, hdisj⟩ := hnodup
  have h2 : ∀ b ∈ l₂, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    have : b = a := huniq b (by simp [List.mem_append, hb]) hmb heq
    exact ha_notin2 (this ▸ hb)
  rw [foldl_store_preserve ofn vfn mask O l₂ _ (fun b hb hmb => h2 b hb hmb)]
  simp only [hma, if_true]
  rw [hoa]
  rw [BlockState.writeMem_readMem]
  have h1 : ∀ b ∈ l₁, mask b → ofn b ≠ O := by
    intro b hb hmb heq
    have hb' : b = a := huniq b (by simp [List.mem_append, hb]) hmb heq
    exact (hdisj b hb a (List.mem_cons_self)) hb'
  rw [foldl_store_preserve ofn vfn mask O l₁ _ (fun b hb hmb => h1 b hb hmb)]
  simp

/-- A masked scatter-store `foldl` leaves all registers untouched. -/
theorem foldl_store_regs {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState)
    (dtype : TileDType) (shape : TileShape) (name : RegName) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).regs dtype shape name
      = s.regs dtype shape name := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons]; cases mask hd <;> simp [ih]

/-- A masked scatter-store `foldl` leaves the program ids untouched. -/
theorem foldl_store_pids {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).pids = s.pids := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih => rw [List.foldl_cons]; cases mask hd <;> simp [ih]

/-- A masked scatter-store `foldl` into `region` leaves any other region `r`
untouched. -/
theorem foldl_store_other_region {α : Type} {region : RegionName}
    (ofn : α → Nat) (vfn : α → ℝ) (mask : α → Bool) (l : List α) (s : BlockState)
    (r : RegionName) (ofs : Nat) (hr : r ≠ region) :
    (l.foldl (fun acc k => if mask k then acc.writeMem region (ofn k) (vfn k) else acc) s).readMem r ofs
      = s.readMem r ofs := by
  induction l generalizing s with
  | nil => rfl
  | cons hd tl ih =>
    rw [List.foldl_cons]; cases mask hd
    · simp only [Bool.false_eq_true, if_false]; exact ih _
    · simp only [if_true]; rw [ih]; exact BlockState.writeMem_readMem_of_ne_region s region (ofn hd) (vfn hd) r ofs hr

/-- A `Finset.univ` sum of `some (g k)` over `WithBot ℝ` equals `some` of the
real sum — the readback bridge for `reduceSum` of an all-`some` tile. -/
theorem withBot_sum_some {B : Nat} (g : Fin B → ℝ) :
    @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun k => (some (g k) : WithBot ℝ))
      = some (Finset.univ.sum g) := by
  show (Finset.univ.sum fun k => ((g k : ℝ) : WithBot ℝ)) = ((Finset.univ.sum g : ℝ) : WithBot ℝ)
  exact (WithBot.coe_sum Finset.univ g).symm

end VeriTile.Triton
