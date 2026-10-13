import Mathlib.Tactic
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.CopyMath

/-- Input column shift of store number `t`, retaining the cached outer/middle tiles. -/
def inputShift (t : Nat) : Nat :=
  let i := t / 12
  let j := t % 12 / 6
  let k := t % 6 / 3
  if t % 3 = 0 then 14 * i
  else if t % 3 = 1 then 14 * i + 6 * j + 2
  else 14 * i + 6 * j + 2 * k + 4

/-- Signed element offset; conversion to the allocation address occurs only afterwards. -/
def offset (sm sn : Int) (shift row col : Nat) : Int :=
  (row : Int) * sm + ((col + shift : Nat) : Int) * sn

/-- Independent ordered copy oracle. Across-store overlap is intentional: later writes win. -/
def expected {α : Type*} (input initial : Nat → α) (sm sn : Int) : Nat → α :=
  (List.range 24).foldl (fun mem t ↦
    (List.range 2).foldl (fun mem row ↦
      (List.range 2).foldl (fun mem col ↦
        Function.update mem (offset sm sn (2 * t) row col).toNat
          (input (offset sm sn (inputShift t) row col).toNat)) mem) mem) initial

def copyTile {α : Type*} (input : Nat → α) (sm sn : Int)
    (mem : Nat → α) (t : Nat) : Nat → α :=
  (List.range 2).foldl (fun mem row ↦
    (List.range 2).foldl (fun mem col ↦
      Function.update mem (offset sm sn (2 * t) row col).toNat
        (input (offset sm sn (inputShift t) row col).toNat)) mem) mem

def expectedPrefix {α : Type*} (input initial : Nat → α) (sm sn : Int) (n : Nat) : Nat → α :=
  (List.range n).foldl (copyTile input sm sn) initial

theorem expectedPrefix_zero {α : Type*} (input initial : Nat → α) (sm sn : Int) :
    expectedPrefix input initial sm sn 0 = initial := rfl

theorem expectedPrefix_succ {α : Type*} (input initial : Nat → α) (sm sn : Int) (n : Nat) :
    expectedPrefix input initial sm sn (n + 1) =
      copyTile input sm sn (expectedPrefix input initial sm sn n) n := by
  simp [expectedPrefix, List.range_succ, List.foldl_append]

theorem expectedPrefix_full {α : Type*} (input initial : Nat → α) (sm sn : Int) :
    expectedPrefix input initial sm sn 24 = expected input initial sm sn := rfl

def footprint (sm sn : Int) (address : Nat) : Prop :=
  ∃ t < 24, ∃ row < 2, ∃ col < 2,
    address = (offset sm sn (2 * t) row col).toNat

private theorem foldl_at {α β : Type*} (f : (Nat → α) → β → Nat → α)
    (xs : List β) (address : Nat)
    (h : ∀ mem x, x ∈ xs → f mem x address = mem address) (initial : Nat → α) :
    xs.foldl f initial address = initial address := by
  induction xs generalizing initial with
  | nil => rfl
  | cons x xs ih =>
    rw [List.foldl_cons, ih (fun mem y hy ↦ h mem y (by simp [hy]))]
    exact h initial x (by simp)

theorem expected_frame {α : Type*} (input initial : Nat → α) (sm sn : Int)
    (address : Nat) (h : ¬ footprint sm sn address) :
    expected input initial sm sn address = initial address := by
  unfold expected
  apply foldl_at
  intro mem t ht
  apply foldl_at
  intro mem row hr
  apply foldl_at
  intro mem col hc
  apply Function.update_of_ne
  intro he
  apply h
  exact ⟨t, List.mem_range.mp ht, row, List.mem_range.mp hr,
    col, List.mem_range.mp hc, he⟩

theorem inputShift_schedule : (List.range 24).map inputShift =
    [0, 2, 4, 0, 2, 6, 0, 8, 10, 0, 8, 12,
     14, 16, 18, 14, 16, 20, 14, 22, 24, 14, 22, 26] := by
  decide

theorem inputShift_bound {t : Nat} (ht : t < 24) : inputShift t ≤ 26 := by
  interval_cases t <;> decide

theorem inputShift_outer {i j k : Nat} (hi : i < 2) (hj : j < 2) (hk : k < 2) :
    inputShift (12 * i + 6 * j + 3 * k) = 14 * i := by
  interval_cases i <;> interval_cases j <;> interval_cases k <;> decide

theorem inputShift_middle {i j k : Nat} (hi : i < 2) (hj : j < 2) (hk : k < 2) :
    inputShift (12 * i + 6 * j + 3 * k + 1) = 14 * i + 6 * j + 2 := by
  interval_cases i <;> interval_cases j <;> interval_cases k <;> decide

theorem inputShift_inner {i j k : Nat} (hi : i < 2) (hj : j < 2) (hk : k < 2) :
    inputShift (12 * i + 6 * j + 3 * k + 2) = 14 * i + 6 * j + 2 * k + 4 := by
  interval_cases i <;> interval_cases j <;> interval_cases k <;> decide

theorem group_schedule {i j : Nat} (hi : i < 2) (hj : j < 2) :
    inputShift (12 * i + 6 * j) = 14 * i ∧
    inputShift (12 * i + 6 * j + 1) = 14 * i + 6 * j + 2 ∧
    inputShift (12 * i + 6 * j + 2) = 14 * i + 6 * j + 4 ∧
    inputShift (12 * i + 6 * j + 3) = 14 * i ∧
    inputShift (12 * i + 6 * j + 4) = 14 * i + 6 * j + 2 ∧
    inputShift (12 * i + 6 * j + 5) = 14 * i + 6 * j + 6 := by
  interval_cases i <;> interval_cases j <;> decide

theorem offset_nonnegative {sm sn : Int} (hm : 0 ≤ sm) (hn : 0 ≤ sn)
    (shift row col : Nat) : 0 ≤ offset sm sn shift row col := by
  unfold offset
  positivity

theorem offset_bound {sm sn : Int} (hm : 0 ≤ sm) (hn : 0 ≤ sn)
    {shift row col bound : Nat} (hs : shift ≤ bound) (hr : row < 2) (hc : col < 2) :
    offset sm sn shift row col ≤ sm + ((bound + 1 : Nat) : Int) * sn := by
  unfold offset
  have hr' : (row : Int) ≤ 1 := by omega
  have hc' : ((col + shift : Nat) : Int) ≤ ((bound + 1 : Nat) : Int) := by omega
  nlinarith [mul_le_mul_of_nonneg_right hr' hm, mul_le_mul_of_nonneg_right hc' hn]

theorem offset_advance {sm sn : Int} (hm : 0 ≤ sm) (hn : 0 ≤ sn)
    (shift row col : Nat) :
    (offset sm sn shift row col).toNat + (2 * sn).toNat =
      (offset sm sn (shift + 2) row col).toNat := by
  have hz := offset_nonnegative hm hn shift row col
  have he : offset sm sn (shift + 2) row col = offset sm sn shift row col + 2 * sn := by
    unfold offset
    push_cast
    ring
  rw [he]
  omega

theorem footprint_bound {sm sn : Int} (hm : 0 ≤ sm) (hn : 0 ≤ sn)
    {address : Nat} (h : footprint sm sn address) :
    address ≤ (sm + 47 * sn).toNat := by
  rcases h with ⟨t, ht, row, hr, col, hc, rfl⟩
  have hb := offset_bound hm hn (bound := 46) (by omega : 2 * t ≤ 46) hr hc
  norm_num at hb
  omega

/-- Contiguous 8×8 strides overlap output tiles; cached input tile 2 wins at address 8. -/
theorem overlap_oracle_witness :
    expected (fun n ↦ (n : Int)) (fun _ ↦ -1) 8 1 8 = 2 ∧
    expected (fun n ↦ (n : Int)) (fun _ ↦ -1) 8 1 47 = 27 ∧
    expected (fun n ↦ (n : Int)) (fun _ ↦ -1) 8 1 55 = 35 ∧
    expected (fun n ↦ (n : Int)) (fun _ ↦ -1) 8 1 56 = -1 := by
  decide

end VeriTile.Bench.TritonBenchG.NestedLoopsProcessing.CopyMath

#guard_msgs (drop info) in
#auditModuleAxioms
