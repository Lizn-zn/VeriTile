import VeriTile.Triton

namespace VeriTile.Bench.TritonBenchG.IndexSelectCat
open VeriTile.Triton

/-- Mathematical row-major enumeration, independent of a kernel or its registers. -/
def gatherLanes (BI BC : Nat) : List (Fin BI × Fin BC) :=
  (List.finRange BI).flatMap fun i => (List.finRange BC).map fun j => (i,j)

/-- The last active matching lane wins in the per-program ordered gather model.
The prior cell is preserved when no active lane addresses it. -/
def gatherCell (BI BC p0 p1 n c s0 s1 : Nat) (xs : Fin BI → Fin BC → ℝ) (out r : RegionName) (o : Nat) (prior : MemCell) : MemCell :=
  (gatherLanes BI BC).foldl (fun cell ij =>
    if p0 * BI + ij.1.val < n ∧ p1 * BC + ij.2.val < c ∧
        r = out ∧ o = (p0 * BI + ij.1.val) * s0 + (p1 * BC + ij.2.val) * s1 then
      MemCell.real (xs ij.1 ij.2)
    else cell) prior


theorem fold_gather_cell {ι : Type} (lanes : List ι) (active : ι → Prop)
    [DecidablePred active] (offset : ι → Nat) (value : ι → ℝ)
    (out r : RegionName) (o : Nat) (s : BlockState) :
    (lanes.foldl (fun state i => if active i then
      state.writeMem out (offset i) (value i) else state) s).mem r o =
    lanes.foldl (fun cell i => if active i ∧ r = out ∧ o = offset i then
      MemCell.real (value i) else cell) (s.mem r o) := by
  induction lanes generalizing s with
  | nil => rfl
  | cons i rest ih =>
    simp only [List.foldl_cons, ih]
    by_cases ha : active i
    · simp only [if_pos ha, ha, true_and]
      congr 1
    · simp [ha]

theorem gatherCell_frame (BI BC p0 p1 n c s0 s1 : Nat) (xs : Fin BI → Fin BC → ℝ) (out r : RegionName) (o : Nat) (prior : MemCell)
    (h : ∀ (i : Fin BI) (j : Fin BC), p0 * BI + i.val < n → p1 * BC + j.val < c →
      r ≠ out ∨ o ≠ (p0 * BI + i.val) * s0 + (p1 * BC + j.val) * s1) :
    gatherCell BI BC p0 p1 n c s0 s1 xs out r o prior = prior := by
  unfold gatherCell
  have hf : ∀ (ij : Fin BI × Fin BC) (cell : MemCell),
      (if p0 * BI + ij.1.val < n ∧ p1 * BC + ij.2.val < c ∧
          r = out ∧ o = (p0 * BI + ij.1.val) * s0 + (p1 * BC + ij.2.val) * s1 then
        MemCell.real (xs ij.1 ij.2) else cell) = cell := by
    intro ij cell
    split_ifs with ha
    · rcases h ij.1 ij.2 ha.1 ha.2.1 with hr | ho
      · exact False.elim (hr ha.2.2.1)
      · exact False.elim (ho ha.2.2.2)
    · rfl
  simp only [hf]
  induction gatherLanes BI BC with
  | nil => rfl
  | cons a rest ih => simpa only [List.foldl_cons] using ih


theorem gatherCell_tile (BI BC p0 p1 n c s0 s1 : Nat) (xs : Fin BI → Fin BC → ℝ) (out r : RegionName) (o : Nat) (prior : MemCell) :
    gatherCell BI BC p0 p1 n c s0 s1 xs out r o prior =
    (TileShape.allIndices [BI,BC]).foldl (fun cell i =>
      if p0 * BI + i.1.val < n ∧ p1 * BC + i.2.1.val < c ∧
          r = out ∧ o = (p0 * BI + i.1.val) * s0 + (p1 * BC + i.2.1.val) * s1 then
        MemCell.real (xs i.1 i.2.1) else cell) prior := by
  simp [gatherCell, gatherLanes, TileShape.allIndices,
    List.foldl_flatMap, List.foldl_map]

theorem signed_gather_address (row : Int) (h : 0 ≤ row) (s0 col s1 : Nat) :
    (row * (s0 : Int) + (col * s1 : Nat)).toNat = row.toNat * s0 + col * s1 := by
  rw [Int.toNat_add (mul_nonneg h (Int.natCast_nonneg _)) (Int.natCast_nonneg _),
    Int.toNat_mul h (Int.natCast_nonneg _)]
  simp [Int.toNat_mul (Int.natCast_nonneg col) (Int.natCast_nonneg s1)]

end VeriTile.Bench.TritonBenchG.IndexSelectCat
