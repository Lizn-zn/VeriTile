import VeriTile.Triton.Core.Shape

/-! Order-preserving index maps for equal-size reshapes. -/
namespace VeriTile.Triton.TileShape

@[simp] theorem length_allIndices (shape : TileShape) :
    (allIndices shape).length = product shape := by
  induction shape with
  | nil => rfl
  | cons n rest ih => simp [allIndices, List.length_flatMap, ih, product]

theorem linearIndex_lt (shape : TileShape) (i : TileIndex shape) :
    linearIndex shape i < product shape := by
  induction shape with
  | nil => exact Nat.zero_lt_one
  | cons n rest ih =>
    change i.1.val * product rest + linearIndex rest i.2 < n * product rest
    calc
      _ < i.1.val * product rest + product rest := Nat.add_lt_add_left (ih i.2) _
      _ = (i.1.val + 1) * product rest := (Nat.succ_mul _ _).symm
      _ ≤ n * product rest := Nat.mul_le_mul_right _ i.1.isLt

/-- The output's row-major position selects exactly the same input position.
Equal element counts ensure every lookup succeeds, including empty shapes. -/
def reshapeIndex {input output : TileShape} (sameSize : product input = product output)
    (i : TileIndex output) : TileIndex input :=
  (allIndices input)[linearIndex output i]'(by
    rw [length_allIndices, sameSize]
    exact linearIndex_lt output i)

end VeriTile.Triton.TileShape
