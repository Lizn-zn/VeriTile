/- Shared enumeration bridges for tile-indexed kernel contracts. -/

import VeriTile.Triton.Memory.KernelSpec.Base

namespace VeriTile.Triton

/-- Decidable equality on tile indices, by induction on the shape. Needed to
locate a tile index inside `TileShape.allIndices`. -/
instance instDecidableEqTileIndex : ∀ (shape : TileShape), DecidableEq (TileIndex shape)
  | [] => fun _ _ => isTrue (by rfl)
  | _ :: rest =>
      have : DecidableEq (TileIndex rest) := instDecidableEqTileIndex rest
      inferInstanceAs (DecidableEq (Fin _ × TileIndex rest))

/-- The position of a tile index inside `TileShape.allIndices`. This is the
**data-level** inverse of the enumeration: `mem_allIndices` only gives
existence, which is not enough to transport a core lane function back to a
tile-indexed one. -/
def tilePos (shape : TileShape) (i : TileIndex shape) :
    Fin (TileShape.allIndices shape).length :=
  ⟨(TileShape.allIndices shape).idxOf i,
    List.idxOf_lt_length_of_mem (TileShape.mem_allIndices shape i)⟩

/-- Round-trip: reading the enumeration at a tile index's own position gives
that index back. -/
@[simp] theorem get_tilePos (shape : TileShape) (i : TileIndex shape) :
    (TileShape.allIndices shape).get (tilePos shape i) = i := by
  simp [tilePos]

/-- The other round-trip: a lane's position is the lane itself. This one is
where `TileShape.allIndices_nodup` is actually needed — without duplicate-
freeness two distinct lanes could share a position and the enumeration would
not be a bijection. -/
@[simp] theorem tilePos_get (shape : TileShape)
    (j : Fin (TileShape.allIndices shape).length) :
    tilePos shape ((TileShape.allIndices shape).get j) = j := by
  apply Fin.ext
  simp only [tilePos, List.get_eq_getElem]
  exact List.Nodup.idxOf_getElem (TileShape.allIndices_nodup shape) _ _

/-- Transport a lane-wise obligation between the core's `Fin`-indexed
enumeration and tile indices. The `←` direction is just instantiation; the
`→` direction is exactly what `tilePos` was introduced for. -/
theorem forall_tileIndex_iff {shape : TileShape} (p : TileIndex shape → Prop) :
    (∀ j : Fin (TileShape.allIndices shape).length,
        p ((TileShape.allIndices shape).get j)) ↔ ∀ i : TileIndex shape, p i := by
  constructor
  · intro h i
    have hi := h (tilePos shape i)
    rwa [get_tilePos] at hi
  · intro h j
    exact h _

end VeriTile.Triton
