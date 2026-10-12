import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Triton.Float8E5Bits

/-- Real projection of a byte tile. Invalid signed bytes and nonfinite E5M2
encodings have no real projection, so the entire expression fails. Zero and
subnormal encodings are supported. The FP interpreter keeps the raw payload
instead and can interpret all 256 encodings. -/
noncomputable def decodeTile (dst : FloatDType) (v : Tile .int shape) :
    Option (Tile dst.toTileDType shape) :=
  if (TileShape.allIndices shape).all (fun i =>
      isInt8 (v.data i) && (ofInt (v.data i)).decodeRat.isSome) then
    some ⟨fun i => dst.ofReal (((ofInt (v.data i)).decodeRat.getD 0 : ℚ) : ℝ)⟩
  else none

theorem decodeTile_of_finite (dst : FloatDType) (v : Tile .int shape)
    (values : TileIndex shape → ℚ)
    (hbyte : ∀ i, isInt8 (v.data i) = true)
    (hfinite : ∀ i, (ofInt (v.data i)).decodeRat = some (values i)) :
    decodeTile dst v = some ⟨fun i => dst.ofReal (values i : ℝ)⟩ := by
  simp [decodeTile, hbyte, hfinite]

/-- A single nonfinite lane prevents a real-valued execution; it is not
silently converted into the negative-infinity sentinel or a default number. -/
theorem decodeTile_nonfinite (dst : FloatDType) (v : Tile .int shape)
    (i : TileIndex shape) (h : (ofInt (v.data i)).decodeRat = none) :
    decodeTile dst v = none := by
  simp only [decodeTile]
  split
  · rename_i hall
    have hi := (List.all_eq_true.mp hall) i (TileShape.mem_allIndices shape i)
    simp [h] at hi
  · rfl

theorem decodeTile_invalid_byte (dst : FloatDType) (v : Tile .int shape)
    (i : TileIndex shape) (h : isInt8 (v.data i) = false) :
    decodeTile dst v = none := by
  simp only [decodeTile]
  split
  · rename_i hall
    have hi := (List.all_eq_true.mp hall) i (TileShape.mem_allIndices shape i)
    simp [h] at hi
  · rfl

end VeriTile.Triton.Float8E5Bits
