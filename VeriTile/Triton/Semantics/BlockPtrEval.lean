/-
VeriTile.Triton.Semantics.BlockPtrEval

`evalOp` reduction lemmas for block-pointer construction / advance / load.

These are the block-pointer companions of the `@[simp] evalOp_*` family in
`EvalOp.lean`. They give the value of:

* `makeBlockPtrDynOffsets` (the `tl.make_block_ptr` with dynamic per-axis
  offsets) — `makeBlockPtr2_eval`;
* `advanceBlockPtr` (`tl.advance`) — `advanceBlockPtr_eval`;
* a no-boundary-check block-pointer real load — `load_blockPtr_none_real`,
  plus the zero-row-offset (`load_blockPtr_K_eval`) and `[rowOff, 0]`-advanced
  (`load_blockPtr_Q_eval`) specializations that resolve to a per-lane `readMem`
  at the contiguous address `base + row · strideT + col · strideS`.

The address/`inBounds` index lemmas they rely on live in
`Core/Types.lean` + `Core/Shape.lean` (`address_2d_offsets`, `inBounds_nil_*`,
`advance_2d_offsets`, the `blockPtr_*_index` forms).
-/

import VeriTile.Triton.Semantics.EvalOp

namespace VeriTile.Triton

open VeriTile.Triton

/-- **`makeBlockPtrDynOffsets` eval** (`tl.make_block_ptr` with dynamic offsets):
evaluates the base-offset op and each offset op, building a constant tile of the
resulting `BlockPtr`. -/
theorem makeBlockPtr2_eval (region : RegionName) (baseOffset : Op .nat [])
    (parentShape : List Nat) (blockShape : TileShape)
    (strides : List Nat) (offsets : List (Op .nat [])) (s : BlockState) :
    evalOp (.makeBlockPtrDynOffsets region baseOffset parentShape blockShape strides offsets) s
      = (do
          let base ← evalOp baseOffset s
          let offs ← offsets.mapM (fun off => do
            let v ← evalOp off s
            some (v.data PUnit.unit))
          some ⟨fun _ =>
            { region := region
            , baseOffset := base.data PUnit.unit
            , parentShape := parentShape
            , blockShape := blockShape
            , strides := strides
            , offsets := offs }⟩) := by
  simp [evalOp]

/-- **`advanceBlockPtr` eval** (`tl.advance`): advances every lane's `BlockPtr`
by `deltas` (per-axis offset add). -/
@[simp] theorem advanceBlockPtr_eval {shape : TileShape}
    (ptr : Op .blockPtr shape) (deltas : List Int) (s : BlockState) :
    evalOp (.advanceBlockPtr ptr deltas) s
      = (do
          let p ← evalOp ptr s
          some ⟨fun i => (p.data i).advance deltas⟩) := by
  simp [evalOp]

/-- **No-boundary-check block-ptr real load.** With an empty `boundary_check`
list, `inBounds` is vacuously `true`, so an unmasked `.real` block-pointer load
reads `readMem` at each lane's resolved address `bp.address (indexToList …)`. -/
theorem load_blockPtr_none_real {shape : TileShape}
    (ptrOp : Op .blockPtr shape) (s : BlockState) (ptrs : Tile .blockPtr shape)
    (hp : evalOp ptrOp s = some ptrs) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun i =>
          some (s.readMem (ptrs.data i).region
            ((ptrs.data i).address (TileShape.indexToList shape i)))⟩ := by
  simp only [evalOp, hp, Option.bind]
  refine congrArg some ?_
  ext i
  simp only [BlockPtr.inBounds_nil_checkedAxes, if_true, BlockState.readMemValue_real]

/-- **Zero-row-offset block-ptr load.** A no-boundary-check `.real` load of a
well-formed 2D block-pointer with offsets `[0, colOff]` reads `readMem` at the
contiguous per-lane address `base + i·strideT + (colOff + j)·strideS`. This is
the `K`-style block-ptr load (key tile, no row advance yet). -/
theorem load_blockPtr_K_eval
    (region : RegionName) (base rows cols BT BS strideT strideS colOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [0, colOff] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + idx.1.val * strideT + (colOff + idx.2.1.val) * strideS))⟩ := by
  rw [load_blockPtr_none_real ptrOp s _ hp]
  refine congrArg some ?_
  ext idx
  simp only [TileShape.blockPtr_address_2d_zero_row_offset_index]

/-- **`[rowOff, 0]`-advanced block-ptr load.** A no-boundary-check `.real` load
of a well-formed 2D block-pointer with offsets `[rowOff, 0]` reads `readMem` at
`base + (rowOff + i)·strideT + j·strideS`. This is the `Q`-style row-advanced
block-ptr load. -/
theorem load_blockPtr_Q_eval
    (region : RegionName) (base rows cols BT BS strideT strideS rowOff : Nat)
    (ptrOp : Op .blockPtr [BT, BS]) (s : BlockState)
    (hp : evalOp ptrOp s = some
      ⟨fun _ : TileIndex [BT, BS] =>
        { region := region, baseOffset := base, parentShape := [rows, cols],
          blockShape := [BT, BS], strides := [strideT, strideS],
          offsets := [rowOff, 0] }⟩) :
    evalOp (.load .real (.blockPtr ptrOp []) .none) s
      = some ⟨fun idx : TileIndex [BT, BS] =>
          some (s.readMem region
            (base + (rowOff + idx.1.val) * strideT + idx.2.1.val * strideS))⟩ := by
  rw [load_blockPtr_none_real ptrOp s _ hp]
  refine congrArg some ?_
  ext idx
  simp only [TileShape.blockPtr_address_2d_row_offset_index]

end VeriTile.Triton
