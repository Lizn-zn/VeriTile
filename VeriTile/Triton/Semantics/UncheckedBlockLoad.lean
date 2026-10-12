import VeriTile.Triton.Semantics.BlockPtrEval

namespace VeriTile.Triton

/-- With no boundary-check axes and no mask, every lane reads memory.
No unspecified padding value is selected. This is a real-interpreter fact,
not a numerical FP rewrite or a premise that unspecified values equal zero. -/
theorem evalOp_unchecked_block_load {dtype : TileDType} {shape : TileShape}
    (p : Op .blockPtr shape) (s : BlockState) :
    evalOp (.load dtype (.blockPtr p []) .none .unspecified) s =
      evalOp (.load dtype (.blockPtr p []) .none .zero) s := by
  simp only [evalOp]
  cases evalOp p s <;> simp [BlockPtr.inBounds]

end VeriTile.Triton
