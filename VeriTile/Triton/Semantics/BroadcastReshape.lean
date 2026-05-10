/-
VeriTile.Triton.Semantics.BroadcastReshape

Tags the kernel-side elementwise / broadcast / reshape operations with the
`tile_elementwise` simp attribute. Kernel correctness proofs can replace the
recurring 8-to-15-name simp list

```
simp [Tile.bop, Tile.cop, Tile.uop, Tile.ptrAdd, Tile.expandDim, Tile.ofReal,
      NumericDType.add, NumericDType.mul, NumericDType.sub, NumericDType.div,
      ComparableDType.lt, ComparableDType.gt, …]
```

with the single

```
simp [tile_elementwise]
```

The set is opt-in: kernel proofs that already work via the explicit list keep
working untouched. New proofs (or proofs being refactored) should prefer
`tile_elementwise`.

The attribute is registered in `Semantics/BroadcastReshape/Attr.lean` (Lean
4 disallows declaring and using a simp attribute in the same file).
-/

import VeriTile.Triton.Semantics.BroadcastReshape.Attr
import VeriTile.Triton.Semantics.Scalar
import VeriTile.Triton.Semantics.TileOps

namespace VeriTile.Triton

/-! ## Tile elementwise ops -/

attribute [tile_elementwise]
  Tile.bop Tile.cop Tile.uop Tile.ptrAdd Tile.bop_data Tile.cop_data
  Tile.uop_data

/-! ## Tile shape coercions / reshapes -/

attribute [tile_elementwise]
  Tile.expandDim Tile.ofReal Tile.natToReal Tile.dot Tile.transpose Tile.select

/-! ## Tile reductions -/

attribute [tile_elementwise]
  Tile.reduceSum Tile.reduceSumDrop Tile.reduceSumKeep
  Tile.reduceSum_false Tile.reduceSum_true
  Tile.reduceSumDrop_data Tile.reduceSumKeep_data
  Tile.reduceMax Tile.reduceMaxDrop Tile.reduceMaxKeep
  Tile.reduceMax_false Tile.reduceMax_true

/-! ## Numeric / comparable / integral dtype operations -/

attribute [tile_elementwise]
  NumericDType.add NumericDType.sub NumericDType.mul NumericDType.div
  IntegralDType.floorDiv IntegralDType.mod
  ComparableDType.lt ComparableDType.le ComparableDType.gt ComparableDType.ge
  ComparableDType.eq ComparableDType.ne

/-! ## Float dtype crossings (`tl.x.to(tl.fpY)` round-trips) -/

attribute [tile_elementwise]
  FloatDType.cast FloatDType.ofWithBot FloatDType.toWithBot

/-! ## TileShape arithmetic exposed by reductions and reshape -/

attribute [tile_elementwise]
  TileShape.axisDim TileShape.eraseAxis TileShape.insertAxisIndex
  TileShape.dropInsertedIndex

end VeriTile.Triton
