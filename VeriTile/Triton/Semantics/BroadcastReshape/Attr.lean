/-
VeriTile.Triton.Semantics.BroadcastReshape.Attr

Registers the `tile_elementwise` simp attribute. Kept in a dedicated file
because Lean 4 disallows declaring and using a simp attribute in the same
file (see `Mathlib/Tactic/Attr/Register.lean` for the same pattern).

The attribute lives in `VeriTile.Triton.Semantics.BroadcastReshape`
(see `BroadcastReshape.lean`); this file only registers it.
-/

import Mathlib.Tactic.Attr.Register

/-- Simp set covering Triton's elementwise + broadcast + reshape kernel
machinery: `Tile.bop`, `Tile.cop`, `Tile.uop`, `Tile.ptrAdd`,
`Tile.expandDim`, `Tile.ofReal`, `Tile.natToReal`, `Tile.dot`,
`Tile.transpose`, `Tile.reduceSum`, plus the per-constructor
`Broadcast.leftIndex`/`Broadcast.rightIndex` simp lemmas, and the
`NumericDType.*` / `ComparableDType.*` / `FloatDType.*` projections that
kernel proofs unfold to reach the algorithm-layer ℝ form.

Use as `simp [tile_elementwise]` in kernel correctness proofs to replace
the recurring `simp [Tile.bop, Tile.cop, Tile.uop, Tile.expandDim, …,
NumericDType.add, …, ComparableDType.lt, …]` 8-to-15-name list. -/
register_simp_attr tile_elementwise
