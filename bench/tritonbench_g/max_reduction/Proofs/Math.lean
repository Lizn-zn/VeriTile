import bench.tritonbench_g.max_reduction.Proofs.Real
import Mathlib.Data.Finset.Lattice.Fold

namespace VeriTile.Bench.TritonBenchG.MaxReduction
open VeriTile.Triton

/-- Mathematical maximum of the active values, using bottom for padding.
The real model reads back an all-padding result as zero. This does not assert
that a hardware negative-infinity value is the real number zero. -/
noncomputable def blockMaximum (M B pid : Nat) (xs : TileIndex [B] → ℝ) : ℝ :=
  WithBot.unbotD 0 ((Finset.univ : Finset (Fin B)).sup
    (fun i => if pid * B + i.val < M then (some (xs (i, PUnit.unit)) : WithBot ℝ) else ⊥))

/-- The kernel-independent finite maximum agrees with the old reduction spec. -/
theorem blockMaximum_eq_tile (M B pid : Nat) (hB : 0 < B)
    (xs : TileIndex [B] → ℝ) :
    blockMaximum M B pid xs = maxTileSpecOf M B pid xs := by
  simp [blockMaximum, maxTileSpecOf, Tile.reduceMax, Tile.reduceMaxDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, hB,
    Finset.sup'_eq_sup]
  rfl

end VeriTile.Bench.TritonBenchG.MaxReduction
