import bench.tritonbench_g.flash_attn.Proofs.RealNumericStep
import bench.tritonbench_g.flash_attn.Proofs.RealSummary

/-! The complete source arithmetic update is the independent block summary.
Only real algebra is used. In particular denom*0 is removed only for a finite
denominator, as established by the prefix invariant. -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
open VeriTile.Triton Formula
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

def maskedTile (p : Layout) (sc : Fin p.BLOCK_M → Fin p.BLOCK_N → ℝ)
    (keep : Fin p.BLOCK_M → Fin p.BLOCK_N → Bool) : Tile .real [p.BLOCK_M,p.BLOCK_N] :=
  ⟨fun i => if keep i.1 i.2.1 then (sc i.1 i.2.1 : WithBot ℝ) else ⊥⟩

theorem maximum_matches (p : Layout) (sc : Fin p.BLOCK_M → Fin p.BLOCK_N → ℝ)
    (keep : Fin p.BLOCK_M → Fin p.BLOCK_N → Bool) (old : Fin p.BLOCK_M → WithBot ℝ) :
    updatedMax p ⟨fun i => old i.1⟩ (maskedTile p sc keep) =
      ⟨fun i => RealTiles.center sc keep old i.1⟩ := by
  apply Tile.ext
  rintro ⟨i, ⟨⟩⟩
  simp only [updatedMax, maskedTile, RealTiles.center, RealTiles.maxScore_pairs]

theorem finite_out_scale (p : Layout) (den : Fin p.BLOCK_M → ℝ)
    (alpha : Tile .real [p.BLOCK_M]) :
    Tile.bop NumericDType.real.add (.consSame .nil)
      (Tile.bop NumericDType.real.mul .scalarR
        (RealTiles.finite (fun i : TileIndex [p.BLOCK_M] => den i.1))
        (Tile.scalar (some 0))) alpha = alpha := by
  apply Tile.ext
  rintro ⟨i, ⟨⟩⟩
  change WithBot.realAdd (WithBot.realMul (some (den i)) (some 0))
    (alpha.data (i,PUnit.unit)) = alpha.data (i,PUnit.unit)
  cases h : alpha.data (i,PUnit.unit) <;> simp [WithBot.realAdd, WithBot.realMul]

theorem denominator_summary (p : Layout) (sc : Fin p.BLOCK_M → Fin p.BLOCK_N → ℝ)
    (keep : Fin p.BLOCK_M → Fin p.BLOCK_N → Bool) (old : Fin p.BLOCK_M → WithBot ℝ)
    (den : Fin p.BLOCK_M → ℝ) (row : Fin p.BLOCK_M) (value : Fin p.BLOCK_N → ℝ) (a : ℝ) :
    let mn := updatedMax p ⟨fun i => old i.1⟩ (maskedTile p sc keep)
    (updatedDenom p (RealTiles.finite (fun i : TileIndex [p.BLOCK_M] => den i.1))
      (rescale p ⟨fun i => old i.1⟩ mn)
      (probabilities p (maskedTile p sc keep) mn)).data (row,PUnit.unit) =
        some (RealFormula.summarize ⟨old row, den row, a⟩
          (RealTiles.pairs (sc row) value (keep row))).denominator := by
  dsimp only
  rw [maximum_matches]
  unfold rescale probabilities maskedTile
  rw [RealTiles.alpha_matches, RealTiles.probabilities_match]
  exact RealTiles.denominator_matches p.BLOCK_M p.BLOCK_N sc keep old den row value a

theorem accumulator_summary (p : Layout) (s : BlockState)
    (sc : Fin p.BLOCK_M → Fin p.BLOCK_N → ℝ)
    (keep : Fin p.BLOCK_M → Fin p.BLOCK_N → Bool) (old : Fin p.BLOCK_M → WithBot ℝ)
    (den : Fin p.BLOCK_M → ℝ) (acc : TileIndex [p.BLOCK_M,p.DIM] → ℝ)
    (value : TileIndex [p.BLOCK_N,p.DIM] → ℝ)
    (vp : Tile .blockPtr [p.BLOCK_N,p.DIM]) (hv : loaded s vp = RealTiles.finite value)
    (row : Fin p.BLOCK_M) (lane : Fin p.DIM) :
    let mn := updatedMax p ⟨fun i => old i.1⟩ (maskedTile p sc keep)
    (updatedAcc p s (RealTiles.finite (fun i : TileIndex [p.BLOCK_M] => den i.1))
      (rescale p ⟨fun i => old i.1⟩ mn) (RealTiles.finite acc)
      (probabilities p (maskedTile p sc keep) mn) vp).data (row,lane,PUnit.unit) =
        some (RealFormula.summarize ⟨old row, den row, acc (row,lane,PUnit.unit)⟩
          (RealTiles.pairs (sc row) (fun j => value (j,lane,PUnit.unit)) (keep row))).numerator := by
  dsimp only
  rw [maximum_matches]
  unfold rescale probabilities maskedTile
  rw [RealTiles.alpha_matches, RealTiles.probabilities_match]
  unfold updatedAcc
  rw [finite_out_scale]
  dsimp only
  erw [hv]
  exact RealTiles.numerator_matches p.BLOCK_M p.BLOCK_N p.DIM sc keep old acc value row lane (den row)

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.RealExecution
