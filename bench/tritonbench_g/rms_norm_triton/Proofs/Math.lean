import bench.tritonbench_g.rms_norm_triton.Proofs.Real
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.TritonBenchG.RmsNormTriton
open VeriTile.Triton
open scoped BigOperators

/-- Independent RMS row formula with the complete active prefix. The public
contracts require N≤B so this finite sum covers the entire requested row. -/
noncomputable def rowFormula (N B : Nat) (eps : ℝ) (xs ws : Fin B → ℝ)
    (i : Fin B) : ℝ :=
  (xs i * (1 / Real.sqrt
    ((∑ j : Fin B, if j.val < N then xs j * xs j else 0) / (N : ℝ) + eps))) * ws i

theorem sumCarrier_eq_active_sum (N B : Nat) (xs : Fin B → ℝ) :
    rmsSumCarrier N B xs = some (∑ j : Fin B, if j.val < N then xs j * xs j else 0) := by
  simp only [rmsSumCarrier, rmsInputTile, Tile.reduceSum, Tile.reduceSumDrop,
    TileShape.axisDim, TileShape.eraseAxis, TileShape.insertAxisIndex, Tile.bop,
    NumericDType.mul]
  have hid {α : Type} (p : Prop) [Decidable p] (a b : α) :
      (if p then if p then a else b else b) = (if p then a else b) := by
    by_cases h : p <;> simp [h]
  simpa only [hid] using TiledL2Norm.reduceSum_masked_sq_eq_some_sum xs (fun j => j.val < N)

theorem spec_eq_rowFormula (N B : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (i : Fin B) :
    rmsNormSpec N B eps xs ws i = rowFormula N B eps xs ws i := by
  simp [rmsNormSpec, rmsRrmsCarrier, sumCarrier_eq_active_sum, rowFormula,
    WithBot.realSqrt, WithBot.unbotD, one_div]
  rfl

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.RmsNormTriton
