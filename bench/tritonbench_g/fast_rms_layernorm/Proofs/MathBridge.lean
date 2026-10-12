import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealExecution
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealMath
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.MathBridge
open VeriTile.Triton RealExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem input_data (N B : Nat) (xs : Fin B → ℝ) (j : Fin B) :
    (rmsFwdInputTile N B xs).data (j,PUnit.unit)=some (RealMath.padded N xs j) := by
  simp only [rmsFwdInputTile,RealMath.padded]
  split <;> rfl

theorem square_sum (N B : Nat) (xs : Fin B → ℝ) :
    rmsFwdSumCarrier N B xs=some (RealMath.squareSum N xs) := by
  have hs : (∑ j : Fin B, WithBot.realMul ((rmsFwdInputTile N B xs).data (j,PUnit.unit))
      ((rmsFwdInputTile N B xs).data (j,PUnit.unit)))=some (RealMath.squareSum N xs) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun j => some ((RealMath.padded N xs j)^2)) := by
        apply Finset.sum_congr rfl
        intro j _
        erw [input_data]
        simp only [WithBot.realMul,Option.map₂_some_some,pow_two]
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [rmsFwdSumCarrier,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,Tile.bop,NumericDType.mul]
  exact hs

theorem inv_carrier (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ) :
    rmsFwdInvVarCarrier N B eps xs=some (RealMath.invStd N eps xs) := by
  simp only [rmsFwdInvVarCarrier,square_sum,Option.map_some,WithBot.realRsqrt_some]
  rfl

theorem inv_eq (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ) :
    rmsFwdInvVarSpec N B eps xs=RealMath.invStd N eps xs := by
  simp only [rmsFwdInvVarSpec,inv_carrier]
  rfl

theorem output_eq (N B : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :
    rmsFwdYSpec N B eps xs ws j=RealMath.output Bool.false N eps xs ws j := by
  simp only [rmsFwdYSpec,inv_carrier,Option.map₂_some_some]
  rfl

theorem gemma_output_eq (N B : Nat) (eps : ℝ) (xs ws : Fin B → ℝ) (j : Fin B) :
    gemmaRmsFwdYSpec N B eps xs ws j=RealMath.output Bool.true N eps xs ws j := by
  simp only [gemmaRmsFwdYSpec,inv_carrier,Option.map₂_some_some]
  change xs j*RealMath.invStd N eps xs*(ws j+1.0)=_
  norm_num [RealMath.output]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.MathBridge
