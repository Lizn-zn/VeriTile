import bench.tritonbench_g.fast_layernorm.Proofs.RealExecution
namespace VeriTile.Bench.TritonBenchG.FastLayernorm.MathBridge
open VeriTile.Triton RealExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

theorem input_data (N B : Nat) (xs : Fin B → ℝ) (j : Fin B) :
    (layernormInputTile N B xs).data (j,PUnit.unit)=some (RealMath.padded N xs j) := by
  simp only [layernormInputTile,RealMath.padded]
  split <;> rfl

theorem mean_carrier (N B : Nat) (xs : Fin B → ℝ) :
    layernormMeanCarrier N B xs=some (RealMath.mean N xs) := by
  have hs : (∑ j : Fin B, (layernormInputTile N B xs).data (j,PUnit.unit)) =
      some (∑ j, RealMath.padded N xs j) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
        (fun j => some (RealMath.padded N xs j)) := by
          apply Finset.sum_congr rfl
          intro j _
          exact input_data N B xs j
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [layernormMeanCarrier,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex]
  erw [hs]
  rfl

theorem centered_data (N B : Nat) (xs : Fin B → ℝ) (j : Fin B) :
    (layernormCenteredTile N B xs).data (j,PUnit.unit)=
      some (RealMath.padded N xs j-RealMath.mean N xs) := by
  simp only [layernormCenteredTile,mean_carrier]
  erw [input_data]
  rfl

theorem variance_carrier (N B : Nat) (xs : Fin B → ℝ) :
    layernormVarCarrier N B xs=some (RealMath.variance N xs) := by
  have hs : (∑ j : Fin B, WithBot.realMul
      ((layernormCenteredTile N B xs).data (j,PUnit.unit))
      ((layernormCenteredTile N B xs).data (j,PUnit.unit))) =
      some (∑ j, (RealMath.padded N xs j-RealMath.mean N xs)^2) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
        (fun j => some ((RealMath.padded N xs j-RealMath.mean N xs)^2)) := by
          apply Finset.sum_congr rfl
          intro j _
          erw [centered_data]
          simp only [WithBot.realMul,pow_two,Option.map₂_some_some]
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [layernormVarCarrier,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,Tile.bop,NumericDType.mul]
  erw [hs]
  rfl

theorem inv_carrier (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ) :
    layernormInvVarCarrier N B eps xs=some (RealMath.invStd N eps xs) := by
  simp only [layernormInvVarCarrier,variance_carrier,Option.map_some,WithBot.realRsqrt_some]
  rfl

theorem output_eq (N B : Nat) (eps : ℝ) (xs ws bs : Fin B → ℝ) (j : Fin B) :
    layernormYSpec N B eps xs ws bs j=RealMath.output N eps xs ws bs j := by
  simp only [layernormYSpec,mean_carrier,inv_carrier,Option.map₂_some_some]
  rfl

theorem inv_eq (N B : Nat) (eps : ℝ) (xs : Fin B → ℝ) :
    invVarFullSpec N B eps xs=RealMath.invStd N eps xs := by
  simp only [invVarFullSpec,inv_carrier]
  rfl

theorem mean_eq (N B : Nat) (xs : Fin B → ℝ) :
    meanFullSpec N B xs=RealMath.mean N xs := by
  simp only [meanFullSpec,mean_carrier]
  rfl

theorem dyw_data (s : BlockState) (dY W : RegionName) (YR N B : Nat) (j : Fin B) :
    (layernormBackwardDYWTile s dY W YR N B).data (j,PUnit.unit)=
      some (RealMath.padded N (fun k : Fin B => s.readMem dY (s.pid*YR+k.val)) j *
        RealMath.padded N (fun k : Fin B => s.readMem W k.val) j) := by
  by_cases hj : j.val<N
  all_goals simp [layernormBackwardDYWTile,layernormBackwardDYTile,layernormBackwardWTile,
    Tile.bop,NumericDType.mul,RealMath.padded,hj,WithBot.realMul]

theorem normed_data (s : BlockState) (X r mu : RegionName) (XR N B : Nat) (j : Fin B) :
    (layernormBackwardNormedTile s X r mu XR N B).data (j,PUnit.unit)=
      some ((RealMath.padded N (fun k : Fin B => s.readMem X (s.pid*XR+k.val)) j-
        s.readMem mu s.pid)*s.readMem r s.pid) := by
  by_cases hj : j.val<N
  all_goals simp [layernormBackwardNormedTile,layernormBackwardInvVar,layernormBackwardMean,
    RealMath.padded,hj]

theorem gradient_eq (s : BlockState) (dY X W r mu : RegionName) (YR XR N B : Nat)
    (j : Fin B) (hj : j.val<N) :
    layernormDXSpec s dY X W r mu YR XR N B j =
      RealMath.gradient N (s.readMem r s.pid) (s.readMem mu s.pid)
        (fun k : Fin B => s.readMem X (s.pid*XR+k.val))
        (fun k : Fin B => s.readMem dY (s.pid*YR+k.val))
        (fun k : Fin B => s.readMem W k.val) j := by
  let q := fun k : Fin B => RealMath.padded N (fun k : Fin B => s.readMem dY (s.pid*YR+k.val)) k *
    RealMath.padded N (fun k : Fin B => s.readMem W k.val) k
  let z := fun k : Fin B => (RealMath.padded N (fun k : Fin B => s.readMem X (s.pid*XR+k.val)) k-
    s.readMem mu s.pid)*s.readMem r s.pid
  have hq : (∑ k : Fin B, (layernormBackwardDYWTile s dY W YR N B).data (k,PUnit.unit))=
      some (∑ k, q k) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun k => some (q k)) := by
        apply Finset.sum_congr rfl
        intro k _
        exact dyw_data s dY W YR N B k
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  have hz : (∑ k : Fin B, WithBot.realMul
      ((layernormBackwardDYWTile s dY W YR N B).data (k,PUnit.unit))
      ((layernormBackwardNormedTile s X r mu XR N B).data (k,PUnit.unit)))=
      some (∑ k, q k*z k) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ (fun k => some (q k*z k)) := by
        apply Finset.sum_congr rfl
        intro k _
        erw [dyw_data,normed_data]
        rfl
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [layernormDXSpec,rowElem,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,Tile.bop,NumericDType.mul]
  erw [hq,hz]
  simp only [Option.map_some,Option.map₂_some_some]
  change (_-_/ (N : ℝ)-_*_/ (N : ℝ))*_=_
  simp only [RealMath.gradient,RealMath.padded,if_pos hj,div_eq_mul_inv,mul_assoc,q,z]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastLayernorm.MathBridge
