import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealExecution
import bench.tritonbench_g.fast_rms_layernorm.Proofs.RealMath
namespace VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardMath
open VeriTile.Triton RealExecution
noncomputable section
set_option maxHeartbeats 2000000
set_option maxRecDepth 20000
set_option linter.unusedSimpArgs false

def q (gemma : Bool) (u : BlockState) (Y W : RegionName) (YR N B : Nat) (j : Fin B) : ℝ :=
  RealMath.padded N (fun k : Fin B => u.readMem Y (u.pid*YR+k.val)) j *
    (if gemma then RealMath.padded N (fun k : Fin B => u.readMem W k.val) j+1
     else RealMath.padded N (fun k : Fin B => u.readMem W k.val) j)

def z (u : BlockState) (X r : RegionName) (XR RR N B : Nat) (j : Fin B) : ℝ :=
  RealMath.padded N (fun k : Fin B => u.readMem X (u.pid*XR+k.val)) j*u.readMem r (u.pid*RR)

theorem normed_data (u : BlockState) (X r : RegionName) (XR RR N B : Nat) (j : Fin B) :
    (rmsBackwardNormedTile u X r XR RR N B).data (j,PUnit.unit)=some (z u X r XR RR N B j) := by
  by_cases hj : j.val<N
  all_goals simp [rmsBackwardNormedTile,z,RealMath.padded,rowElem,hj]

theorem plain_dyw (u : BlockState) (Y W : RegionName) (YR N B : Nat) (j : Fin B) :
    (rmsBackwardDYTilePlain u Y W YR N B).data (j,PUnit.unit)=some (q Bool.false u Y W YR N B j) := by
  by_cases hj : j.val<N
  all_goals norm_num [rmsBackwardDYTilePlain,q,RealMath.padded,rowElem,hj]

theorem plain_sum (u : BlockState) (Y X W r : RegionName) (YR XR RR N B : Nat) :
    rmsBackwardRowSumCarrierPlain u Y X W r YR XR RR N B=
      some (∑ j,q Bool.false u Y W YR N B j*z u X r XR RR N B j) := by
  have hs : (∑ j : Fin B,WithBot.realMul
      ((rmsBackwardDYTilePlain u Y W YR N B).data (j,PUnit.unit))
      ((rmsBackwardNormedTile u X r XR RR N B).data (j,PUnit.unit)))=
      some (∑ j,q Bool.false u Y W YR N B j*z u X r XR RR N B j) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
        (fun j => some (q Bool.false u Y W YR N B j*z u X r XR RR N B j)) := by
          apply Finset.sum_congr rfl
          intro j _
          erw [plain_dyw,normed_data]
          rfl
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [rmsBackwardRowSumCarrierPlain,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,Tile.bop,NumericDType.mul]
  exact hs

theorem plain_gradient (u : BlockState) (Y X W r : RegionName) (YR XR RR N B : Nat)
    (j : Fin B) (hj : j.val<N) :
    rmsBackwardDYSpecPlain u Y X W r YR XR RR N B j=
      RealMath.gradient Bool.false N (u.readMem r (u.pid*RR))
        (fun k : Fin B => u.readMem X (u.pid*XR+k.val))
        (fun k : Fin B => u.readMem Y (u.pid*YR+k.val))
        (fun k : Fin B => u.readMem W k.val) j := by
  simp only [rmsBackwardDYSpecPlain,plain_sum,Option.map_some]
  change (u.readMem r (u.pid*RR)/(N : ℝ))*((N : ℝ)*
    (u.readMem Y (u.pid*YR+j.val)*u.readMem W j.val)-
    u.readMem X (u.pid*XR+j.val)*u.readMem r (u.pid*RR)*
      (∑ k,q Bool.false u Y W YR N B k*z u X r XR RR N B k))=_
  norm_num [RealMath.gradient,q,z,RealMath.padded,hj]

theorem gemma_dyw (u : BlockState) (Y W : RegionName) (YR N B : Nat) (j : Fin B) :
    (rmsBackwardDYTileGemma u Y W YR N B).data (j,PUnit.unit)=some (q Bool.true u Y W YR N B j) := by
  by_cases hj : j.val<N
  all_goals norm_num [rmsBackwardDYTileGemma,q,RealMath.padded,rowElem,hj]

theorem gemma_sum (u : BlockState) (Y X W r : RegionName) (YR XR RR N B : Nat) :
    rmsBackwardRowSumCarrierGemma u Y X W r YR XR RR N B=
      some (∑ j,q Bool.true u Y W YR N B j*z u X r XR RR N B j) := by
  have hs : (∑ j : Fin B,WithBot.realMul
      ((rmsBackwardDYTileGemma u Y W YR N B).data (j,PUnit.unit))
      ((rmsBackwardNormedTile u X r XR RR N B).data (j,PUnit.unit)))=
      some (∑ j,q Bool.true u Y W YR N B j*z u X r XR RR N B j) := by
    calc
      _ = @Finset.sum (Fin B) (WithBot ℝ) _ Finset.univ
        (fun j => some (q Bool.true u Y W YR N B j*z u X r XR RR N B j)) := by
          apply Finset.sum_congr rfl
          intro j _
          erw [gemma_dyw,normed_data]
          rfl
      _ = _ := WithBot.sum_someTerm_eq_some _ _
  simp only [rmsBackwardRowSumCarrierGemma,Tile.reduceSum,Tile.reduceSumDrop,
    TileShape.axisDim,TileShape.eraseAxis,TileShape.insertAxisIndex,Tile.bop,NumericDType.mul]
  exact hs

theorem gemma_gradient (u : BlockState) (Y X W r : RegionName) (YR XR RR N B : Nat)
    (j : Fin B) (hj : j.val<N) :
    rmsBackwardDYSpecGemma u Y X W r YR XR RR N B j=
      RealMath.gradient Bool.true N (u.readMem r (u.pid*RR))
        (fun k : Fin B => u.readMem X (u.pid*XR+k.val))
        (fun k : Fin B => u.readMem Y (u.pid*YR+k.val))
        (fun k : Fin B => u.readMem W k.val) j := by
  simp only [rmsBackwardDYSpecGemma,gemma_sum,Option.map_some]
  change (u.readMem r (u.pid*RR)/(N : ℝ))*((N : ℝ)*
    (u.readMem Y (u.pid*YR+j.val)*(u.readMem W j.val+1.0))-
    u.readMem X (u.pid*XR+j.val)*u.readMem r (u.pid*RR)*
      (∑ k,q Bool.true u Y W YR N B k*z u X r XR RR N B k))=_
  norm_num [RealMath.gradient,q,z,RealMath.padded,hj]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.FastRmsLayernorm.BackwardMath
