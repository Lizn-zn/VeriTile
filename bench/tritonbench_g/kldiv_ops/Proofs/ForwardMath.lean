import bench.tritonbench_g.kldiv_ops.Proofs.ForwardStep
import bench.tritonbench_g.kldiv_ops.Proofs.BackwardMemory

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton FullBackward
open scoped BigOperators

/-- The source's two masked loads agree with the independently padded pointwise formula. -/
theorem lossValue_padded {B : Nat} (s : BlockState) (Y T : RegionName)
    (yBase tBase N chunk : Nat) (eps : ℝ) (logTarget : Bool) (j : TileIndex [B]) :
    lossValue s Y T yBase tBase N (chunk * B : Nat) eps logTarget j =
      FullMath.paddedLoss logTarget eps
        (fun k ↦ s.readMem Y (yBase + k)) (fun k ↦ s.readMem T (tBase + k))
        N (chunk * B + j.1.val) := by
  by_cases h : chunk * B + j.1.val < N
  · have ha : active N (chunk * B : Nat) j = Bool.true := by
      change decide (((chunk * B : Nat) : Int) + j.1.val < (N : Int)) = Bool.true
      apply decide_eq_true
      exact_mod_cast h
    simp only [lossValue, value, ha, ite_true, address_nat, FullMath.paddedLoss,
      if_pos h, Nat.add_assoc]
  · have ha : active N (chunk * B : Nat) j = Bool.false := by
      change decide (((chunk * B : Nat) : Int) + j.1.val < (N : Int)) = Bool.false
      apply decide_eq_false
      exact_mod_cast h
    simp only [lossValue, value, ha, Bool.false_eq_true, ite_false,
      FullMath.paddedLoss, if_neg h]

/-- Summing the actual block values gives the independent mathematical block loss. -/
theorem block_total (s : BlockState) (Y T : RegionName)
    (yBase tBase N B chunk : Nat) (eps : ℝ) (logTarget : Bool) :
    (∑ j : Fin B, lossValue s Y T yBase tBase N (chunk * B : Nat)
      eps logTarget (j, PUnit.unit)) =
      FullMath.blockLoss logTarget eps
        (fun k ↦ s.readMem Y (yBase + k)) (fun k ↦ s.readMem T (tBase + k))
        N B chunk := by
  unfold FullMath.blockLoss
  apply Finset.sum_congr rfl
  intro j _
  exact lossValue_padded s Y T yBase tBase N chunk eps logTarget (j, PUnit.unit)

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
