import bench.tritonbench_g.block_sparse_attn.Proofs.RealAccumulation

/-! Eliminate the arbitrary masked-load values on stored rows. Causality makes
every key beyond context have zero probability, so only active V addresses
must be pinned to the independent initial array. Partial query blocks remain
allowed: the row bound is required only for the particular observed row. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueFormula
open VeriTile.Triton RealNormalizationValues
open scoped BigOperators
noncomputable section

def keyStart (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs) (BN visit : Nat) : Int :=
  xs.columnIndex ((schedule.layout * schedule.columnStride : Nat) +
    CSRMath.first schedule xs + (visit : Int)) * (BN : Int)

def value (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs) (BN BD : Nat)
    (second : Bool) (visit : Nat) (j : Fin BN) (d : Fin BD) : ℝ :=
  xs.value (CSRMath.selectedKey schedule xs BN visit j) (if second then BD + d.val else d.val)

def inputs (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks : Nat) (scale : ℝ) (visit : Nat) : Inputs BM BN BD :=
  RealNormalizationMath.inputs
    (fun n r => CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale n)
    (value schedule xs BN BD Bool.false) (value schedule xs BN BD Bool.true) visit

theorem inactive_score (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BN BD blocks queryRow context visit : Nat) (scale : ℝ) (even : Bool) (j : Fin BN)
    (hq : queryRow < context)
    (hi : ¬ RealValueLoad.active (keyStart schedule xs BN visit) context even j) :
    CSRMath.score schedule xs BN BD blocks queryRow scale visit j = ⊥ := by
  have hk : ¬ CSRMath.selectedKey schedule xs BN visit j ≤ (queryRow : Int) := by
    have hi' : ¬ (j.val : Int) + keyStart schedule xs BN visit < (context : Int) :=
      fun h => hi (Or.inr h)
    change ¬ keyStart schedule xs BN visit + (j.val : Int) ≤ (queryRow : Int)
    omega
  simp [CSRMath.score, hk]

theorem inactive_probability (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks context visit : Nat) (scale : ℝ) (even : Bool) (r : Fin BM) (j : Fin BN)
    (hq : schedule.queryBlock * BM + r.val < context)
    (hi : ¬ RealValueLoad.active (keyStart schedule xs BN visit) context even j) :
    (inputs schedule xs BM BN BD blocks scale visit).scaledProbability r j = 0 := by
  have hscore := inactive_score schedule xs BN BD blocks _ context visit scale even j hq hi
  simp only [Inputs.scaledProbability, Inputs.probability]
  change StreamingMath.shifted (CSRMath.score schedule xs BN BD blocks
    (schedule.queryBlock * BM + r.val) scale visit j) _ * _ = 0
  rw [hscore, StreamingMath.shifted_bot_left, zero_mul]

/-- Both actual dot branches compute the independent recurrence on every
stored row. Memory pinning is required only for active V lanes; undef remains
unconstrained. The pointer tile is still to be supplied by the source prefix. -/
theorem update_eq_next (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks context visit stride : Nat) (scale : ℝ) (even second : Bool)
    (s : BlockState) (ptrs : Tile .ptr [BN, BD]) (r : Fin BM) (d : Fin BD)
    (hq : schedule.queryBlock * BM + r.val < context)
    (hv : ∀ j d, RealValueLoad.active (keyStart schedule xs BN visit) context even j →
      s.readMem (RealValueLoad.address ptrs (keyStart schedule xs BN visit) stride second j d).1
        (RealValueLoad.address ptrs (keyStart schedule xs BN visit) stride second j d).2 =
      value schedule xs BN BD second visit j d) :
    let x := inputs schedule xs BM BN BD blocks scale visit
    RealAccumulation.update x.scaledProbability
      (if second then x.scaledSecond else x.scaledAccumulator)
      (RealValueLoad.loaded s ptrs (keyStart schedule xs BN visit) stride context even second) r d =
        StreamingMath.accumulated
          (CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale)
          (fun n j => value schedule xs BN BD second n j d) (visit + 1) := by
  dsimp only
  unfold RealAccumulation.update
  rw [RealValueLoad.weighted_eq s ptrs (keyStart schedule xs BN visit) stride context even second
    ((inputs schedule xs BM BN BD blocks scale visit).scaledProbability r)
    (value schedule xs BN BD second visit)
    (fun j h => inactive_probability schedule xs BM BN BD blocks context visit scale even r j hq h) hv]
  cases second <;> rfl

/-- The visit result is a weighted sum of the initial array, not a formula
defined from the load or its final register. -/
theorem update_eq_formula (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BM BN BD blocks context visit stride : Nat) (scale : ℝ) (even second : Bool)
    (s : BlockState) (ptrs : Tile .ptr [BN, BD]) (r : Fin BM) (d : Fin BD)
    (hq : schedule.queryBlock * BM + r.val < context)
    (hv : ∀ j d, RealValueLoad.active (keyStart schedule xs BN visit) context even j →
      s.readMem (RealValueLoad.address ptrs (keyStart schedule xs BN visit) stride second j d).1
        (RealValueLoad.address ptrs (keyStart schedule xs BN visit) stride second j d).2 =
      value schedule xs BN BD second visit j d) :
    let x := inputs schedule xs BM BN BD blocks scale visit
    RealAccumulation.update x.scaledProbability
      (if second then x.scaledSecond else x.scaledAccumulator)
      (RealValueLoad.loaded s ptrs (keyStart schedule xs BN visit) stride context even second) r d =
        StreamingMath.moment
          (CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale)
          (fun n j => value schedule xs BN BD second n j d) (visit + 1) /
        StreamingMath.mass
          (CSRMath.score schedule xs BN BD blocks (schedule.queryBlock * BM + r.val) scale) (visit + 1) := by
  dsimp only
  rw [update_eq_next schedule xs BM BN BD blocks context visit stride scale even second s ptrs r d hq hv,
    StreamingMath.accumulated_eq_formula]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealValueFormula
