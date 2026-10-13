import bench.tritonbench_g.kldiv_ops.Proofs.ForwardReductionInvariant

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton

/-- The entire positive-stride reduction loop accumulates enough blocks to cover the row. -/
theorem reduction_loop_run (initial : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase outputBase N B : Nat) (eps : ℝ) (logTarget : Bool)
    (reduction : Nat) (hB : 0 < B) (hR : reduction ≠ 0) :
    ∀ c s, ReductionInv initial Y T O yBase tBase outputBase N B c eps logTarget s →
      ∃ covered final, N ≤ covered * B ∧
        stepForRangeIntAux "i" (c * B : Nat) N B
          (body Y T O sy st so N B eps logTarget reduction) s = some final ∧
        ReductionInv initial Y T O yBase tBase outputBase N B covered eps logTarget final := by
  intro c s
  induction h : (N - c * B) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hi
    by_cases hd : N ≤ c * B
    · refine ⟨c, s, hd, ?_, hi⟩
      exact stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) (by exact_mod_cast hd)
    · obtain ⟨middle, hm, hinv⟩ := reduction_invariant_step initial s Y T O
        sy st so yBase tBase outputBase N B c eps logTarget reduction hR hi
      have hnext : N - (c + 1) * B < remaining := by
        rw [Nat.add_mul, Nat.one_mul]
        omega
      obtain ⟨covered, final, hcover, hf, hfinal⟩ :=
        ih (N - (c + 1) * B) hnext (c + 1) middle rfl hinv
      refine ⟨covered, final, hcover, ?_, hfinal⟩
      rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hB)
        (by exact_mod_cast (by omega : c * B < N)), hm]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul, Nat.cast_add] using hf

/-- At loop exit, the carried accumulator is the independent whole-row loss. -/
theorem ReductionInv.complete {initial final : BlockState} {Y T O : RegionName}
    {yBase tBase outputBase N B covered : Nat} {eps : ℝ} {logTarget : Bool}
    (h : ReductionInv initial Y T O yBase tBase outputBase N B covered eps logTarget final)
    (hcover : N ≤ covered * B) :
    final.regs .real [] "loss_sum" = some (Tile.scalar (some
      (FullMath.rowLoss logTarget eps
        (fun k ↦ initial.readMem Y (yBase + k))
        (fun k ↦ initial.readMem T (tBase + k)) N))) := by
  rw [h.acc, FullMath.prefixLoss_eq_rowLoss logTarget eps _ _ N B covered hcover]

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
