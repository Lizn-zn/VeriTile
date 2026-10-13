import bench.tritonbench_g.kldiv_triton.Proofs.ForwardInvariant

namespace VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
open VeriTile.Triton

/-- The complete positive-stride signed loop, without a single-block restriction. -/
theorem loop_run (initial : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase base N B reduction : Nat) (flag : Bool)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    ∀ c s, Inv initial Y T O yBase tBase base N B c reduction flag s →
      ∃ covered final, N ≤ covered * B ∧
        stepForRangeIntAux "i" (c * B : Nat) N B (body Y T O sy st so N B flag reduction) s =
          some final ∧
        Inv initial Y T O yBase tBase base N B covered reduction flag final := by
  intro c s
  induction h : (N - c * B) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hi
    by_cases hd : N ≤ c * B
    · refine ⟨c, s, hd, ?_, hi⟩
      exact stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) (by exact_mod_cast hd)
    · have hc : c * B < N := by omega
      obtain ⟨middle, hm, hinv⟩ :=
        invariant_step initial s Y T O sy st so yBase tBase base N B c reduction flag
          hB hc hYO hTO hi
      have hnext : N - (c + 1) * B < remaining := by
        rw [Nat.add_mul, Nat.one_mul]
        omega
      obtain ⟨covered, final, hcover, hf, hfinal⟩ :=
        ih (N - (c + 1) * B) hnext (c + 1) middle rfl hinv
      refine ⟨covered, final, hcover, ?_, hfinal⟩
      rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hB) (by exact_mod_cast hc), hm]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul, Nat.cast_add] using hf

/-- All actual outputs have the independent pointwise or per-chunk formula. -/
theorem Inv.complete {initial final : BlockState} {Y T O : RegionName}
    {yBase tBase base N B covered reduction : Nat} {flag : Bool}
    (h : Inv initial Y T O yBase tBase base N B covered reduction flag final)
    (hB : 0 < B) (hcover : N ≤ covered * B) :
    ∀ k, k < FullMath.outputCount N B reduction →
      final.readMem O (base + k) =
        FullMath.forwardOutput flag (fun i ↦ initial.readMem Y (yBase + i))
          (fun i ↦ initial.readMem T (tBase + i)) N B reduction k := by
  intro k hk
  by_cases hr : reduction = 0
  · have hn : k < N := by simpa [FullMath.outputCount, hr] using hk
    have ht : k / B < covered := (Nat.div_lt_iff_lt_mul hB).mpr (by omega)
    have hl := Nat.mod_lt k hB
    have he : k / B * B + k % B = k := by
      simpa only [Nat.mul_comm] using Nat.div_add_mod k B
    simpa only [FullMath.forwardOutput, hr, if_true, he] using
      h.elements hr (k / B) (k % B) ht hl (by omega)
  · have hn : k < FullMath.rounds N B := by simpa [FullMath.outputCount, hr] using hk
    have hcount := (FullMath.rounds_le_iff N B covered hB).mpr hcover
    simpa only [FullMath.forwardOutput, hr, if_false] using h.chunks hr k (by omega)

end VeriTile.Bench.TritonBenchG.KldivTriton.FullForward
#guard_msgs (drop info) in
#auditModuleAxioms
