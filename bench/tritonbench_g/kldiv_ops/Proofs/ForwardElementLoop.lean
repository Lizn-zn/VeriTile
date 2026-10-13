import bench.tritonbench_g.kldiv_ops.Proofs.ForwardElementInvariant

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullForward
open VeriTile.Triton

/-- Positive-stride source execution visits the whole row, including a partial last block. -/
theorem element_loop_run (initial : BlockState) (Y T O : RegionName)
    (sy st so yBase tBase outputBase N B : Nat) (eps : ℝ) (logTarget : Bool)
    (hB : 0 < B) (hYO : Y ≠ O) (hTO : T ≠ O) :
    ∀ c s, ElementInv initial Y T O yBase tBase outputBase N B c eps logTarget s →
      ∃ covered final, N ≤ covered * B ∧
        stepForRangeIntAux "i" (c * B : Nat) N B (body Y T O sy st so N B eps logTarget 0) s =
          some final ∧
        ElementInv initial Y T O yBase tBase outputBase N B covered eps logTarget final := by
  intro c s
  induction h : (N - c * B) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hi
    by_cases hd : N ≤ c * B
    · refine ⟨c, s, hd, ?_, hi⟩
      exact stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) (by exact_mod_cast hd)
    · obtain ⟨middle, hm, hinv⟩ :=
        element_invariant_step initial s Y T O sy st so yBase tBase outputBase N B c
          eps logTarget hYO hTO hi
      have hnext : N - (c + 1) * B < remaining := by
        rw [Nat.add_mul, Nat.one_mul]
        omega
      obtain ⟨covered, final, hcover, hf, hfinal⟩ :=
        ih (N - (c + 1) * B) hnext (c + 1) middle rfl hinv
      refine ⟨covered, final, hcover, ?_, hfinal⟩
      rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hB) (by exact_mod_cast (by omega : c * B < N)), hm]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul, Nat.cast_add] using hf

/-- A covered invariant yields the pointwise formula at every logical column. -/
theorem ElementInv.complete {initial final : BlockState} {Y T O : RegionName}
    {yBase tBase outputBase N B covered : Nat} {eps : ℝ} {logTarget : Bool}
    (h : ElementInv initial Y T O yBase tBase outputBase N B covered eps logTarget final)
    (hB : 0 < B) (hcover : N ≤ covered * B) :
    ∀ k, k < N → final.readMem O (outputBase + k) =
      FullMath.loss logTarget eps (initial.readMem Y (yBase + k))
        (initial.readMem T (tBase + k)) := by
  intro k hk
  have ht : k / B < covered := (Nat.div_lt_iff_lt_mul hB).mpr (by omega)
  have hl := Nat.mod_lt k hB
  have he : k / B * B + k % B = k := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod k B
  simpa only [he] using h.values (k / B) (k % B) ht hl (by omega)

end VeriTile.Bench.TritonBenchG.KldivOps.FullForward

#guard_msgs (drop info) in
#auditModuleAxioms
