import bench.tritonbench_g.kldiv_ops.Proofs.BackwardInvariant

namespace VeriTile.Bench.TritonBenchG.KldivOps.FullBackward
open VeriTile.Triton

/-- Positive-stride source execution visits the whole row, including a partial last block. -/
theorem loop_run (initial : BlockState) (T O : RegionName)
    (st so targetBase outputBase N B : Nat) (logTarget : Bool)
    (hB : 0 < B) (hTO : T ≠ O) :
    ∀ c s, Inv initial T O targetBase outputBase N B c logTarget s →
      ∃ covered final, N ≤ covered * B ∧
        stepForRangeIntAux "i" (c * B : Nat) N B (body T O st so N B logTarget) s =
          some final ∧
        Inv initial T O targetBase outputBase N B covered logTarget final := by
  intro c s
  induction h : (N - c * B) using Nat.strong_induction_on generalizing c s with
  | _ remaining ih =>
    intro hi
    by_cases hd : N ≤ c * B
    · refine ⟨c, s, hd, ?_, hi⟩
      exact stepForRangeIntAux.step_ge (Nat.ne_of_gt hB) (by exact_mod_cast hd)
    · obtain ⟨middle, hm, hinv⟩ :=
        invariant_step initial s T O st so targetBase outputBase N B c logTarget hTO hi
      have hnext : N - (c + 1) * B < remaining := by
        rw [Nat.add_mul, Nat.one_mul]
        omega
      obtain ⟨covered, final, hcover, hf, hfinal⟩ :=
        ih (N - (c + 1) * B) hnext (c + 1) middle rfl hinv
      refine ⟨covered, final, hcover, ?_, hfinal⟩
      rw [stepForRangeIntAux.step_lt (Nat.ne_of_gt hB) (by exact_mod_cast (by omega : c * B < N)), hm]
      simpa only [Option.bind_some, Nat.add_mul, Nat.one_mul, Nat.cast_add] using hf

/-- A covered invariant yields the pointwise formula at every logical column. -/
theorem Inv.complete {initial final : BlockState} {T O : RegionName}
    {targetBase outputBase N B covered : Nat} {logTarget : Bool}
    (h : Inv initial T O targetBase outputBase N B covered logTarget final)
    (hB : 0 < B) (hcover : N ≤ covered * B) :
    ∀ k, k < N → final.readMem O (outputBase + k) =
      gradient logTarget (initial.readMem T (targetBase + k)) := by
  intro k hk
  have ht : k / B < covered := (Nat.div_lt_iff_lt_mul hB).mpr (by omega)
  have hl := Nat.mod_lt k hB
  have he : k / B * B + k % B = k := by
    simpa only [Nat.mul_comm] using Nat.div_add_mod k B
  simpa only [he] using h.values (k / B) (k % B) ht hl (by omega)

end VeriTile.Bench.TritonBenchG.KldivOps.FullBackward

#guard_msgs (drop info) in
#auditModuleAxioms
