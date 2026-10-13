import bench.tritonbench_g.flash_attn.Proofs.RealInvariantMath
import bench.tritonbench_g.attention_fwd_triton3.Proofs.RealTiles

/-! A bridge from the reusable tile-level attention sums to FlashAttention's
independent prefix formula. The seed here is exactly (-infinity, 0, 0). -/
namespace VeriTile.Bench.TritonBenchG.FlashAttn.RealMath
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

theorem maxScore_eq_maximum (xs : List (ℝ × ℝ)) :
    RealFormula.maxScore xs = maximum xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simpa [RealFormula.maxScore, maximum] using congrArg ((x.1 : WithBot ℝ) ⊔ ·) ih

theorem summary_eq_running (xs : List (ℝ × ℝ)) :
    let z := RealFormula.summarize ⟨⊥, 0, 0⟩ xs
    (z.center, z.denominator, z.numerator) = running xs := by
  by_cases hn : xs = []
  · subst xs
    simp [RealFormula.summarize, RealFormula.maxScore, RealFormula.sumAt, running]
  obtain ⟨m, hm⟩ := maximum_finite xs hn
  obtain ⟨hd, ha⟩ := anchored xs
  have hc : RealFormula.maxScore xs = (m : WithBot ℝ) := (maxScore_eq_maximum xs).trans hm
  apply Prod.ext
  · simpa [RealFormula.summarize, maxScore_eq_maximum] using (running_max xs).symm
  apply Prod.ext
  · simp only [RealFormula.summarize, bot_sup_eq, zero_mul, zero_add, hc]
    rw [RealFormula.sumAt_factor]
    simpa [hm, RealFormula.unshiftedSum, denominator] using hd.symm
  · simp only [RealFormula.summarize, bot_sup_eq, zero_mul, zero_add, hc]
    rw [RealFormula.sumAt_factor]
    simpa [hm, RealFormula.unshiftedSum, numerator] using ha.symm

end
end VeriTile.Bench.TritonBenchG.FlashAttn.RealMath

namespace VeriTile.Bench.TritonBenchG.FlashAttn.Formula
open VeriTile.Triton
open VeriTile.Bench.TritonBenchG.AttentionFwdTriton3
noncomputable section

def prefixSummary (p : Layout) (read : RegionName → Nat → ℝ) (Q K V : RegionName)
    (pid₀ pid₁ i d c : Nat) : RealFormula.RowState :=
  RealFormula.summarize ⟨⊥, 0, 0⟩ (prefixPairs p read Q K V pid₀ pid₁ i d c)

theorem blockPairs_finRange (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d c : Nat) :
    blockPairs p read Q K V pid₀ pid₁ i d c =
      RealTiles.pairs
        (fun j : Fin p.BLOCK_N => score p read Q K pid₀ pid₁ i (c * p.BLOCK_N + j.val))
        (fun j : Fin p.BLOCK_N => read V (vOffset p pid₁ (c * p.BLOCK_N + j.val) d))
        (fun j : Fin p.BLOCK_N => visible p pid₀ i (c * p.BLOCK_N + j.val)) := by
  unfold blockPairs blockKeys RealTiles.pairs
  rw [← List.map_coe_finRange_eq_range]
  simp only [List.map_map, List.filter_map, Function.comp_def]

theorem prefixSummary_succ (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d c : Nat) :
    RealFormula.summarize (prefixSummary p read Q K V pid₀ pid₁ i d c)
      (blockPairs p read Q K V pid₀ pid₁ i d c) =
        prefixSummary p read Q K V pid₀ pid₁ i d (c + 1) := by
  simp only [prefixSummary, RealFormula.summarize_append, prefixPairs_succ]

private theorem maxScore_map_value (ks : List Nat) (f v w : Nat → ℝ) :
    RealFormula.maxScore (ks.map (fun k => (f k, v k))) =
      RealFormula.maxScore (ks.map (fun k => (f k, w k))) := by
  induction ks with
  | nil => rfl
  | cons k ks ih => simp only [List.map_cons, RealFormula.maxScore, ih]

theorem prefixSummary_value_independent (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d e c : Nat) :
    (prefixSummary p read Q K V pid₀ pid₁ i d c).center =
      (prefixSummary p read Q K V pid₀ pid₁ i e c).center ∧
    (prefixSummary p read Q K V pid₀ pid₁ i d c).denominator =
      (prefixSummary p read Q K V pid₀ pid₁ i e c).denominator := by
  have hm := maxScore_map_value (prefixKeys p pid₀ i c)
    (fun k => score p read Q K pid₀ pid₁ i k)
    (fun k => read V (vOffset p pid₁ k d)) (fun k => read V (vOffset p pid₁ k e))
  simp only [prefixSummary, prefixPairs, RealFormula.summarize, bot_sup_eq,
    zero_mul, zero_add, hm]
  simp [RealFormula.sumAt, List.map_map, Function.comp_def]

theorem prefixSummary_full (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d : Nat) :
    let z := prefixSummary p read Q K V pid₀ pid₁ i d
      (Geometry.tiles (hi p pid₀) p.BLOCK_N)
    (z.center, z.denominator, z.numerator) =
      RealMath.running (pairs p read Q K V pid₀ pid₁ i d) := by
  simpa only [prefixSummary, prefixPairs_full] using
    RealMath.summary_eq_running (pairs p read Q K V pid₀ pid₁ i d)

@[simp] theorem prefixSummary_zero (p : Layout) (read : RegionName → Nat → ℝ)
    (Q K V : RegionName) (pid₀ pid₁ i d : Nat) :
    prefixSummary p read Q K V pid₀ pid₁ i d 0 = ⟨⊥, 0, 0⟩ := by
  simp [prefixSummary, RealFormula.summarize, RealFormula.maxScore, RealFormula.sumAt]

end
#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.FlashAttn.Formula
