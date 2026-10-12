import bench.tritonbench_g.context_attn_bloom.Proofs.NormalizedFormula

namespace VeriTile.Bench.TritonBenchG.ContextAttnBloom.Formula
open VeriTile.Triton Math NormalizedFormula
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option maxRecDepth 10000
set_option linter.unusedSimpArgs false

/-- Raw exponential weights over the visited blocks, including the actual
padded lanes and finite causal sentinel. -/
def denominator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val) else 0

def numerator (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) : ℝ :=
  ∑ j : Fin (count p l a*p.BN),
    if j.val < c*p.BN then Real.exp (score p l a scale i j.val)*value p l a j.val d else 0

/-- Independent normalized-block formula. Once an entire block lies in the
future of this query, the source leaves the previous accumulator unscaled.
Its finite sentinel still contributes to the prefix denominator and numerator. -/
def expected (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) : Nat → ℝ
  | 0 => 0
  | c+1 =>
    expected p l a scale i d c *
      (if ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch then
        denominator p l a scale i c / denominator p l a scale i (c+1) else 1) +
      (numerator p l a scale i d (c+1)-numerator p l a scale i d c) /
        denominator p l a scale i (c+1)

def pairs (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat) :
    Fin (count p l a*p.BN) → ℝ × ℝ :=
  fun j => (score p l a scale i j.val/Real.log 2,value p l a j.val d)

theorem denominator_eq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) :
    weightSum (count p l a*p.BN) (c*p.BN) (pairs p l a scale i d) =
      denominator p l a scale i c := by
  unfold weightSum gKeysUpto
  rw [g_filterMap_finRange_sum]
  simp only [pairs,natural_exp_score,denominator]

theorem numerator_eq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat) :
    weightedValueSum (count p l a*p.BN) (c*p.BN) (pairs p l a scale i d) =
      numerator p l a scale i d c := by
  unfold weightedValueSum gKeysUpto
  rw [g_filterMap_finRange_sum]
  simp only [pairs,natural_exp_score,numerator]

theorem denominator_pos (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i c : Nat)
    (hBN : 0 < p.BN) (hc : 0 < c) (hcount : c ≤ count p l a) :
    0 < denominator p l a scale i c := by
  have hs : 0 < count p l a*p.BN := Nat.mul_pos (by omega) hBN
  have hz : 0 < c*p.BN := Nat.mul_pos hc hBN
  unfold denominator
  apply Finset.sum_pos'
  · intro j _
    split_ifs <;> positivity
  · refine ⟨⟨0,hs⟩,Finset.mem_univ _,?_⟩
    simp only [hz,↓reduceIte]
    exact Real.exp_pos _

theorem accumulator_eq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d c : Nat)
    (hp : 0 ≤ a.prompt l.batch) :
    accumulator (count p l a*p.BN) p.BN (queryRow p l i+a.prompt l.batch).toNat
      (pairs p l a scale i d) c = expected p l a scale i d c := by
  induction c with
  | zero => rfl
  | succ c ih =>
    have hg : c*p.BN ≤ (queryRow p l i+a.prompt l.batch).toNat ↔
        ((c*p.BN : Nat) : Int) ≤ (queryRow p l i : Int)+a.prompt l.batch := by omega
    simp only [accumulator,expected,ih,denominator_eq,numerator_eq,hg]

theorem formula_eq (p : Layout) (l : Launch) (a : Inputs) (scale : ℝ) (i d : Nat)
    (hBN : 0 < p.BN) (hp : 0 ≤ a.prompt l.batch) :
    Math.formula p l a scale i d = expected p l a scale i d (count p l a) := by
  unfold Math.formula
  rw [agrees _ _ _ _ _ hBN (by omega)]
  exact accumulator_eq p l a scale i d _ hp

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ContextAttnBloom.Formula
