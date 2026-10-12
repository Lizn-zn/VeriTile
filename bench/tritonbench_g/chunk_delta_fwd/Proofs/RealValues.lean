import bench.tritonbench_g.chunk_delta_fwd.Proofs.RealInnerLoop

/-! Relate the source's masked, padded tiles to independent logical matrices.
Full key ownership requires i_k = 0 and K ≤ BK, not BK ≤ K. Values in padded
key lanes do not affect the K-term correction. -/
namespace VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealValues
open VeriTile.Triton RealProgram
open scoped BigOperators
noncomputable section
set_option maxHeartbeats 1000000
set_option linter.unusedSimpArgs false

def arrays (p : Layout) (l : Launch) (a : Inputs) (seed : Nat → Nat → ℝ) : RealMath.Inputs where
  key := fun t e => a.key (l.head * p.keyHead + e * p.keyDim + t * p.keyTime)
  delta := fun t e => a.delta (l.head * p.keyHead + t * p.keyTime + e * p.keyDim)
  value := fun t j => a.value (l.head * p.valueHead + t * p.valueTime + j * p.valueDim)
  seed := seed

private theorem padded_sum (K BK : Nat) (h : K ≤ BK) (f : Nat → ℝ) :
    (∑ e : Fin BK, if e.val < K then f e.val else 0) = ∑ e : Fin K, f e.val := by
  rw [Fin.sum_univ_eq_sum_range (fun e => if e < K then f e else 0) BK,
    Fin.sum_univ_eq_sum_range f K, ← Finset.sum_filter]
  have hs : (Finset.range BK).filter (fun e => e < K) = Finset.range K := by
    ext e
    simp only [Finset.mem_filter, Finset.mem_range]
    omega
  rw [hs]

/-- The public logical state is needed only on active key/value lanes. -/
def Represents (p : Layout) (l : Launch) (tile : StateValue p) (state : Nat → Nat → ℝ) : Prop :=
  ∀ (e : Fin p.BK) (j : Fin p.BV), e.val < p.K → valueIndex p l j.val < p.V →
    tile e j = state e.val (valueIndex p l j.val)

theorem corrected_eq (p : Layout) (l : Launch) (a : Inputs) (seed state : Nat → Nat → ℝ)
    (frozen : StateValue p) (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hf : Represents p l frozen state) (t : Nat) (j : Fin p.BV)
    (ha : activeValue p l t j.val) :
    correctedValue p l a frozen t j =
      RealMath.corrected (arrays p l a seed) p.K state t (valueIndex p l j.val) := by
  unfold correctedValue RealMath.corrected
  congr 1
  · simp only [inputValue, ha, ite_true, arrays, valueAddress]
  · calc
      _ = ∑ e : Fin p.BK, if e.val < p.K then
          (arrays p l a seed).delta t e.val * state e.val (valueIndex p l j.val) else 0 := by
        apply Finset.sum_congr rfl
        intro e _
        by_cases he : e.val < p.K
        · simp only [deltaValue, keyIndex, hk, Nat.zero_mul, Nat.zero_add, ha.1, he,
            and_self, ite_true, hf e j he ha.2, arrays, deltaAddress]
        · simp only [deltaValue, keyIndex, hk, Nat.zero_mul, Nat.zero_add, he, and_false,
            ite_false, zero_mul]
      _ = _ := padded_sum p.K p.BK hK
        (fun e => (arrays p l a seed).delta t e * state e (valueIndex p l j.val))

/-- Under the actual BC-divides-BT geometry, the completed inner sum is one
BT-row mathematical update, including a partial final time chunk. -/
theorem accumulated_eq (p : Layout) (l : Launch) (a : Inputs) (seed state : Nat → Nat → ℝ)
    (outer : Nat) (frozen acc : StateValue p) (hk : l.key = 0) (hK : p.K ≤ p.BK)
    (hBC : 0 < p.BC) (hdiv : p.BC ∣ p.BT) (hf : Represents p l frozen state)
    (e : Fin p.BK) (j : Fin p.BV) (he : e.val < p.K) (hj : valueIndex p l j.val < p.V) :
    RealInnerLoop.accumulated p l a outer (RealInnerLoop.count p) frozen acc e j =
      acc e j + ∑ c : Fin p.BT, if outer * p.BT + c.val < p.T then
        (arrays p l a seed).key (outer * p.BT + c.val) e.val *
          RealMath.corrected (arrays p l a seed) p.K state (outer * p.BT + c.val) (valueIndex p l j.val)
      else 0 := by
  rw [RealInnerLoop.accumulated_flat]
  have hspan : RealInnerLoop.count p * p.BC = p.BT :=
    RealMath.complete_inner_span p.BT p.BC hBC hdiv
  rw [hspan]
  congr 1
  apply Finset.sum_congr rfl
  intro c _
  by_cases ht : outer * p.BT + c.val < p.T
  · rw [corrected_eq p l a seed state frozen hk hK hf _ j ⟨ht,hj⟩]
    simp only [keyValue, keyIndex, hk, Nat.zero_mul, Nat.zero_add, he, ht, and_self,
      ite_true, keyAddress, arrays]
  · simp only [keyValue, ht, and_false, ite_false, zero_mul]

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.ChunkDeltaFwd.RealValues
