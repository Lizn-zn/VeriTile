import bench.tritonbench_g.block_sparse_attn.Proofs.CSRMath

/-! Relate an actual recurrence to the initial-array formula on one observed
row. Values on causally masked lanes may differ, and neither unobserved query
rows nor unvisited CSR entries are constrained. The separate execution proof
must supply the score and visible-value equalities below. -/
namespace VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealObservedFormula
open VeriTile.Triton
open scoped BigOperators
noncomputable section

/-- Equal visited scores and visible values suffice; arbitrary values on a
zero-weight lane never affect the real normalized result. -/
theorem accumulated_eq {BN : Nat} (left right : Nat → Fin BN → WithBot ℝ)
    (leftValue rightValue : Nat → Fin BN → ℝ) (k : Nat)
    (hs : ∀ n, n < k → ∀ j, left n j = right n j)
    (hv : ∀ n, n < k → ∀ j, right n j ≠ ⊥ → leftValue n j = rightValue n j) :
    StreamingMath.accumulated left leftValue k = StreamingMath.accumulated right rightValue k := by
  rw [StreamingMath.accumulated_eq_formula, StreamingMath.accumulated_eq_formula]
  have hmass : StreamingMath.mass left k = StreamingMath.mass right k := by
    apply Finset.sum_congr rfl
    intro n hn
    apply Finset.sum_congr rfl
    intro j _
    rw [hs n (Finset.mem_range.mp hn) j]
  have hmoment : StreamingMath.moment left leftValue k = StreamingMath.moment right rightValue k := by
    apply Finset.sum_congr rfl
    intro n hn
    have hn' := Finset.mem_range.mp hn
    apply Finset.sum_congr rfl
    intro j _
    rw [hs n hn' j]
    by_cases hb : right n j = ⊥
    · simp [hb]
    · rw [hv n hn' j hb]
  rw [hmass, hmoment]

/-- The stored row has the independent CSR attention value even if actual
masked loads do not equal the mathematical input array. -/
theorem attention (schedule : CSRMath.Schedule) (xs : CSRMath.Inputs)
    (BN BD blocks queryRow channel : Nat) (scale : ℝ)
    (actualScore : Nat → Fin BN → WithBot ℝ) (actualValue : Nat → Fin BN → ℝ)
    (hs : ∀ n, n < CSRMath.count schedule xs → ∀ j,
      actualScore n j = CSRMath.score schedule xs BN BD blocks queryRow scale n j)
    (hv : ∀ n, n < CSRMath.count schedule xs → ∀ j,
      CSRMath.selectedKey schedule xs BN n j ≤ (queryRow : Int) →
      actualValue n j = xs.value (CSRMath.selectedKey schedule xs BN n j) channel) :
    StreamingMath.accumulated actualScore actualValue (CSRMath.count schedule xs) =
      CSRMath.attention schedule xs BN BD blocks queryRow channel scale := by
  rw [accumulated_eq actualScore (CSRMath.score schedule xs BN BD blocks queryRow scale)
    actualValue (fun n j => xs.value (CSRMath.selectedKey schedule xs BN n j) channel)
    (CSRMath.count schedule xs) hs (by
      intro n hn j hscore
      apply hv n hn j
      by_contra h
      exact hscore (by simp [CSRMath.score, h]))]
  exact CSRMath.recurrence_eq_attention schedule xs BN BD blocks queryRow channel scale

#guard_msgs (drop info) in
#auditModuleAxioms
end
end VeriTile.Bench.TritonBenchG.BlockSparseAttn.RealObservedFormula
