import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardNonInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
open VeriTile Triton
open scoped BigOperators

noncomputable def unpaddedExpected (s : BlockState) (P G : RegionName)
    (row N K n k : Nat) : ℝ :=
  RealMath.backward N (fun j ↦ s.readMem P (offset row N K j k))
    (fun j ↦ s.readMem G (offset row N K j k)) n

def PaddingProductsVanish (s : BlockState) (P G : RegionName)
    (row N K padded k : Nat) : Prop :=
  ∀ j, N≤j → j<padded →
    s.undef P (offset row N K j k) * s.undef G (offset row N K j k) = 0

theorem expected_eq_of_padding (s : BlockState) (P G : RegionName)
    (row N K padded n k : Nat) (hpadlen : N≤padded) (hn : n<N) (hk : k<K)
    (hpad : PaddingProductsVanish s P G row N K padded k) :
    expected s P G row N K padded n k = unpaddedExpected s P G row N K n k := by
  have he := RealMath.backwardPadded_eq_of_zero N (padded-N)
    (fun j ↦ sample s P row N K j k) (fun j ↦ sample s G row N K j k) n
    (by
      intro j hj
      have hnj : ¬N+j<N := by omega
      simp only [sample, hnj, false_and, ↓reduceIte]
      exact hpad (N+j) (by omega) (by omega))
  rw [Nat.add_sub_of_le hpadlen] at he
  unfold expected
  rw [he]
  unfold RealMath.backward unpaddedExpected RealMath.backward
  simp only [sample, hn, hk, and_self, ↓reduceIte]
  congr 2
  apply Finset.sum_congr rfl
  intro j hj
  simp [Finset.mem_range.mp hj]

theorem zero_undef_padding (s : BlockState) (P G : RegionName)
    (row N K padded k : Nat) (hu : s.undef=fun _ _ ↦ 0) :
    PaddingProductsVanish s P G row N K padded k := by
  intro j _ _
  simp [hu]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardNonInner
