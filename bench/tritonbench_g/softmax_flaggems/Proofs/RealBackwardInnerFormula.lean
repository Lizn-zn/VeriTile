import bench.tritonbench_g.softmax_flaggems.Proofs.RealBackwardInnerIO

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
open VeriTile Triton
open scoped BigOperators

noncomputable def unpaddedExpected (s : BlockState) (P G : RegionName)
    (M N r n : Nat) : ℝ :=
  RealMath.backward N (fun j ↦ s.readMem P (offset N r j))
    (fun j ↦ s.readMem G (offset N r j)) n

def PaddingProductsVanish (s : BlockState) (P G : RegionName)
    (M N padded r : Nat) : Prop :=
  ∀ j, N≤j → j<padded →
    s.undef P (offset N r j) * s.undef G (offset N r j) = 0

theorem expected_eq_of_padding (s : BlockState) (P G : RegionName)
    (M N padded r n : Nat) (hpadlen : N≤padded) (hn : n<N) (hr : r<M)
    (hpad : PaddingProductsVanish s P G M N padded r) :
    expected s P G M N padded r n = unpaddedExpected s P G M N r n := by
  have he := RealMath.backwardPadded_eq_of_zero N (padded-N)
    (fun j ↦ sample s P M N r j) (fun j ↦ sample s G M N r j) n
    (by
      intro j hj
      have hnj : ¬N+j<N := by omega
      simp only [sample, hnj, and_false, ↓reduceIte]
      exact hpad (N+j) (by omega) (by omega))
  rw [Nat.add_sub_of_le hpadlen] at he
  unfold expected
  rw [he]
  unfold RealMath.backward unpaddedExpected RealMath.backward
  simp only [sample, hn, hr, and_self, ↓reduceIte]
  congr 2
  apply Finset.sum_congr rfl
  intro j hj
  simp [Finset.mem_range.mp hj]

theorem zero_undef_padding (s : BlockState) (P G : RegionName)
    (M N padded r : Nat) (hu : s.undef=fun _ _ ↦ 0) :
    PaddingProductsVanish s P G M N padded r := by
  intro j _ _
  simp [hu]

#guard_msgs (drop info) in
#auditModuleAxioms
end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealBackwardInner
