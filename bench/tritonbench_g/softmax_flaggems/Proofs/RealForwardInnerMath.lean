import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardInnerLoop
import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerPrepare

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
open VeriTile Triton
open scoped BigOperators

noncomputable def denominator (s : BlockState) (X : RegionName) (row N : Nat) : ℝ :=
  ∑ n : Fin N, Real.exp (s.readMem X (row * N + n.val))

noncomputable def expected (s : BlockState) (X : RegionName) (row N n : Nat) : ℝ :=
  Real.exp (s.readMem X (row * N + n)) / denominator s X row N

def outputFootprint (Y : RegionName) (row N : Nat) (addr : RegionName × Nat) : Prop :=
  addr.1 = Y ∧ ∃ n, n < N ∧ addr.2 = row * N + n

noncomputable def globalMax {B : Nat} (hB : 0 < B)
    (mf : TileIndex [B] → WithBot ℝ) : WithBot ℝ :=
  Finset.univ.sup' (⟨⟨0, hB⟩, Finset.mem_univ _⟩ : (Finset.univ : Finset (Fin B)).Nonempty)
    (fun i ↦ mf (i, PUnit.unit))

noncomputable def globalSum {B : Nat} (mf : TileIndex [B] → WithBot ℝ)
    (zf : TileIndex [B] → ℝ) (mx : WithBot ℝ) : ℝ :=
  ∑ i : Fin B, zf (i, PUnit.unit) *
    (WithBot.realExp (WithBot.realSub (mf (i, PUnit.unit)) mx)).unbotD 0

noncomputable def finalMax (s : BlockState) (X : RegionName) (row N B : Nat) (hB : 0 < B) : WithBot ℝ :=
  globalMax hB (fun i ↦ (laneState s X row N B (RealAddress.rounds N B) i).1)

noncomputable def finalSum (s : BlockState) (X : RegionName) (row N B : Nat) (hB : 0 < B) : ℝ :=
  globalSum (fun i ↦ (laneState s X row N B (RealAddress.rounds N B) i).1)
    (fun i ↦ (laneState s X row N B (RealAddress.rounds N B) i).2) (finalMax s X row N B hB)

theorem final_denominator (s : BlockState) (X : RegionName) (row N B : Nat)
    (hN : 0 < N) (hB : 0 < B) :
    ∃ m : ℝ, finalMax s X row N B hB = some m ∧
      finalSum s X row N B hB = Real.exp (-m) * denominator s X row N := by
  have hd : denominator s X row N = RealForwardNonInner.denominator s X row N 1 0 := by
    simp only [denominator, RealForwardNonInner.denominator, Nat.mul_one, Nat.add_zero]
    exact Fin.sum_univ_eq_sum_range (fun n ↦ Real.exp (s.readMem X (row * N + n))) N
  have h := RealForwardNonInner.final_denominator s X row N 1 0 B 1 hN hB ⟨0, by omega⟩ (by omega)
  change ∃ m : ℝ, _ = some m ∧ _ = Real.exp (-m) * RealForwardNonInner.denominator s X row N 1 0 at h
  rw [← hd] at h
  simpa [finalMax, finalSum, globalMax, globalSum, laneState,
    RealForwardNonInner.finalMax, RealForwardNonInner.finalSum,
    RealForwardNonInner.columnMax, RealForwardNonInner.columnSum,
    RealForwardNonInner.laneState] using h

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardInner
