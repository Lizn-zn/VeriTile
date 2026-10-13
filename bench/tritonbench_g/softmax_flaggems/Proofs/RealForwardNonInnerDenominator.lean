import bench.tritonbench_g.softmax_flaggems.Proofs.RealForwardNonInnerReduce

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
open VeriTile Triton
open scoped BigOperators

noncomputable def denominator (s : BlockState) (X : RegionName) (row N K k : Nat) : ℝ :=
  ∑ n ∈ Finset.range N, Real.exp (s.readMem X (row * N * K + n * K + k))

noncomputable def expected (s : BlockState) (X : RegionName) (row N K n k : Nat) : ℝ :=
  Real.exp (s.readMem X (row * N * K + n * K + k)) / denominator s X row N K k

 theorem maximum_finite (s : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (hN : 0 < N) (hB : 0 < B) (hc : 0 < c)
    (k : Fin C) (hk : kbase + k.val < K) :
    columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1) k ≠ ⊥ := by
  have hzero : (laneState s X row N K kbase B C c (⟨0, hB⟩, k, PUnit.unit)).1 ≠ ⊥ := by
    simpa only [laneState, if_pos hk] using
      RealForwardModel.state_zero_lane_nonbot N B c
        (fun n ↦ s.readMem X (row * N * K + n * K + (kbase + k.val))) hN hc
  intro hbot
  have hle : (laneState s X row N K kbase B C c (⟨0, hB⟩, k, PUnit.unit)).1 ≤
      columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1) k := by
    unfold columnMax
    exact Finset.le_sup' (fun i : Fin B ↦
      (laneState s X row N K kbase B C c (i, k, PUnit.unit)).1)
      (Finset.mem_univ ⟨0, hB⟩)
  rw [hbot] at hle
  exact hzero (le_antisymm hle bot_le)

 theorem denominator_closed (s : BlockState) (X : RegionName)
    (row N K kbase B C c : Nat) (hN : 0 < N) (hB : 0 < B) (hc : 0 < c)
    (hcover : N ≤ c * B) (k : Fin C) (hk : kbase + k.val < K) :
    ∃ m : ℝ,
      columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1) k = some m ∧
      columnSum (fun i ↦ (laneState s X row N K kbase B C c i).1)
        (fun i ↦ (laneState s X row N K kbase B C c i).2)
        (columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1)) k =
        Real.exp (-m) * denominator s X row N K (kbase + k.val) := by
  have hn := maximum_finite s X row N K kbase B C c hN hB hc k hk
  obtain ⟨m, hm⟩ : ∃ m : ℝ,
      columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1) k = some m := by
    cases hcase : columnMax hB (fun i ↦ (laneState s X row N K kbase B C c i).1) k with
    | bot => exact (hn hcase).elim
    | coe m => exact ⟨m, rfl⟩
  refine ⟨m, hm, ?_⟩
  unfold columnSum
  rw [hm]
  apply RealMath.tiled_online_denominator N B c hB hcover
  · intro i
    simpa only [laneState, if_pos hk] using
      RealForwardModel.state_consistent N B i.val c
        (fun n ↦ s.readMem X (row * N * K + n * K + (kbase + k.val)))
  · intro i hi
    apply RealForwardModel.state_bot_mass N B i.val c
      (fun n ↦ s.readMem X (row * N * K + n * K + (kbase + k.val)))
    simpa only [laneState, if_pos hk] using hi

 theorem stored_value (x m d : ℝ) :
    (WithBot.realDiv (WithBot.realExp (WithBot.realSub (some x) (some m)))
      (some (Real.exp (-m) * d))).unbotD 0 = Real.exp x / d := by
  change Real.exp (x - m) / (Real.exp (-m) * d) = Real.exp x / d
  rw [Real.exp_sub, Real.exp_neg]
  by_cases hd : d = 0
  · simp [hd]
  · have hm := Real.exp_ne_zero m
    field_simp

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealForwardNonInner
