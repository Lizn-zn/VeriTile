import bench.tritonbench_g.softmax_flaggems.Proofs.RealAddress

namespace VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress

/-- Every active row belongs to one of the source's ceiling-numbered tiles. -/
theorem active_tile_lt {N B n : Nat} (hB : 0 < B) (hn : n < N) :
    n / B < rounds N B := by
  exact (Nat.div_lt_iff_lt_mul hB).mpr (Nat.lt_of_lt_of_le hn (coverage hB))

/-- The source's reverse base selects exactly the next quotient class. -/
theorem reverse_tile_quotient {N B c i : Nat} (hB : 0 < B) (hi : i < B) :
    (previous N B - c * B + i) / B = rounds N B - 1 - c := by
  rw [reverse_base]
  rw [Nat.add_comm, Nat.add_mul_div_right _ _ hB, Nat.div_eq_of_lt hi, Nat.zero_add]

theorem active_reverse_lane {N B c n : Nat} (hB : 0 < B)
    (hq : n / B = rounds N B - 1 - c) :
    ∃ i : Fin B, n = previous N B - c * B + i.val := by
  refine ⟨⟨n % B, Nat.mod_lt n hB⟩, ?_⟩
  rw [reverse_base, ← hq]
  simpa only [Nat.add_comm, Nat.mul_comm] using (Nat.mod_add_div n B).symm

/-- Rows already written by the reverse loop cannot alias the next tile. -/
theorem reverse_old_disjoint {N B c n i : Nat} (hB : 0 < B)
    (hc : c < rounds N B) (hold : rounds N B - c ≤ n / B) (hi : i < B) :
    n ≠ previous N B - c * B + i := by
  intro heq
  have hq := reverse_tile_quotient (N := N) (c := c) hB hi
  rw [← heq] at hq
  omega

theorem reverse_new_or_old {N B c n : Nat} (hc : c < rounds N B)
    (hnew : rounds N B - (c + 1) ≤ n / B) :
    n / B = rounds N B - 1 - c ∨ rounds N B - c ≤ n / B := by
  omega

end VeriTile.Bench.TritonBenchG.SoftmaxFlaggems.RealAddress
