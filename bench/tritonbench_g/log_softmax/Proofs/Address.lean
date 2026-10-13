import bench.tritonbench_g.log_softmax.Proofs.Real
import bench.tritonbench_g.softmax_flaggems.Proofs.RealAddress

namespace VeriTile.Bench.TritonBenchG.LogSoftmax.SourceAddress
open VeriTile.Triton

/-- Only active lanes need distinct addresses. This allows BLOCK_N to exceed N. -/
theorem active_injective {M N K R C p₀ p₁ : Nat} (hK : 0 < K)
    {i j : TileIndex [R, C]}
    (hi : active p₀ M N R i) (hj : active p₀ M N R j)
    (heq : outOffset p₀ p₁ N K R i = outOffset p₀ p₁ N K R j) : i = j := by
  have hmul : ((p₀ * R + i.1.val) * N + i.2.1.val) * K =
      ((p₀ * R + j.1.val) * N + j.2.1.val) * K := by
    simpa only [outOffset, Nat.add_mul, Nat.add_right_cancel_iff] using heq
  have hrow := Nat.eq_of_mul_eq_mul_right hK hmul
  obtain ⟨hm, hn⟩ := SoftmaxFlaggems.RealAddress.row_column_injective hi.2 hj.2 hrow
  apply Prod.ext
  · exact Fin.ext (Nat.add_left_cancel hm)
  · exact Prod.ext (Fin.ext hn) (Subsingleton.elim _ _)

/-- A valid inner program coordinate and active row/column give the actual bound. -/
theorem active_bound {M N K R C p₀ p₁ : Nat} (hk : p₁ < K)
    (i : TileIndex [R, C]) (hi : active p₀ M N R i) :
    outOffset p₀ p₁ N K R i < M * N * K := by
  exact SoftmaxFlaggems.RealAddress.non_inner_bound hi.1 hi.2 hk

end VeriTile.Bench.TritonBenchG.LogSoftmax.SourceAddress
