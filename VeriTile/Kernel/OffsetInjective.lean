/-
VeriTile.Math.OffsetInjective

Reusable output-offset injectivity lemmas shared by the bench GEMM / norm
kernels. Each kernel's store-correctness theorem needs the per-lane address map
to be injective (distinct lanes write distinct cells, so the memory readback
equals the spec). These addresses are affine in the tile coordinates; injectivity
follows from a one-glance geometric bound (column width ≤ row stride, or nonzero
stride) via Euclidean-division uniqueness — proved once here instead of per
kernel.
-/
import Mathlib.Algebra.Order.Group.Nat
import Mathlib.Tactic
import VeriTile.Core.Shape

namespace VeriTile

open VeriTile

/-- **1-D affine address injectivity.** `k ↦ C + k·stride` over `Fin I` is
injective as soon as the stride is nonzero. (Used by row-reduction kernels whose
output lane `k` maps to `base + k·stride_out`.) -/
theorem affine1D_inj {I : Nat} (C stride : Nat) (h : 0 < stride) :
    Function.Injective (fun k : Fin I => C + k.val * stride) := by
  intro a b hab
  simp only at hab
  have hmul : a.val * stride = b.val * stride := by omega
  exact Fin.ext (Nat.eq_of_mul_eq_mul_right h hmul)

/-- **2-D row-major address injectivity.** `(i,j) ↦ C + i·S + j` over a
`BLOCK_M × BLOCK_N` tile is injective whenever the column extent fits the row
stride (`J ≤ S`): the column offset `j < J ≤ S` never carries into the `i·S`
digit, so the map is a base-`S` encoding of `(i,j)`. `C` is the per-program
constant base (pid offsets, etc.), identical across lanes, so it cancels. -/
theorem rowMajor2D_inj {I J : Nat} (C S : Nat) (hJ : J ≤ S) :
    Function.Injective (fun idx : TileIndex [I, J] => C + idx.1.val * S + idx.2.1.val) := by
  intro a b h
  obtain ⟨ai, aj, au⟩ := a
  obtain ⟨bi, bj, bu⟩ := b
  simp only at h
  have ha : aj.val < S := lt_of_lt_of_le aj.isLt hJ
  have hb : bj.val < S := lt_of_lt_of_le bj.isLt hJ
  have hpos : 0 < S := by omega
  have h2 : ai.val * S + aj.val = bi.val * S + bj.val := by omega
  have l1 : (ai.val * S + aj.val) / S = ai.val := by
    rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt ha,
      Nat.zero_add]
  have l2 : (bi.val * S + bj.val) / S = bi.val := by
    rw [Nat.add_comm, Nat.mul_comm, Nat.add_mul_div_left _ _ hpos, Nat.div_eq_of_lt hb,
      Nat.zero_add]
  have hi : ai.val = bi.val := by rw [← l1, ← l2, h2]
  have hmul : ai.val * S = bi.val * S := by rw [hi]
  have hj : aj.val = bj.val := by omega
  cases au; cases bu
  rw [Fin.ext hi, Fin.ext hj]

end VeriTile
