/-
VeriTile.Semantics.Offset

Offset abstractions and tile-index injectivity helpers.

`linearOffset` names the classic Triton 1D block-strided store/load address
`pid · stride + lane` (`pid * stride + tl.arange(0, BLOCK)`). The injectivity
helpers below discharge the recurring `Function.Injective` obligations that
arise when reducing `BlockState.scatter_readback_prop_masked_nd` over 1D/2D
tile indices; keeping them beside `linearOffset` co-locates the offset
abstraction with the lemmas about it. The readback semantics themselves stay
in `Semantics.State`.
-/

import VeriTile.Semantics.State

namespace VeriTile

namespace BlockState

/-! ### Common 1D offset-injectivity helpers

These cover the recurring `Function.Injective` proof obligations that arise
when reducing `BlockState.scatter_readback_prop_masked_nd` over a 1D
block-local index. Without them, every bench proof rebuilds the same
3-line proof. -/

/-- The 1D block-local offset `idx ↦ base + idx.1.val` is injective. -/
theorem tileIndex1d_base_offset_injective {BLOCK : Nat} (base : Nat) :
    Function.Injective
      (fun idx : TileIndex [BLOCK] => base + idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ h
  exact Prod.ext (Fin.ext (Nat.add_left_cancel h)) rfl

/-- The 1D block-local strided offset `idx ↦ base + idx.1.val * stride` is
injective when `stride ≠ 0`. -/
theorem tileIndex1d_base_strided_offset_injective {BLOCK : Nat}
    (base stride : Nat) (h_stride : stride ≠ 0) :
    Function.Injective
      (fun idx : TileIndex [BLOCK] => base + idx.1.val * stride) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ h
  have hval : a.val * stride = b.val * stride := Nat.add_left_cancel h
  have heq : a.val = b.val := Nat.eq_of_mul_eq_mul_right (Nat.pos_of_ne_zero h_stride) hval
  exact Prod.ext (Fin.ext heq) rfl

/-- The bare 1D block-local offset `idx ↦ idx.1.val` is injective. -/
theorem tileIndex1d_offset_injective {BLOCK : Nat} :
    Function.Injective
      (fun idx : TileIndex [BLOCK] => idx.1.val) := by
  rintro ⟨a, _⟩ ⟨b, _⟩ h
  exact Prod.ext (Fin.ext h) rfl

/-! ### 2D tile-index injectivity helpers

The 2D analogue of the 1D helpers above, for the recurring offset form
`fun idx : TileIndex [M, N] => base + idx.1.val * Nstride + idx.2.1.val`. -/

/-- The 2D block-local offset `idx ↦ base + idx.1.val * Nstride + idx.2.1.val`
is injective when `Nstride > N` (so the row coordinate `idx.1.val` separates
distinct rows in the offset value, given that `idx.2.1.val < N`). -/
theorem tileIndex2d_base_row_major_injective
    {M N : Nat} (base Nstride : Nat) (h_lt : N ≤ Nstride) :
    Function.Injective
      (fun idx : TileIndex [M, N] =>
        base + idx.1.val * Nstride + idx.2.1.val) := by
  rintro ⟨⟨a1, ha1⟩, ⟨a2, ha2⟩, _⟩ ⟨⟨b1, hb1⟩, ⟨b2, hb2⟩, _⟩ h
  -- After subtracting `base`, we have `a1 * Nstride + a2 = b1 * Nstride + b2`
  -- with `a2 < N ≤ Nstride` and `b2 < N ≤ Nstride`. Standard div/mod argument.
  simp only at h
  have hN_pos : 0 < N := lt_of_le_of_lt (Nat.zero_le _) ha2
  have hStride_pos : 0 < Nstride := lt_of_lt_of_le hN_pos h_lt
  have ha2' : a2 < Nstride := lt_of_lt_of_le ha2 h_lt
  have hb2' : b2 < Nstride := lt_of_lt_of_le hb2 h_lt
  -- Rewrite associativity then cancel.
  have h' : base + (a1 * Nstride + a2) = base + (b1 * Nstride + b2) := by
    rw [← Nat.add_assoc, ← Nat.add_assoc]; exact h
  have hkey : a1 * Nstride + a2 = b1 * Nstride + b2 := Nat.add_left_cancel h'
  -- a1 and a2 are the "div" and "mod" of LHS by Nstride; similarly for b.
  have ha_div : (a1 * Nstride + a2) / Nstride = a1 := by
    rw [Nat.add_comm (a1 * Nstride) a2, Nat.mul_comm a1 Nstride,
        Nat.add_mul_div_left _ _ hStride_pos, Nat.div_eq_of_lt ha2',
        Nat.zero_add]
  have hb_div : (b1 * Nstride + b2) / Nstride = b1 := by
    rw [Nat.add_comm (b1 * Nstride) b2, Nat.mul_comm b1 Nstride,
        Nat.add_mul_div_left _ _ hStride_pos, Nat.div_eq_of_lt hb2',
        Nat.zero_add]
  have ha_mod : (a1 * Nstride + a2) % Nstride = a2 := by
    rw [Nat.add_comm (a1 * Nstride) a2, Nat.mul_comm a1 Nstride,
        Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ha2']
  have hb_mod : (b1 * Nstride + b2) % Nstride = b2 := by
    rw [Nat.add_comm (b1 * Nstride) b2, Nat.mul_comm b1 Nstride,
        Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb2']
  have h1 : a1 = b1 := by rw [← ha_div, hkey, hb_div]
  have h2 : a2 = b2 := by rw [← ha_mod, hkey, hb_mod]
  subst h1; subst h2; rfl

/-- The 2D block-local offset with separate row/col strides:
`idx ↦ base + idx.1.val * Mstride + idx.2.1.val * Nstride`, injective when
`Nstride ≠ 0` and `N * Nstride ≤ Mstride` (so the row coordinate dominates
the column contribution). -/
theorem tileIndex2d_base_strided_injective {M N : Nat}
    (base Mstride Nstride : Nat)
    (hN_pos : 0 < Nstride) (h_lt : N * Nstride ≤ Mstride) :
    Function.Injective
      (fun idx : TileIndex [M, N] =>
        base + idx.1.val * Mstride + idx.2.1.val * Nstride) := by
  rintro ⟨⟨a1, ha1⟩, ⟨a2, ha2⟩, _⟩ ⟨⟨b1, hb1⟩, ⟨b2, hb2⟩, _⟩ h
  simp only at h
  -- The offsets equal, so cancel base then derive a1=b1, a2=b2.
  have h' : a1 * Mstride + a2 * Nstride = b1 * Mstride + b2 * Nstride := by
    have := Nat.add_left_cancel
      (show base + (a1 * Mstride + a2 * Nstride) =
            base + (b1 * Mstride + b2 * Nstride) by
        rw [← Nat.add_assoc, ← Nat.add_assoc]; exact h)
    exact this
  -- a2 * Nstride < N * Nstride ≤ Mstride and similarly for b2.
  have ha2' : a2 * Nstride < Mstride :=
    lt_of_lt_of_le ((Nat.mul_lt_mul_right hN_pos).mpr ha2) h_lt
  have hb2' : b2 * Nstride < Mstride :=
    lt_of_lt_of_le ((Nat.mul_lt_mul_right hN_pos).mpr hb2) h_lt
  have hM_pos : 0 < Mstride := lt_of_le_of_lt (Nat.zero_le _) ha2'
  -- a1 = (a1 * Mstride + a2 * Nstride) / Mstride
  have ha_div : (a1 * Mstride + a2 * Nstride) / Mstride = a1 := by
    rw [Nat.add_comm (a1 * Mstride) (a2 * Nstride), Nat.mul_comm a1 Mstride,
        Nat.add_mul_div_left _ _ hM_pos, Nat.div_eq_of_lt ha2', Nat.zero_add]
  have hb_div : (b1 * Mstride + b2 * Nstride) / Mstride = b1 := by
    rw [Nat.add_comm (b1 * Mstride) (b2 * Nstride), Nat.mul_comm b1 Mstride,
        Nat.add_mul_div_left _ _ hM_pos, Nat.div_eq_of_lt hb2', Nat.zero_add]
  have h1 : a1 = b1 := by rw [← ha_div, h', hb_div]
  -- Substitute a1 = b1 and cancel.
  subst h1
  have h2 : a2 * Nstride = b2 * Nstride := by
    have := h'
    omega
  have h2eq : a2 = b2 := Nat.eq_of_mul_eq_mul_right hN_pos h2
  subst h2eq; rfl

end BlockState

/-- Classic 1D block-strided store/load address for a single program axis:
lane `i` of program `pid` addresses `pid · stride + i`. This is the most common
Triton 1D offset (`pid * stride + tl.arange(0, BLOCK)`). The logical row length
`stride` and the (possibly padded) lane bound `B` are independent. -/
def linearOffset (s : BlockState) (stride : Nat) {B : Nat} (i : Fin B) : Nat :=
  s.pid * stride + i.val

end VeriTile
