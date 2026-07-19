/-
VeriTile.Triton.Semantics.LaneBridge

The row-major bridge between a flat `⊨`-skin lane space `Fin (M * N)` and a
2-D tile index `TileIndex [M, N]`.

Every `⊨` (KernelIO) skin carries its data channels on a flat lane index
`Fin B`, while many Triton kernels store a genuinely two-dimensional tile
`[BLOCK_M, BLOCK_N]`. The standard identification is row-major:

    lane j   ↦   (j / N, j % N)          (`Lane2D.decode`)
    (r, c)   ↦   r * N + c               (`Lane2D.encode`)

This module owns that mechanism once, with exactly the lemma set the kernel
ports need: the div/mod computation rules, both round-trips, injectivity, the
packaged `Equiv`, and the offset-arithmetic rule that lets a `base + row`
address be rewritten in terms of the flat lane.

### No positivity side-condition

`decode` needs `0 < N` to take `j % N`, and `encode`'s bound needs `0 < N`
too — but **neither takes it as a hypothesis**. A lane `j : Fin (M * N)`
inhabits `Fin (M * N)`, which is empty unless `0 < N`, so `j` itself supplies
the proof; symmetrically the column coordinate `idx.2.1 : Fin N` supplies it
on the encode side. Consumers therefore never need to thread a spurious
`0 < N` (or `0 < M`) hypothesis through their specs.
-/

import VeriTile.Triton.Core.Shape

namespace VeriTile.Triton

/-! ## `Lane2D`: the row-major lane ↔ 2-D tile index bridge for a `[M, N]`
tile flattened into `Fin (M * N)` lanes. -/
namespace Lane2D

variable {M N : Nat}

/-- `0 < N` is free from a lane of the flattened space: `Fin (M * N)` is empty
when `N = 0`. -/
theorem pos_of_lane (j : Fin (M * N)) : 0 < N :=
  Nat.pos_of_ne_zero fun h => absurd j.isLt (by simp [h])

/-- Row-major **decode**: the `[M, N]` tile cell of flat lane `j`, namely
`(j / N, j % N)`. -/
def decode (j : Fin (M * N)) : TileIndex [M, N] :=
  (⟨j.val / N, Nat.div_lt_of_lt_mul
      (Nat.lt_of_lt_of_le j.isLt (Nat.le_of_eq (Nat.mul_comm M N)))⟩,
   ⟨j.val % N, Nat.mod_lt _ (pos_of_lane j)⟩,
   PUnit.unit)

/-- Row-major **encode**: the flat lane of tile cell `idx`, namely
`idx.row * N + idx.col`. Inverse of `decode`. -/
def encode (idx : TileIndex [M, N]) : Fin (M * N) :=
  ⟨idx.1.val * N + idx.2.1.val, by
    calc idx.1.val * N + idx.2.1.val
        < idx.1.val * N + N := Nat.add_lt_add_left idx.2.1.isLt _
      _ = (idx.1.val + 1) * N := (Nat.succ_mul idx.1.val N).symm
      _ ≤ M * N := Nat.mul_le_mul_right N idx.1.isLt⟩

/-! ### Computation rules

`decode_row` / `decode_col` are `rfl`; `encode_val` is deliberately **not**
`@[simp]` so that `encode_div` / `encode_mod` keep firing on the `encode`-headed
form instead of racing an unfolded sum. -/

/-- The row coordinate of `decode j` is `j / N`. -/
@[simp] theorem decode_row (j : Fin (M * N)) : (decode j).1.val = j.val / N := rfl

/-- The column coordinate of `decode j` is `j % N`. -/
@[simp] theorem decode_col (j : Fin (M * N)) : (decode j).2.1.val = j.val % N := rfl

/-- Definitional value of `encode`. Not `@[simp]`: use `encode_div` /
`encode_mod` / `decode_encode` as the rewrite rules. -/
theorem encode_val (idx : TileIndex [M, N]) :
    (encode idx).val = idx.1.val * N + idx.2.1.val := rfl

/-- Commuted spelling of `encode_val`, for kernels whose local encode was
written `col + row * N`. -/
theorem encode_val_comm (idx : TileIndex [M, N]) :
    (encode idx).val = idx.2.1.val + idx.1.val * N :=
  Nat.add_comm _ _

/-- Dividing an encoded lane by the row length recovers the row. -/
@[simp] theorem encode_div (idx : TileIndex [M, N]) :
    (encode idx).val / N = idx.1.val := by
  have hc : idx.2.1.val < N := idx.2.1.isLt
  have hN : 0 < N := Nat.lt_of_le_of_lt (Nat.zero_le _) hc
  rw [encode_val, Nat.mul_comm, Nat.mul_add_div hN, Nat.div_eq_of_lt hc,
    Nat.add_zero]

/-- Reducing an encoded lane mod the row length recovers the column. -/
@[simp] theorem encode_mod (idx : TileIndex [M, N]) :
    (encode idx).val % N = idx.2.1.val := by
  rw [encode_val, Nat.mul_comm, Nat.mul_add_mod,
    Nat.mod_eq_of_lt idx.2.1.isLt]

/-! ### Round-trips and injectivity -/

/-- `decode` undoes `encode`. -/
@[simp] theorem decode_encode (idx : TileIndex [M, N]) :
    decode (encode idx) = idx :=
  Prod.ext (Fin.ext (encode_div idx)) (Prod.ext (Fin.ext (encode_mod idx)) rfl)

/-- `encode` undoes `decode`. -/
@[simp] theorem encode_decode (j : Fin (M * N)) : encode (decode j) = j :=
  Fin.ext <| by
    show j.val / N * N + j.val % N = j.val
    rw [Nat.mul_comm]
    exact Nat.div_add_mod j.val N

theorem encode_injective : Function.Injective (encode (M := M) (N := N)) :=
  Function.LeftInverse.injective decode_encode

theorem decode_injective : Function.Injective (decode (M := M) (N := N)) :=
  Function.LeftInverse.injective encode_decode

/-- The row-major flattening packaged as an equivalence
`Fin (M * N) ≃ TileIndex [M, N]`. -/
def equiv : Fin (M * N) ≃ TileIndex [M, N] where
  toFun := decode
  invFun := encode
  left_inv := encode_decode
  right_inv := decode_encode

@[simp] theorem equiv_apply (j : Fin (M * N)) : equiv j = decode j := rfl

@[simp] theorem equiv_symm_apply (idx : TileIndex [M, N]) :
    equiv.symm idx = encode idx := rfl

/-! ### Offset arithmetic

Kernels address a row-major tile as `(rowBase + row) * N + col`. These two
rules move that address between the lane view and the tile view. -/

/-- A row-offset 2-D address in the lane view collapses to a flat offset. -/
theorem rowBase_addr_decode (a : Nat) (j : Fin (M * N)) :
    (a + j.val / N) * N + j.val % N = a * N + j.val := by
  rw [Nat.add_mul, Nat.add_assoc, Nat.mul_comm (j.val / N) N, Nat.div_add_mod]

/-- A row-offset 2-D address in the tile view is the flat offset of its lane. -/
theorem rowBase_addr_encode (a : Nat) (idx : TileIndex [M, N]) :
    (a + idx.1.val) * N + idx.2.1.val = a * N + (encode idx).val := by
  rw [encode_val, Nat.add_mul, Nat.add_assoc]

end Lane2D

end VeriTile.Triton
