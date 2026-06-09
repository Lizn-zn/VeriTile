/-
VeriTile.Triton.Math.Matmul

Shared GEMM building blocks for the bench matmul kernels (`matmul_triton1`,
`matmul_kernel`, `matmul_tma`, …).

- `gemmSum` : the genuine matrix-product accumulator `Σ_{k < len} a k · b k`,
  parameterized by per-row / per-column element accessors `a, b : ℕ → ℝ`. Each
  kernel instantiates `a`/`b` with its own memory-layout readback; the algebra
  (block step, base case) is proved once here.
- `tile_dot_data` : evaluation of a single `tl.dot` output cell to the same
  finite sum — the bridge from the DSL `Tile.dot` op to `gemmSum`'s summand.
-/
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Tactic
import VeriTile.Triton.Semantics

namespace VeriTile.Triton

open VeriTile.Triton

/-- **Genuine GEMM dot-accumulator** over ℝ: `Σ_{k < len} a k · b k`. The kernels'
`matmulSpec` (`len = K`) and `accPartial … c` (`len = c · BLOCK_K`) are instances. -/
noncomputable def gemmSum (a b : Nat → ℝ) (len : Nat) : ℝ :=
  (Finset.range len).sum (fun k => a k * b k)

@[simp] theorem gemmSum_zero (a b : Nat → ℝ) : gemmSum a b 0 = 0 := by
  simp [gemmSum]

/-- **One block-step** of the accumulator: extending the contraction by one
`BLOCK_K`-block adds that block's `BLOCK_K`-key dot. This is the shared content of
every kernel's `accPartial_succ`. -/
theorem gemmSum_blockSucc (a b : Nat → ℝ) (BLOCK_K c : Nat) :
    gemmSum a b ((c + 1) * BLOCK_K)
      = gemmSum a b (c * BLOCK_K)
        + Finset.univ.sum
            (fun e : Fin BLOCK_K => a (c * BLOCK_K + e.val) * b (c * BLOCK_K + e.val)) := by
  unfold gemmSum
  have h : (c + 1) * BLOCK_K = c * BLOCK_K + BLOCK_K := by ring
  rw [h, Finset.sum_range_add]
  congr 1
  rw [Finset.sum_range (fun e => a (c * BLOCK_K + e) * b (c * BLOCK_K + e))]

/-- **`tl.dot` output-cell evaluation.** Given readbacks `fx`/`fy` for row `i` of
`x` and column `j` of `y`, the `(i,j)` cell of `Tile.dot x y` is `Σ_e fx e · fy e`.
Shared by every matmul kernel's per-iteration dot recipe. -/
theorem tile_dot_data (M K N : Nat) (x : Tile .real [M, K]) (y : Tile .real [K, N])
    (i : Fin M) (j : Fin N) (fx : Fin K → ℝ) (fy : Fin K → ℝ)
    (hx : ∀ e : Fin K, x.data (i, e, PUnit.unit) = some (fx e))
    (hy : ∀ e : Fin K, y.data (e, j, PUnit.unit) = some (fy e)) :
    (Tile.dot [] x y).data (i, j, PUnit.unit)
      = some (Finset.univ.sum fun e : Fin K => fx e * fy e) := by
  rw [Tile.dot_nil_data]
  rw [show (@Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ
        (fun e => Option.map₂ (· * ·) (x.data (i, e, PUnit.unit)) (y.data (e, j, PUnit.unit))))
      = @Finset.sum (Fin K) (WithBot ℝ) _ Finset.univ (fun e => (some (fx e * fy e) : WithBot ℝ))
      from Finset.sum_congr rfl (fun e _ => by rw [hx e, hy e]; rfl)]
  show (Finset.univ.sum fun e => ((fx e * fy e : ℝ) : WithBot ℝ)) = _
  rw [← WithBot.coe_sum]; rfl

end VeriTile.Triton
