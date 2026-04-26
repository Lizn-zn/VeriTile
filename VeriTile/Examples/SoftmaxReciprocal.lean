/-
VeriTile.Examples.SoftmaxReciprocal

Real Triton optimization: replace `y = e/s` (per-element division) with
`inv_s = 1/s; y = e * inv_s` (one division total + one multiply per element).
Algorithmically equivalent in ℝ since e/s = e * (1/s) when s ≠ 0.

The "div" side of the comparison is `stableSoftmaxKernel` (already defined
in `SoftmaxEq.lean` and proven correct via `softmax_stable_correct`).
This file adds only the reciprocal-form kernel and the equivalence theorem.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Examples.SoftmaxEq

namespace VeriTile.Examples

open VeriTile.Triton

/-- Stable softmax with precomputed reciprocal. Saves N-1 divisions vs
    the per-element-divide form (`stableSoftmaxKernel`). -/
def softmaxRecipKernel (N : Nat) : Kernel := triton {
  pid    := tl.program_id(0)
  offs   := pid * $(N) + tl.arange($(N))
  x      := tl.load(X, offs)
  m      := tl.max(x)
  e      := tl.exp(x - m)
  s      := tl.sum(e)
  inv_s  := 1 / s
  y      := e * inv_s
  tl.store(Y, offs, y)
}

/-- The load-bearing math identity: division equals multiplication by
    reciprocal (for non-zero divisor). -/
theorem div_eq_mul_inv_real (a s : ℝ) (hs : s ≠ 0) : a / s = a * (1 / s) := by
  field_simp

/-- Closed-form spec for `softmaxRecipKernel`'s `Y[pid*N+i]` cell. -/
noncomputable def stableRecipSpec {N : Nat} (xs : Fin N → ℝ) (m : ℝ) (i : Fin N) : ℝ :=
  Real.exp (xs i - m) * (1 / ∑ j, Real.exp (xs j - m))

theorem softmax_recip_correct
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (_h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (softmaxRecipKernel N) s) N s.pid i
        = some (stableRecipSpec xs (tileMax hN xs) i) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  intro i
  have hcast :
      ∀ k : Fin (n + 1),
        realToNat ((↑s.pid : ℝ) * (↑(n + 1) : ℝ) + (↑(↑k : ℕ) : ℝ))
          = s.pid * (n + 1) + k.val := by
    intro k
    unfold realToNat
    have heq :
        ((↑s.pid : ℝ) * (↑(n + 1) : ℝ) + (↑(↑k : ℕ) : ℝ))
          = ((s.pid * (n + 1) + k.val : ℕ) : ℝ) := by
      push_cast; ring
    rw [heq]; exact Nat.floor_natCast _
  have h_inj : Function.Injective (fun k : Fin (n + 1) => s.pid * (n + 1) + k.val) := by
    intro a b hab
    exact Fin.ext (Nat.add_left_cancel hab)
  simp [observeY, exec, softmaxRecipKernel, stepStmts, stepStmt, evalOp,
        Value.bop, Value.uop, Value.reduceSum, Value.reduceMax,
        BlockState.setReg, BlockState.readMem,
        stableRecipSpec, tileMax]
  simp only [show ((n : ℝ) + 1) = ((n + 1 : ℕ) : ℝ) by push_cast; ring]
  unfold InputLoaded at _h_x
  simp_rw [hcast, _h_x]
  exact BlockState.scatter_readback _ _ _ h_inj i

/-- **Refinement: stable softmax with per-element division ≡ stable softmax
    with precomputed reciprocal.** Composes via `div_eq_mul_inv_real`. -/
theorem softmax_reciprocal_refinement
    (N : Nat) (hN : 0 < N) (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoaded s N xs) :
    ∀ i : Fin N,
      observeY (exec (stableSoftmaxKernel N) s) N s.pid i =
      observeY (exec (softmaxRecipKernel N) s) N s.pid i := by
  intro i
  rw [softmax_stable_correct N hN s xs h_x i,
      softmax_recip_correct N hN s xs h_x i]
  congr 1
  -- Goal: stableSpec xs (tileMax hN xs) i = stableRecipSpec xs (tileMax hN xs) i
  -- These differ only by a / b vs a * (1/b).
  unfold stableSpec stableRecipSpec
  have h_sum_pos : 0 < ∑ j, Real.exp (xs j - tileMax hN xs) := by
    apply Finset.sum_pos
    · intro j _; exact Real.exp_pos _
    · obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
      exact ⟨⟨0, hN⟩, Finset.mem_univ _⟩
  exact div_eq_mul_inv_real _ _ (ne_of_gt h_sum_pos)

end VeriTile.Examples
