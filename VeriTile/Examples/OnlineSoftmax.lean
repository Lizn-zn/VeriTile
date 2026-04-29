/-
VeriTile.Examples.OnlineSoftmax

Tier 2 kernel-pair (PAPER CENTERPIECE): online softmax recurrence ≡ batch
softmax. The streaming form

  m_0 = -∞,  l_0 = 0
  m_{k+1} = max(m_k, x_k)
  l_{k+1} = exp(m_k − m_{k+1}) · l_k + exp(x_k − m_{k+1})

produces the same (m, l) as the one-shot batch form

  m = max(x_0, ..., x_{N-1})
  l = Σ exp(x_i − m)

This is the algorithmic core of FlashAttention; Phase C will reuse the
recurrence at the kernel level.

For Phase B we use a degenerate "block size 1" form so that the loop
iterates over single elements, exposing the recurrence directly. Phase C's
FA kernel will instantiate this with full Bk-size blocks.
-/

import VeriTile.Triton.Core
import VeriTile.Triton.Semantics
import VeriTile.Triton.DSL
import VeriTile.Triton.LoopInvariant
import VeriTile.Examples.Common
import VeriTile.Examples.SoftmaxEq  -- reuse stableSoftmaxKernel + tileMax
import VeriTile.Examples.WelfordMath  -- reuse castFin helper

namespace VeriTile.Examples

open VeriTile.Triton

/-- Batch softmax kernel — same as the existing `stableSoftmaxKernel` from
    `Examples/SoftmaxEq.lean`. Restated here for clarity / direct reference. -/
def batchSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel :=
  stableSoftmaxKernel xReg yReg N

/-- Online softmax kernel: maintains `(m, l)` registers across single elements
    of the input tile. The Phase C FA kernel will lift this to KV-blocks. -/
def onlineSoftmaxKernel (xReg yReg : RegionName) (N : Nat) : Kernel := triton {
  pid := tl.program_id(0)
  m   := -inf    -- ℝ-channel sentinel (Op.negInf), finite stand-in for -∞
  l   := 0
  tl.for i in $(N) {
    xi    := tl.load($(xReg) + (pid * $(N) + i))
    m_new := tl.max(m, xi)
    l     := tl.exp(m - m_new) * l + tl.exp(xi - m_new)
    m     := m_new
  }
  -- Phase B focuses on (m, l) at the end. Phase C will add the per-element
  -- Y[i] = exp(x_i - m) / l divide.
}

/-- Online softmax math: streaming `m_k` after k iterations. -/
noncomputable def onlineSoftmaxM {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => -1e38
  | k + 1 =>
      if h : k < N then max (onlineSoftmaxM xs k) (xs ⟨k, h⟩)
      else onlineSoftmaxM xs k

/-- Online softmax math: streaming `l_k` after k iterations. -/
noncomputable def onlineSoftmaxL {N : Nat} (xs : Fin N → ℝ) : Nat → ℝ
  | 0     => 0
  | k + 1 =>
      if h : k < N then
        let m_old := onlineSoftmaxM xs k
        let m_new := onlineSoftmaxM xs (k + 1)
        Real.exp (m_old - m_new) * onlineSoftmaxL xs k + Real.exp (xs ⟨k, h⟩ - m_new)
      else onlineSoftmaxL xs k

/-- Batch softmax math. -/
noncomputable def batchSoftmaxM {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  tileMax hN xs  -- defined in SoftmaxEq.lean

noncomputable def batchSoftmaxL {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ) : ℝ :=
  ∑ i, Real.exp (xs i - batchSoftmaxM hN xs)

/-! ### Prefix lemmas (induction on `k`) -/

/-- Prefix form of the **max** recurrence: for any prefix length `j+1 ≤ N`,
    the running `onlineSoftmaxM xs (j+1)` matches the batch max over the
    first `j+1` inputs.

    The hypothesis `h_lo` is required because the seed value `M_0 = -1e38`
    is a finite stand-in for `-∞` (see `Op.negInf` in `Triton.Semantics`).
    Without it, e.g. at `k = 1` the recurrence would give
    `max (-1e38) (xs 0)` which only equals `xs 0` when `xs 0 ≥ -1e38`. -/
private theorem online_softmax_prefix_M {N : Nat} (xs : Fin N → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    ∀ j : Nat, ∀ (hk : j+1 ≤ N),
      onlineSoftmaxM xs (j+1) = (Finset.univ : Finset (Fin (j+1))).sup'
        Finset.univ_nonempty
        (fun i => xs (castFin hk i)) := by
  intro j
  induction j with
  | zero =>
    intro hk
    have hj_lt : 0 < N := hk
    have hM : onlineSoftmaxM xs 1 = max (-1e38) (xs ⟨0, hj_lt⟩) := by
      show (if h : 0 < N then max (onlineSoftmaxM xs 0) (xs ⟨0, h⟩) else _) = _
      simp [hj_lt, onlineSoftmaxM]
    rw [hM]
    have hsup : (Finset.univ : Finset (Fin 1)).sup' Finset.univ_nonempty
        (fun i => xs (castFin hk i)) = xs ⟨0, hj_lt⟩ := by rfl
    rw [hsup]
    exact max_eq_right (h_lo ⟨0, hj_lt⟩)
  | succ j ih =>
    intro hk
    have hj : j+1 ≤ N := Nat.le_of_succ_le hk
    have hj_lt : j+1 < N := hk
    have hM : onlineSoftmaxM xs (j+1+1) =
        max (onlineSoftmaxM xs (j+1)) (xs ⟨j+1, hj_lt⟩) := by
      show (if h : j+1 < N then max (onlineSoftmaxM xs (j+1)) (xs ⟨j+1, h⟩) else _) = _
      simp [hj_lt]
    rw [hM, ih hj]
    rw [Finset.sup'_congr Finset.univ_nonempty
          (Fin.univ_castSuccEmb (j+1)) (fun _ _ => rfl)]
    rw [Finset.sup'_cons (H := by simp)]
    rw [Finset.sup'_map]
    simp only [Function.comp, Fin.castSuccEmb_apply, max_comm]
    rfl

/-- Prefix form of the **sum** recurrence: for any prefix length `k ≤ N`,
    the running `onlineSoftmaxL xs k` equals the batch-style sum
    `∑ i, exp(xs i − onlineSoftmaxM xs k)` restricted to the first `k`
    inputs.  No hypothesis on the magnitude of `xs` is needed here — only
    the M-side identity at `k = N` requires it. -/
private theorem online_softmax_prefix_L {N : Nat} (xs : Fin N → ℝ) :
    ∀ k : Nat, ∀ (hk : k ≤ N),
      onlineSoftmaxL xs k =
        ∑ i : Fin k, Real.exp (xs (castFin hk i) - onlineSoftmaxM xs k) := by
  intro k
  induction k with
  | zero =>
    intro _
    show (0 : ℝ) = _
    simp
  | succ j ih =>
    intro hk
    have hj : j ≤ N := Nat.le_of_succ_le hk
    have hj_lt : j < N := hk
    have ih' := ih hj
    have hL : onlineSoftmaxL xs (j+1) =
        Real.exp (onlineSoftmaxM xs j - onlineSoftmaxM xs (j+1)) * onlineSoftmaxL xs j
        + Real.exp (xs ⟨j, hj_lt⟩ - onlineSoftmaxM xs (j+1)) := by
      show (if h : j < N then
          Real.exp (onlineSoftmaxM xs j - onlineSoftmaxM xs (j+1)) * onlineSoftmaxL xs j
          + Real.exp (xs ⟨j, h⟩ - onlineSoftmaxM xs (j+1)) else _) = _
      simp [hj_lt]
    rw [hL, ih']
    rw [Finset.mul_sum]
    rw [Fin.sum_univ_castSucc (n := j)
          (f := fun i => Real.exp (xs (castFin hk i) - onlineSoftmaxM xs (j+1)))]
    have h1 : ∀ i : Fin j,
        castFin hk i.castSucc = castFin hj i := fun _ => rfl
    have h2 : (castFin hk (Fin.last j) : Fin N) = ⟨j, hj_lt⟩ := rfl
    simp_rw [h1, h2]
    congr 1
    apply Finset.sum_congr rfl
    intros i _
    rw [← Real.exp_add]
    ring_nf

/-- **The math identity (paper centerpiece)**: the online recurrence at depth
    `N` produces the same `(m, l)` as the batch form.

    The hypothesis `h_lo : ∀ i, -1e38 ≤ xs i` is necessary because Phase B's
    `Op.negInf` is the finite stand-in `-1e38` rather than the IEEE `-∞`.
    Without this assumption, the recurrence's first step
    `M_1 = max(-1e38, xs 0)` could disagree with the batch `M = xs 0`
    when `xs 0 < -1e38`.  Phase C will replace `negInf` with a proper
    `⊥` sentinel and drop the hypothesis. -/
theorem online_softmax_recurrence_eq_batch
    {N : Nat} (hN : 0 < N) (xs : Fin N → ℝ)
    (h_lo : ∀ i, (-1e38 : ℝ) ≤ xs i) :
    onlineSoftmaxM xs N = batchSoftmaxM hN xs ∧
    onlineSoftmaxL xs N = batchSoftmaxL hN xs := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_succ_of_ne_zero hN.ne'
  have hcastFin_id : ∀ i : Fin (n+1),
      castFin (le_refl (n+1)) i = i := fun _ => rfl
  have hM_eq : onlineSoftmaxM xs (n+1) = batchSoftmaxM hN xs := by
    rw [online_softmax_prefix_M xs h_lo n (le_refl _)]
    simp_rw [hcastFin_id]
    rfl
  refine ⟨hM_eq, ?_⟩
  rw [online_softmax_prefix_L xs (n+1) (le_refl _), hM_eq]
  simp_rw [hcastFin_id]
  rfl

/-- Operational correctness: the online softmax kernel computes (m, l)
    matching `onlineSoftmaxM xs N` and `onlineSoftmaxL xs N`. -/
theorem online_softmax_correct
    (xReg yReg : RegionName) (N : Nat) (hN : 0 < N)
    (s : BlockState) (xs : Fin N → ℝ)
    (h_x : InputLoadedAt s xReg N xs) :
    let final := exec (onlineSoftmaxKernel xReg yReg N) s
    final.bind (fun s' => s'.regs "m" >>= Value.asScalar)
        = some (onlineSoftmaxM xs N)
    ∧ final.bind (fun s' => s'.regs "l" >>= Value.asScalar)
        = some (onlineSoftmaxL xs N) := by
  sorry

end VeriTile.Examples
